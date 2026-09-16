#include "llm/llm_analyzer.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>
#include <turbojpeg.h>

#include "llm/detection_json.h"
#include "llm/llm_config.h"
#include "postprocess/postprocess.h"

namespace llm {

namespace {

constexpr std::size_t kMaxRecentRecords = 20;  // /api/llm/status 展示的最近结论条数

long long steadyNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

long long wallNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// JPEG 编码统一走 turbojpeg（NEON SIMD，且是项目既有依赖，CI 构建同样可用），
// 避免引入 cv::imencode 的 imgcodecs 链接依赖。失败返回空（当次退化为纯 JSON 复检）。
std::vector<unsigned char> encodeJpeg(const cv::Mat& bgr, int quality) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return {};
    }
    tjhandle compressor = tjInitCompress();
    if (!compressor) {
        return {};
    }
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    std::vector<unsigned char> jpeg;
    if (tjCompress2(compressor, bgr.data, bgr.cols, static_cast<int>(bgr.step), bgr.rows,
                    TJPF_BGR, &jpeg_buf, &jpeg_size, TJSAMP_420, quality, TJFLAG_FASTDCT) == 0) {
        jpeg.assign(jpeg_buf, jpeg_buf + jpeg_size);
    }
    tjFree(jpeg_buf);
    tjDestroy(compressor);
    return jpeg;
}

// 复检提示词：要求一行式中文结论。
// 说明：结构化 JSON 输出（{"verdict":...}）对云端大模型可用，但 0.6B~3B 级板端
// 模型实测会复读输入数据跑偏，故统一收敛为自然语言结论（llm_report 按关键词归类）。
constexpr const char* kRecheckSystem =
    "你是一名工业视觉质检助手。用户给出目标检测模型的输出摘要(JSON)和现场截图。"
    "请判断这些检出目标是否像真实缺陷，用中文按以下格式回答，"
    "不要输出 JSON、不要复述数据：\n结论：真实缺陷 或 疑似误报\n理由：一句话（60字内）";

}  // namespace

LlmAnalyzer::LlmAnalyzer(const LlmConfig& config) : config_(config) {
    provider_.reset(createProvider(config_));
    enabled_ = true;
    std::printf("[Llm] analyzer enabled: provider=%s model=%s min_conf=%.2f interval=%.1fs%s\n",
                config_.provider.c_str(), config_.model.c_str(), config_.min_conf, config_.interval_s,
                config_.results_path.empty() ? ""
                                             : (" results=" + config_.results_path).c_str());
}

LlmAnalyzer::~LlmAnalyzer() {
    stop();
}

void LlmAnalyzer::start() {
    if (running_.exchange(true)) {
        return;
    }
    if (!config_.results_path.empty()) {
        std::lock_guard<std::mutex> lock(file_mutex_);
        results_file_.open(config_.results_path, std::ios::app);
    }
    worker_ = std::thread(&LlmAnalyzer::workerLoop, this);
}

void LlmAnalyzer::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) {
        // join 保证结论完整落盘、无对象生命周期竞争（detach 曾引入退出期
        // ESRCH abort）。代价：worker 若阻塞在 provider 请求中，退出最多等
        // llm_timeout_s——属 llm.yaml 指向不可达服务的配置错误场景，应修配置。
        worker_.join();
    }
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (results_file_.is_open()) {
        results_file_.close();
    }
}

void LlmAnalyzer::observe(const TaskResult& result) {
    const object_detect_result_list* detect = getDetectResultList(result);
    if (!detect) {
        return;  // PoC 口径：累计统计只覆盖 detect 任务（工业缺陷检测主场景）
    }
    frames_observed_.fetch_add(1);
    std::lock_guard<std::mutex> lock(records_mutex_);
    for (int i = 0; i < detect->count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const object_detect_result& r = detect->results[i];
        char* name = coco_cls_to_name(r.cls_id);
        ++class_counts_[(name && *name) ? name : ("class_" + std::to_string(r.cls_id))];
    }
}

bool LlmAnalyzer::passesGate(const TaskResult& result) const {
    if (config_.submit_all) {
        return true;
    }
    const object_detect_result_list* detect = getDetectResultList(result);
    if (!detect) {
        return false;
    }
    for (int i = 0; i < detect->count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        if (detect->results[i].prop < config_.min_conf) {
            return true;  // 存在 borderline 检出 → 需要复检
        }
    }
    return false;
}

void LlmAnalyzer::submit(const PipelineFrame& frame) {
    if (!enabled_ || !running_.load() || !passesGate(frame.result)) {
        return;
    }

    // 限速：距上次提交不足 interval_s 直接跳过（原子读，无锁快路径）
    const long long now_ms = steadyNowMs();
    const long long last = last_submit_ms_.load();
    if (last >= 0 &&
        now_ms - last < static_cast<long long>(config_.interval_s * 1000.0)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (last_submit_ms_.load() >= 0 &&
            now_ms - last_submit_ms_.load() <
                static_cast<long long>(config_.interval_s * 1000.0)) {
            return;  // 并发下二次确认（多输出线程场景）
        }
        last_submit_ms_.store(now_ms);
    }

    // 门控通过才序列化/克隆（输出线程，微秒级）；必须在帧缓冲释放前完成。
    // 零拷贝帧 matFrame 为空 → 退化为纯 JSON 复检。
    AnalysisJob job;
    job.frame_index = frame.index;
    job.source = frame.sourceName;
    job.detections_json = taskResultToJson(frame.result, {});
    const bool want_image = config_.send_image && provider_->supportsVision();
    if (want_image && !frame.matFrame.empty()) {
        job.image_bgr = frame.matFrame.clone();
        job.with_image = true;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.size() >= static_cast<std::size_t>(config_.queue_size)) {
            queue_.pop_front();  // 丢最旧：LLM 分析是增值层，宁可少不可堵
            dropped_.fetch_add(1);
        }
        job.id = next_job_id_++;
        queue_.push_back(std::move(job));
        submitted_.fetch_add(1);
    }
    queue_cv_.notify_one();
}

LlmResponse LlmAnalyzer::chat(const std::string& question, const std::string& stats_context_json) {
    // 完整 statusJson 有数 KB，小参数量板端模型消化不了（慢且跑偏），抽关键计数
    nlohmann::json compact = nlohmann::json::object();
    nlohmann::json parsed = nlohmann::json::parse(stats_context_json, nullptr, false);
    if (!parsed.is_discarded()) {
        if (parsed.contains("detections_by_class")) {
            compact["detections_by_class"] = parsed["detections_by_class"];
        }
        if (parsed.contains("counters")) {
            compact["recheck_counters"] = parsed["counters"];
        }
        if (parsed.contains("frames_observed")) {
            compact["frames_observed"] = parsed["frames_observed"];
        }
    }
    LlmRequest request;
    request.system =
        "你是一名工业质检系统助手。根据检测统计(JSON)用中文简洁回答（100字内），"
        "数据中没有的信息不要编造。";
    request.prompt = "检测统计：" + compact.dump() + "\n问题：" + question;
    return provider_->generate(request);
}

AnalysisRecord LlmAnalyzer::runJob(AnalysisJob& job) {
    AnalysisRecord record;
    record.id = job.id;
    record.frame_index = job.frame_index;
    record.source = job.source;
    record.detections_json = job.detections_json;
    record.timestamp_ms = wallNowMs();

    // LLM prompt 用紧凑摘要：小参数量模型对长 JSON 容易复读跑偏，
    // 只喂 borderline 条目（≤8 个）+ 计数；完整 schema v1 JSON 仍保留在
    // detections_json（JSONL 落盘 / 内存缓存）中。
    std::string prompt_data;
    nlohmann::json parsed = nlohmann::json::parse(job.detections_json, nullptr, false);
    if (!parsed.is_discarded() && parsed.contains("dets")) {
        nlohmann::json compact = {
            {"frame", parsed.value("frame", 0)},
            {"total_detections", parsed["dets"].size()},
            {"low_conf_threshold", config_.min_conf},
        };
        nlohmann::json low = nlohmann::json::array();
        for (const auto& d : parsed["dets"]) {
            if (d.value("score", 1.0) < config_.min_conf && low.size() < 8) {
                low.push_back(d);
            }
        }
        compact["low_conf_items"] = low;
        prompt_data = compact.dump();
    } else {
        prompt_data = job.detections_json;  // 非 detect 类型原样传递
    }

    LlmRequest request;
    request.system = kRecheckSystem;
    request.prompt = "目标检测输出摘要：\n" + prompt_data +
                     "\n请复检其中低置信度条目，只输出约定格式的 JSON 结论。";
    if (job.with_image && !job.image_bgr.empty()) {
        // 发送前缩放截图（默认长边 640），控制上传体积与时延
        cv::Mat scaled = job.image_bgr;
        const double max_side = std::max(job.image_bgr.cols, job.image_bgr.rows);
        if (config_.image_max_side > 0 && max_side > config_.image_max_side) {
            const double scale = config_.image_max_side / max_side;
            cv::resize(job.image_bgr, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        request.image_jpeg = encodeJpeg(scaled, 80);
    }

    LlmResponse response = provider_->generate(request);
    record.ok = response.ok;
    record.verdict = std::move(response.text);
    record.error = std::move(response.error);
    record.elapsed_ms = response.elapsed_ms;
    return record;
}

void LlmAnalyzer::persistRecord(const AnalysisRecord& record) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (!results_file_.is_open()) {
        return;
    }
    nlohmann::json json = {
        {"id", record.id},
        {"ts", record.timestamp_ms},
        {"frame_index", record.frame_index},
        {"source", record.source},
        {"ok", record.ok},
        {"elapsed_ms", record.elapsed_ms},
        {"detections", nlohmann::json::parse(record.detections_json,
                                             nullptr, false)},
    };
    if (record.ok) {
        json["verdict"] = record.verdict;
    } else {
        json["error"] = record.error;
    }
    results_file_ << json.dump() << "\n";
    results_file_.flush();
}

void LlmAnalyzer::workerLoop() {
    for (;;) {
        AnalysisJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&]() { return !running_.load() || !queue_.empty(); });
            if (!running_.load()) {
                return;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
        }

        // 熔断：连续 5 次失败 → 开断 5 分钟（期间入队 job 直接记失败）；
        // 到时放行一个探测 job，成功即闭合（可靠性包2）
        const auto now = std::chrono::steady_clock::now();
        const bool circuit_open =
            consecutive_failures_.load() >= 5 &&
            now.time_since_epoch().count() < circuit_open_until_ms_.load();
        if (circuit_open) {
            ++circuit_skip_count_;
            AnalysisRecord skip{};
            skip.id = job.id;
            skip.ok = false;
            skip.error = "circuit open: provider 连续失败，跳过复检";
            persistRecord(skip);
            {
                std::lock_guard<std::mutex> lock(records_mutex_);
                records_.push_back(skip);
                while (records_.size() > static_cast<std::size_t>(config_.max_results_keep)) {
                    records_.pop_front();
                }
            }
            failed_.fetch_add(1);
            continue;
        }

        AnalysisRecord record = runJob(job);
        if (record.ok) {
            completed_.fetch_add(1);
            if (consecutive_failures_.load() > 0) {
                std::printf("[Llm] provider 恢复（此前连续失败 %d 次）\n",
                            consecutive_failures_.load());
            }
            consecutive_failures_.store(0);
        } else {
            failed_.fetch_add(1);
            const int fails = consecutive_failures_.fetch_add(1) + 1;
            if (fails == 5) {
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(300);
                circuit_open_until_ms_.store(until.time_since_epoch().count());
                circuit_skip_count_.store(0);
                std::fprintf(stderr,
                             "[Llm][warn] provider 连续 5 次失败，熔断 5 分钟（期间事件照常流转，仅跳过复检）\n");
            }
            std::printf("[Llm] recheck #%lld failed: %s\n", record.id, record.error.c_str());
        }
        persistRecord(record);
        {
            std::lock_guard<std::mutex> lock(records_mutex_);
            records_.push_back(std::move(record));
            while (records_.size() > static_cast<std::size_t>(config_.max_results_keep)) {
                records_.pop_front();
            }
        }
    }
}

std::string LlmAnalyzer::statusJson() const {
    nlohmann::json recent = nlohmann::json::array();
    nlohmann::json by_class = nlohmann::json::object();
    const bool circuit_open =
        std::chrono::steady_clock::now().time_since_epoch().count() < circuit_open_until_ms_.load();
    {
        std::lock_guard<std::mutex> lock(records_mutex_);
        for (const auto& kv : class_counts_) {
            by_class[kv.first] = kv.second;
        }
        const std::size_t start = records_.size() > kMaxRecentRecords
                                      ? records_.size() - kMaxRecentRecords
                                      : 0;
        for (auto it = records_.begin() + start; it != records_.end(); ++it) {
            nlohmann::json item = {
                {"id", it->id},
                {"ts", it->timestamp_ms},
                {"frame_index", it->frame_index},
                {"source", it->source},
                {"ok", it->ok},
                {"elapsed_ms", it->elapsed_ms},
            };
            if (it->ok) {
                item["verdict"] = it->verdict;
            } else {
                item["error"] = it->error;
            }
            recent.push_back(std::move(item));
        }
    }
    nlohmann::json json = {
        {"enabled", enabled_},
        {"provider", config_.provider},
        {"model", config_.model},
        {"min_conf", config_.min_conf},
        {"interval_s", config_.interval_s},
        {"frames_observed", frames_observed_.load()},
        {"counters",
         {{"submitted", submitted_.load()},
          {"dropped", dropped_.load()},
          {"completed", completed_.load()},
          {"failed", failed_.load()}}},
        {"circuit",
         {{"consecutive_failures", consecutive_failures_.load()}, {"open", circuit_open}}},
        {"detections_by_class", by_class},
        {"recent", recent},
    };
    return json.dump();
}

// ---- 进程级单例 ---------------------------------------------------------------

namespace {

// initGlobalAnalyzer 惰性创建；刻意泄漏（永不析构），避免退出期
// 静态析构顺序与流水线/worker 线程的竞争
std::atomic<LlmAnalyzer*> g_analyzer{nullptr};

}  // namespace

LlmAnalyzer* globalAnalyzer() {
    return g_analyzer.load(std::memory_order_acquire);
}

void initGlobalAnalyzer() {
    static std::once_flag once;
    std::call_once(once, []() {
        LlmConfig config = LlmConfig::fromDefaultLocation();
        if (!config.enabled) {
            return;  // 未配置 llm.yaml：模块静默禁用
        }
        LlmAnalyzer* analyzer = new LlmAnalyzer(config);
        analyzer->start();
        g_analyzer.store(analyzer, std::memory_order_release);
        // 正常退出路径兜底：main 返回时先停 worker（join 在途推理），把结论落盘，
        // 避免 LLM 慢调用跨过进程生命周期导致记录丢失
        std::atexit([]() {
            if (LlmAnalyzer* a = g_analyzer.load(std::memory_order_acquire)) {
                a->stop();
            }
        });
    });
}

}  // namespace llm
