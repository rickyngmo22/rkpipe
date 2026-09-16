#include "daemon/llm_review.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <utility>
#include <vector>

#include "daemon/daemon_internal.h"
#include "daemon/mini_json.h"
#include "llm/llm_provider.h"
#include "utils/json_escape.h"

namespace {

std::string envStr(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
}

int envInt(const char* name, int def) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return def;
    }
    const int n = std::atoi(v);
    return n > 0 ? n : def;
}

bool readFileBytes(const std::string& path, std::string* out, size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return !out->empty() && out->size() <= max_bytes;
}

// 从模型回复 content 里抠出 JSON 结论（容忍 markdown 代码围栏与前后说明文字）
bool parseVerdictFromContent(const std::string& content, std::string* verdict, std::string* reason,
                             double* confidence) {
    const size_t b = content.find('{');
    const size_t e = content.rfind('}');
    if (b == std::string::npos || e == std::string::npos || e <= b) {
        return false;
    }
    bool ok = false;
    mini_json::JsonValue v = mini_json::JsonParser(content.substr(b, e - b + 1)).parse(ok);
    if (!ok || v.type != mini_json::JsonValue::Type::Object) {
        return false;
    }
    *verdict = v.getString("verdict");
    *reason = v.getString("reason");
    *confidence = v.getNumber("confidence", 0.0);
    return *verdict == "true" || *verdict == "false" || *verdict == "uncertain";
}

}  // namespace

LlmReviewer::Config LlmReviewer::fromEnv() {
    Config c;
    const std::string base_url = envStr("RK_PIPE_LLM_BASE_URL");
    c.llm.api_key = envStr("RK_PIPE_LLM_API_KEY");
    c.llm.model = envStr("RK_PIPE_LLM_MODEL");
    c.extra_prompt = envStr("RK_PIPE_LLM_EXTRA_PROMPT");
    c.timeout_ms = envInt("RK_PIPE_LLM_TIMEOUT_MS", 20000);
    c.concurrency = envInt("RK_PIPE_LLM_MAX_CONCURRENCY", 2);
    c.rate_per_min = std::atoi(envStr("RK_PIPE_LLM_RATE_PER_MIN").c_str());  // 0=不限
    if (c.rate_per_min < 0) {
        c.rate_per_min = 30;
    }
    c.queue_depth = envInt("RK_PIPE_LLM_QUEUE_DEPTH", 64);
    c.drop_false = envStr("RK_PIPE_LLM_DROP_FALSE") == "1";
    c.enabled = !base_url.empty() && !c.llm.model.empty();
    if (!c.enabled) {
        c.concurrency = 0;
    }
    if (c.concurrency > 8) {
        c.concurrency = 8;
    }
    // provider 侧配置：OpenAI 兼容 cloud；endpoint 由 base_url 归一补全 /chat/completions
    c.llm.provider = "cloud";
    c.llm.enabled = c.enabled;
    c.llm.endpoint = base_url;
    while (!c.llm.endpoint.empty() && c.llm.endpoint.back() == '/') {
        c.llm.endpoint.pop_back();
    }
    if (c.enabled && (c.llm.endpoint.size() < std::string("/chat/completions").size() ||
                      c.llm.endpoint.rfind("/chat/completions") !=
                          c.llm.endpoint.size() - std::string("/chat/completions").size())) {
        c.llm.endpoint += "/chat/completions";
    }
    c.llm.timeout_s = static_cast<double>(c.timeout_ms) / 1000.0;
    c.llm.temperature = 0.1;
    c.llm.max_tokens = 300;
    c.llm.send_image = true;
    return c;
}

LlmReviewer::LlmReviewer(Config cfg, const std::string& tmp_dir, EventStore* store,
                         std::function<void(long, const std::string&)> forward_cb)
    : cfg_(std::move(cfg)), tmp_dir_(tmp_dir), store_(store), forward_cb_(std::move(forward_cb)) {
    if (!cfg_.enabled) {
        return;
    }
    provider_ = llm::createProvider(cfg_.llm);
    last_call_ = std::chrono::steady_clock::now() - std::chrono::hours(1);
    for (int i = 0; i < cfg_.concurrency; ++i) {
        workers_.emplace_back(&LlmReviewer::workerLoop, this);
    }
    std::fprintf(stderr, "[daemon][llm] reviewer started: model=%s provider=%s concurrency=%d rate=%d/min timeout=%dms\n",
                 cfg_.llm.model.c_str(), provider_->name(), cfg_.concurrency, cfg_.rate_per_min, cfg_.timeout_ms);
}

LlmReviewer::~LlmReviewer() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
    }
    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    delete provider_;
    provider_ = nullptr;
    if (!queue_.empty()) {
        std::fprintf(stderr, "[daemon][llm] dropped %zu pending review job(s) on shutdown\n", queue_.size());
    }
}

bool LlmReviewer::submit(long event_id, const std::string& event_body_json) {
    if (!cfg_.enabled) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.size() >= static_cast<size_t>(cfg_.queue_depth)) {
            ++dropped_;
            std::fprintf(stderr, "[daemon][llm][warn] queue full (depth=%d), dropped %ld review(s)\n",
                         cfg_.queue_depth, dropped_.load());
            return false;
        }
        queue_.push_back(Job{event_id, event_body_json});
    }
    queue_cv_.notify_one();
    return true;
}

void LlmReviewer::workerLoop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&] { return stop_.load() || !queue_.empty(); });
            if (stop_.load()) {
                return;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        callVlm(job);
    }
}

void LlmReviewer::callVlm(const Job& job) {
    // 限速：按最小间隔错开（多 worker 共享同一节拍）
    if (cfg_.rate_per_min > 0) {
        const auto min_interval = std::chrono::milliseconds(60000 / cfg_.rate_per_min);
        std::chrono::steady_clock::time_point target;
        {
            std::lock_guard<std::mutex> lock(rate_mutex_);
            target = last_call_ + min_interval;
            const auto now = std::chrono::steady_clock::now();
            last_call_ = target > now ? target : now;
        }
        const auto now = std::chrono::steady_clock::now();
        if (target > now) {
            std::this_thread::sleep_for(target - now);
        }
    }

    // 告警现场抓图（子进程已先存图后发 webhook；缺失/超限时退化为纯文本复核）
    std::vector<unsigned char> snapshot_jpeg;
    std::string rule_hint;
    {
        bool ok = false;
        mini_json::JsonValue body = mini_json::JsonParser(job.body).parse(ok);
        if (ok && body.type == mini_json::JsonValue::Type::Object) {
            // 规则级业务提示（event_rules 的 llm_hint，随事件上报）
            rule_hint = body.getString("llm_hint");
            const mini_json::JsonValue* snap = body.get("snapshot");
            if (snap && snap->type == mini_json::JsonValue::Type::String && !snap->str.empty()) {
                std::string raw;
                if (readFileBytes(snap->str, &raw, 8ull << 20)) {
                    snapshot_jpeg.assign(raw.begin(), raw.end());
                } else {
                    std::fprintf(stderr, "[daemon][llm] snapshot unavailable, text-only review: %s\n",
                                 snap->str.c_str());
                }
            }
        }
    }

    // 经统一 provider 抽象（OpenAI 兼容 chat/completions，libcurl 直连）执行复核
    llm::LlmRequest req;
    req.system = buildSystemPrompt();
    req.prompt = buildUserPrompt(job.body, cfg_.extra_prompt, rule_hint);
    req.image_jpeg = std::move(snapshot_jpeg);
    const llm::LlmResponse resp = provider_->generate(req);

    std::string verdict, reason, err;
    double confidence = 0.0;
    bool ok = false;
    if (resp.ok) {
        if (parseVerdictFromContent(resp.text, &verdict, &reason, &confidence)) {
            ok = true;
        } else {
            err = "unexpected reply: " + resp.text.substr(0, 120);
        }
    } else {
        err = "vlm failed: " + resp.error.substr(0, 160);
    }

    if (store_) {
        EventStore::Review review;
        review.source = "llm";
        review.confidence = confidence;
        review.ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
        if (ok) {
            review.verdict = verdict;
            review.reason = reason;
        } else {
            review.verdict = "error";
            review.reason = err;
        }
        store_->attachReview(job.event_id, review);
        if (g_flywheel) {
            g_flywheel->onReview(job.event_id, job.body, review);
        }
    }

    if (ok && cfg_.drop_false && verdict == "false") {
        std::fprintf(stderr, "[daemon][llm] event %ld verdict=false, forward dropped (%s)\n", job.event_id,
                     reason.c_str());
        return;
    }
    if (forward_cb_) {
        // 结论增补进 body 尾部（body 以 '}' 结尾）；复检失败 fail-open：原样转发
        std::string out_body = job.body;
        if (ok && !out_body.empty() && out_body.back() == '}') {
            out_body.pop_back();
            std::ostringstream llm;
            llm << ",\"llm\":{\"verdict\":\"" << verdict << "\",\"confidence\":" << confidence
                << ",\"reason\":\"" << jsonEscape(reason) << "\"}";
            out_body += llm.str() + "}";
        }
        forward_cb_(job.event_id, out_body);
    }
}

std::string LlmReviewer::buildSystemPrompt() {
    return "你是安防监控告警复核助手。根据提供的告警现场画面与事件数据，判断该告警是否真实。"
           "只输出一个 JSON 对象，不要输出其他内容："
           "{\"verdict\":\"true|false|uncertain\",\"reason\":\"一句话中文理由\",\"confidence\":0.0到1.0}。"
           "verdict=true 表示画面与告警相符（告警可信），false 表示误报（画面与告警明显不符），"
           "uncertain 表示画面无法判断。";
}

std::string LlmReviewer::buildUserPrompt(const std::string& event_body_json,
                                         const std::string& extra_prompt,
                                         const std::string& rule_hint) {
    std::ostringstream p;
    p << "告警事件数据(JSON)：" << event_body_json << "\n"
      << "规则语义：line_cross=目标穿越绊线(画面中应有目标正经过绊线)；"
      << "intrusion=目标进入禁区(画面中应有目标位于警戒区域内)；"
      << "dwell=目标在禁区滞留超时(画面中应有目标停留在区域内)；"
      << "absence=离岗(本应有人的区域内持续无人，画面中区域应为空)；"
      << "near_distance=目标距离过近；count=目标数量超阈值。\n";
    if (!extra_prompt.empty()) {
        p << "业务补充说明：" << extra_prompt << "\n";
    }
    if (!rule_hint.empty()) {
        p << "该规则的业务提示：" << rule_hint << "\n";
    }
    p << "请结合画面判断该告警是否真实。";
    return p.str();
}
