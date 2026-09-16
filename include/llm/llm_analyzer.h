#pragma once

// 检测结果 → LLM 异步分析器。
//
// 线程模型：流水线输出线程只调 submit()/observe()（入队 + 原子计数，绝不阻塞），
// 独立 worker 线程消费队列做 JPEG 编码与 LLM 调用（秒级网络往返不拖累输出帧率）；
// chat() 由 web 客户端线程同步调用。
//
// 进程级单例：闭源核心逐帧回调 finalizePipelineFrameOutput（开源函数），该回调没有
// 用户上下文指针可用，故用惰性初始化的进程级实例在挂接点与 web 端点间共享；
// 刻意泄漏（永不析构），避免退出期静态析构顺序与流水线线程的竞争。

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "core/pipeline_frame.h"
#include "llm/llm_config.h"
#include "llm/llm_provider.h"

namespace llm {

// worker 待处理任务
struct AnalysisJob {
    long long id = 0;
    int frame_index = 0;
    std::string source;
    std::string detections_json;   // 提交时已序列化的检测结果
    cv::Mat image_bgr;             // 提交时已 clone 的全帧（worker 内缩放/编码）
    bool with_image = false;
};

// 一条分析结论（内存缓存 + JSONL 落盘 + /api/llm/status 展示）
struct AnalysisRecord {
    long long id = 0;
    long long timestamp_ms = 0;
    int frame_index = 0;
    std::string source;
    std::string detections_json;
    bool ok = false;
    std::string verdict;   // ok 时为模型输出文本
    std::string error;     // !ok 时为错误描述
    double elapsed_ms = 0.0;
};

class LlmAnalyzer {
public:
    explicit LlmAnalyzer(const LlmConfig& config);
    ~LlmAnalyzer();

    void start();
    void stop();

    bool enabled() const { return enabled_; }

    // 逐帧累计观测（不做限速/过滤，统计口径完整）；非 detect 任务为空操作
    void observe(const TaskResult& result);

    // 复检提交：限速（interval_s）+ 阈值门控（min_conf/submit_all）内部完成。
    // 检测结果用模块自带序列化（detection_json）；
    // frame.matFrame 为空（零拷贝帧）时退化为纯 JSON 复检，非空且 provider 支持视觉时
    // 按需 clone 截图（缓冲生命周期与调用方解耦）。非阻塞。
    void submit(const PipelineFrame& frame);

    // 同步问答（/api/llm/chat）：问题 + 检测统计上下文 → 单轮生成
    LlmResponse chat(const std::string& question, const std::string& stats_context_json);

    // 状态与最近结论 JSON（/api/llm/status）
    std::string statusJson() const;

private:
    void workerLoop();
    AnalysisRecord runJob(AnalysisJob& job);
    void persistRecord(const AnalysisRecord& record);
    bool passesGate(const TaskResult& result) const;

    LlmConfig config_;
    std::unique_ptr<LlmProvider> provider_;
    bool enabled_ = false;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<AnalysisJob> queue_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    // 熔断（可靠性包2）：provider 连续失败 N 次 → 开断 M 秒（期间提交的帧
    // 直接记失败、不发起请求）；到时半开探测一次，成功即闭合。
    // 背景：llm.yaml 端点不可达/挂起时每条事件都陪等 timeout_s，退出路径
    // 也被拉长（本机 8080 rkllm 残留服务曾致偶发卡死）。
    std::atomic<int> consecutive_failures_{0};
    std::atomic<long long> circuit_open_until_ms_{0};  // steady 时钟 ms（跨线程读）
    std::atomic<int> circuit_skip_count_{0};
    std::atomic<long long> last_submit_ms_{-1};  // 提交限速（steady_clock 毫秒）

    mutable std::mutex records_mutex_;
    std::deque<AnalysisRecord> records_;
    long long next_job_id_ = 1;
    std::map<std::string, long long> class_counts_;  // observe 累计（类别名 -> 次数）
    std::atomic<std::uint64_t> frames_observed_{0};

    std::atomic<std::uint64_t> submitted_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> failed_{0};

    std::mutex file_mutex_;
    std::ofstream results_file_;
};

// 进程级单例（挂接点与 web 端点共享）。globalAnalyzer() 可能为 null（未配置/未启用）。
LlmAnalyzer* globalAnalyzer();
void initGlobalAnalyzer();  // 惰性初始化（幂等）：按 fromDefaultLocation 发现配置，enabled 才创建

}  // namespace llm
