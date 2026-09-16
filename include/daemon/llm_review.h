#pragma once

// LLM 复检器（daemon 侧）：
//   - /internal/event 收到事件后入队，worker 经 llm::LlmProvider（OpenAI 兼容
//     /chat/completions，见 src/llm/）对「事件 JSON + 告警现场抓图」做真实性复核，
//     产出 verdict=true/false/uncertain
//   - 结论挂接到 EventStore（GET /api/events 可查），随后转发业务 webhook（body 增补 llm 字段）
//   - fail-open：VLM 超时/报错/队列满 → 原始事件照常转发，不阻断告警链路
//
// 配置全部走环境变量（api_key 不进 streams.json / 命令行）：
//   RK_PIPE_LLM_BASE_URL       OpenAI 兼容根地址，如 http://127.0.0.1:8000/v1（空=关闭复检；
//                              自动补 /chat/completions 后作为 provider endpoint）
//   RK_PIPE_LLM_API_KEY        Bearer token（本地网关可留空）
//   RK_PIPE_LLM_MODEL          模型名，如 qwen-vl-plus / glm-4v-flash
//   RK_PIPE_LLM_EXTRA_PROMPT   业务补充说明（可空），拼进 user prompt
//   RK_PIPE_LLM_TIMEOUT_MS     单次调用超时（默认 20000）
//   RK_PIPE_LLM_MAX_CONCURRENCY并发 worker 数（默认 2）
//   RK_PIPE_LLM_RATE_PER_MIN   每分钟最多调用次数（默认 30，0=不限）
//   RK_PIPE_LLM_QUEUE_DEPTH    复检队列深度（默认 64，满则丢弃该次复检）
//   RK_PIPE_LLM_DROP_FALSE     1=verdict 为 false 时不转发业务 webhook（默认 0，转发并带结论）
//
// HTTP 由 llm::LlmProvider（libcurl 直连）执行；未链 libcurl 的构建经
// RK_PIPE_LLM_NO_CLOUD 降级桩返回明确错误，复检失败照旧 fail-open。

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "daemon/events_store.h"
#include "llm/llm_config.h"
#include "llm/llm_provider.h"

class LlmReviewer {
public:
    struct Config {
        bool enabled = false;
        LlmConfig llm;            // provider 侧配置（provider 固定 cloud；endpoint 由 base_url 归一）
        std::string extra_prompt;
        int timeout_ms = 20000;
        int concurrency = 2;
        int rate_per_min = 30;
        int queue_depth = 64;
        bool drop_false = false;
    };

    static Config fromEnv();

    // store/forward_cb 可为空；tmp_dir 为 daemon dump 目录（预留给请求转储诊断）
    LlmReviewer(Config cfg, const std::string& tmp_dir, EventStore* store,
                std::function<void(long, const std::string&)> forward_cb);
    ~LlmReviewer();

    bool enabled() const { return cfg_.enabled; }
    // 入队复检；false=未启用或队满（调用方应 fail-open 直接转发）
    bool submit(long event_id, const std::string& event_body_json);

private:
    struct Job {
        long event_id = 0;
        std::string body;
    };

    void workerLoop();
    // 执行一次 VLM 复核；ok=false 时 err 说明原因（verdict 不可用）
    void callVlm(const Job& job);
    static std::string buildSystemPrompt();
    static std::string buildUserPrompt(const std::string& event_body_json, const std::string& extra_prompt,
                                       const std::string& rule_hint);

    Config cfg_;
    std::string tmp_dir_;
    EventStore* store_ = nullptr;
    std::function<void(long, const std::string&)> forward_cb_;
    llm::LlmProvider* provider_ = nullptr;  // createProvider 所有权归本类

    std::deque<Job> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};

    // 简单限速：相邻两次调用最小间隔
    std::mutex rate_mutex_;
    std::chrono::steady_clock::time_point last_call_{};
    std::atomic<long> dropped_{0};
};
