#pragma once

// 事件存储（daemon 侧告警/事件闭环，对应路线图 T4.4）：
//   - 子进程经 POST /internal/event 上报事件 → 分配自增 id → JSONL 落盘 + 内存环形缓冲
//   - LLM/人工复检结论挂接到事件（一并落盘，重启后随文件回放恢复）
//   - GET /api/events 按 task/type/since/limit 查询
//
// 文件格式（每行一个 JSON，body 为子进程原样上报内容）：
//   {"id":1,"recv_ts":1690000000000,"body":{"ts":...,"task":"detect",...}}
//   {"kind":"review","event_id":1,"source":"llm","verdict":"true","reason":"...","confidence":0.9,"ts":...}
// 文件超限轮转为 .1（单文件，仅保留一代）。

#include <deque>
#include <condition_variable>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>

#include "daemon/mini_json.h"

class EventStore {
public:
    // 复检结论（LLM 或人工）
    struct Review {
        std::string verdict;   // true / false / uncertain / error
        std::string reason;    // 一句话理由或错误信息
        double confidence = 0.0;
        std::string source;    // llm / manual
        int64_t ts_ms = 0;
    };

    struct Item {
        long id = 0;
        int64_t recv_ts_ms = 0;
        mini_json::JsonValue body;  // 子进程上报的原始 JSON（对象）
        bool has_review = false;
        Review review;
    };

    // file_path 为空 = 只用内存环形缓冲
    EventStore(const std::string& file_path, size_t mem_capacity, size_t file_max_bytes);

    // 追加事件（body 须为完整 JSON 对象文本），返回分配的 id；非法 body 返回 -1
    long append(const std::string& body);
    // 挂接复检结论（内存 + 落盘）；事件已不在环形缓冲时仅落盘并返回 false
    bool attachReview(long event_id, const Review& review);
    // 查询：task=body.task，type=body.event 或 body.rule_type；since=仅 id 更大者；
    // limit>0 时返回最近的 limit 条（按 id 升序返回）
    std::vector<Item> query(const std::string& task, const std::string& type, long since, int limit) const;
    bool get(long id, Item* out) const;
    size_t size() const;

    // ---- SSE 支持（/api/events/stream）----
    // 单调版本号：append / attachReview 均递增
    long version() const;
    // 最近一次被挂接复检结论的事件 id（无则 0）——SSE 增量推送 review 帧用
    long lastReviewEventId() const;
    // 阻塞直到版本号 != known 或超时；返回当前版本号
    long waitVersion(long known, int timeout_ms) const;

private:
    void loadTailLocked();
    void appendLineLocked(const std::string& line);
    static bool matchType(const mini_json::JsonValue& body, const std::string& type);

    std::string file_path_;
    size_t mem_capacity_;
    size_t file_max_bytes_;

    mutable std::mutex mutex_;
    mutable std::condition_variable version_cv_;
    long version_ = 0;
    long last_review_event_id_ = 0;
    std::deque<Item> items_;
    long next_id_ = 1;
};
