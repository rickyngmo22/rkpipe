#pragma once

// 误报→训练样本 飞轮（daemon 侧）：复检结论为 false（LLM 或人工）时，
// 把事件现场快照归档为训练候选样本 + manifest.jsonl 元数据。
// 上游用 tools/flywheel/export_yolo.py 打包为标注工程（数据集骨架）。
//
// 目录结构：
//   <root>/manifest.jsonl                     一行一样本（见 buildFlywheelRecord）
//   <root>/<task>/<type>/<event_id>_<ms>.jpg  快照副本（原快照目录受 GC 治理，此处独立保存）
//
// 配置：--flywheel-dir <dir> 或环境变量 RK_PIPE_FLYWHEEL_DIR；空=禁用（零开销）。

#include <string>

#include "daemon/events_store.h"

// 纯逻辑：由事件 body 与复检结论构造归档记录（不碰文件系统，可单测）。
// ok=false 表示无可归档内容（非 false 结论 / 无快照字段 / 非法 body）。
struct FlywheelRecord {
    bool ok = false;
    std::string rel_dir;      // <task>/<type>
    std::string filename;     // <event_id>_<ts_ms>.jpg
    std::string snapshot_src; // 事件 body 里的原始快照路径
    std::string manifest_json;
};

FlywheelRecord buildFlywheelRecord(long event_id,
                                   const std::string& body_json,
                                   const EventStore::Review& review);

class EventFlywheel {
public:
    // root 为空 = 禁用
    explicit EventFlywheel(std::string root);

    bool enabled() const { return !root_.empty(); }

    // 复检挂接后调用；verdict=false 且 body 带快照时归档（IO 失败仅打印告警，不影响主链路）
    void onReview(long event_id, const std::string& body_json, const EventStore::Review& review) const;

private:
    std::string root_;
};
