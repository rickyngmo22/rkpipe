#pragma once

// M13 动作识别运行时（EventEngine 同模式：开源实现、闭源核心在 pose 输出路径调用）。
// pose 跟踪结果逐帧喂入 PoseSequenceBuffer（时序骨架窗，见 postprocess/pose_sequence.h），
// 攒满窗口逐轨迹 ST-GCN 推理（action_interval 节流），结果以 ActionTaskResult 换入
// frame.result（内嵌 pose 数据，M0 CompositeClsTaskResult 惯例），可选叠加动作标签。
//
// 生命周期与 EventEngine 一致：随流水线构造（AppConfig 注入四件套：
// action_model_path/action_labels/action_window_t/action_interval），输出线程逐帧 update。
// 模型加载失败降级为 inactive（告警一次，流水线行为退回纯 pose）。

#include <string>
#include <unordered_map>
#include <vector>

#include "config/app_config.h"
#include "core/pipeline_frame.h"
#include "core/rknn_context.h"
#include "postprocess/pose_sequence.h"

class ActionRuntime {
public:
    explicit ActionRuntime(const AppConfig& options);
    ~ActionRuntime();

    ActionRuntime(const ActionRuntime&) = delete;
    ActionRuntime& operator=(const ActionRuntime&) = delete;

    bool active() const { return active_; }
    const std::vector<std::string>& labels() const { return labels_; }

    // pose 输出路径每帧调用（tracking 之后；未激活时 no-op）。
    // 喂缓冲 → 攒满窗逐轨迹推理 → frame.result 换为 ActionTaskResult（每帧换，
    // 动作级联激活期间 pose 帧结果类型恒定）；overlay_enabled 且有新鲜动作时
    // 在 frame.matFrame 上标注动作标签（与 EventEngine.drawOverlay 同惯例）。
    void update(PipelineFrame& frame, bool overlay_enabled);

private:
    bool active_ = false;
    PoseSequenceBuffer buffer_;
    double interval_s_ = 0.0;
    std::vector<std::string> labels_;
    rknn_app_context_t action_ctx_{};
    bool model_loaded_ = false;
    std::unordered_map<int, long long> last_infer_ms_;  // track_id → 上次推理 steady ms
};
