#pragma once

// 事件规则引擎（D1 绊线客流 / D2 区域入侵·滞留·离岗 的公共底座）。
// detect + tracking 输出线程侧逐帧评估，产出结构化规则事件，由 AlertRuntime 上报。
//
// 规则（yaml event_* 配置，见 app_config.h）：
//   line_cross  - 目标轨迹穿越绊线（方向 A->B / B->A，内置双向计数）
//   intrusion   - 目标进入事件区域（每次进入触发）
//   dwell       - 目标在区域内持续超过 event_dwell_seconds（每次进入只触发一次）
//   absence     - 区域出现过人后持续无人超过 event_absence_seconds（离岗，触发一次）
//
// 纯逻辑 + OpenCV core，无硬件依赖，CI 单测可覆盖（几何/状态机全部可测）。

#include <chrono>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "detection/simple_object_tracker.h"

struct RuleEvent {
    std::string type;    // "line_cross" | "intrusion" | "dwell" | "absence"
    int track_id = 0;    // absence（区域级事件）为 0
    int cls_id = -1;
    std::string detail;  // "A->B"/"B->A" 或 滞留/离岗秒数
};

class EventEngine {
public:
    explicit EventEngine(const class AppConfig& options);

    bool active() const { return active_; }
    // 规则需要绊线/区域几何；缺几何的规则在构造时剔除并打一次告警日志
    bool lineCrossEnabled() const { return line_cross_enabled_; }
    bool intrusionEnabled() const { return intrusion_enabled_; }
    bool dwellEnabled() const { return dwell_enabled_; }
    bool absenceEnabled() const { return absence_enabled_; }

    const std::vector<cv::Point>& line() const { return line_; }
    const std::vector<cv::Point>& region() const { return region_; }
    // 绊线双向计数（演示画面用；absence/入侵为事件流）
    long lineCrossAB() const { return cross_ab_; }
    long lineCrossBA() const { return cross_ba_; }

    // 每输出帧调用一次（须在 tracker.update 之后传入本帧 tracked；即使空列表也要喂，
    // absence 计时与轨迹老化依赖连续帧）。
    std::vector<RuleEvent> update(const std::vector<TrackedDetection>& tracked,
                                  std::chrono::steady_clock::time_point now);

    // 绊线/区域可视化（黄色绊线 + 半透明青色区域 + 双向计数）
    void drawOverlay(cv::Mat& frame) const;

private:
    struct TrackState {
        cv::Point2f prev_center{0.0f, 0.0f};
        bool has_prev = false;
        bool in_region = false;
        std::chrono::steady_clock::time_point enter_time{};
        bool dwell_fired = false;
        std::chrono::steady_clock::time_point last_seen{};
        int cls_id = -1;
    };

    bool allowedClass(int cls_id) const;
    bool pointInRegion(const cv::Point2f& p) const;

    bool active_ = false;
    bool line_cross_enabled_ = false;
    bool intrusion_enabled_ = false;
    bool dwell_enabled_ = false;
    bool absence_enabled_ = false;
    std::vector<cv::Point> line_;    // 恰两点
    std::vector<cv::Point> region_;  // >=3 点
    std::unordered_set<int> classes_;
    double dwell_seconds_ = 10.0;
    double absence_seconds_ = 30.0;
    double stale_seconds_ = 5.0;  // 轨迹超过该时长未见即清除状态

    std::unordered_map<int, TrackState> tracks_;
    long cross_ab_ = 0;
    long cross_ba_ = 0;

    // 区域级占用状态（absence 规则）
    bool region_ever_occupied_ = false;
    bool absence_fired_ = false;
    bool region_empty_timing_ = false;
    std::chrono::steady_clock::time_point region_empty_since{};
};
