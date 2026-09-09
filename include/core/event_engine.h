#pragma once

// 事件规则引擎（D1 绊线 / D2 区域规则的公共底座，支持命名多规则）。
// detect + tracking 输出线程侧逐帧评估，产出结构化规则事件，由 AlertRuntime 上报。
//
// 两种配置形态：
//   1) 扁平键（向后兼容，app_config.h 的 event_* 字段）：event_line/event_line_cross/
//      event_region/event_intrusion/event_dwell/event_absence/event_classes ——
//      合成为一条匿名规则（旧行为逐帧一致）；
//   2) 命名多规则（环境变量 RK_PIPE_EVENT_RULES=<yaml 文件>，当前核心库即可用；
//      AppConfig 为布局耦合类型不可加字段，event_rules 配置键随核心库 Release 协同）：
//      注意——当前核心用自己的扁平键检查门控引擎评估，纯命名配置时引擎不被调用；
//      流水线内生效需同时给扁平键作"唤醒"（如 event_line_cross: 1 + 合法 event_line，
//      命名模式优先级更高，扁平合成被跳过，唤醒键本身不产生规则事件）。2026-09 板端
//      gdb 实证：无唤醒键断点 0 命中，带唤醒键 1800/1800 帧评估；告警派发随核心
//      后续版本提供。叠加可视化（web 预览）同样需唤醒键。
//        - {id: "gate",  type: "line_cross", line: "x1,y1,x2,y2", direction: "both|A2B|B2A",
//           ref_point: "center|bottom"}
//        - {id: "yard",  type: "intrusion",  region: "x1,y1,...", classes: "0"}
//        - {id: "loiter",type: "dwell",      region: "...", dwell_seconds: 10}
//        - {id: "post",  type: "absence",    region: "...", absence_seconds: 30}
//        - {id: "crowd", type: "crowd",      region: "...", min_count: 3}   区域内人数超限
//        - {id: "left",  type: "abandoned",  region: "...", seconds: 60}    区域内静止遗留物
//        - {id: "fall",  type: "fall",  [region: "..."], fall_aspect: 1.2, fall_seconds: 2}
//          跌倒：框宽高比（躺倒形态）持续超阈值即告警，region 可选（缺省全画面）；
//          建议 classes: "0"（只对人形生效，避免车辆等天然宽扁目标误报）。
//        - {id: "spd",   type: "speed", [region: "..."], speed_limit: 300,
//           px_per_meter: 0, speed_seconds: 1}
//          测速：跟踪框速度（EMA 平滑，px/s）持续超阈值告警；px_per_meter>0 时 detail
//          换算 km/h（px_per_meter 为像素→米标定系数）。
//      direction 过滤即逆行/方向合规：只上报指定方向的穿越（计数仍双向累计）。
//      ref_point（line_cross 可选）："center"（默认，框中心）| "bottom"（脚底中心，
//      行人跨线检测的行业惯例）。
//      命名规则的事件经 RuleEvent.detail 前缀 "[规则id] " 携带规则身份（RuleEvent 为
//      布局耦合类型，字段不可扩展）；同一 YAML 内 llm_hint 字段经 ruleHint() 透出。
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
    std::string type;    // "line_cross"|"intrusion"|"dwell"|"absence"|"crowd"|"abandoned"|"fall"|"speed"
    int track_id = 0;    // absence/crowd（区域级事件）为 0
    int cls_id = -1;
    std::string detail;  // "A->B"/"B->A"、滞留/离岗秒数、聚集人数等；命名规则带 "[id] " 前缀
};

// 规则计数快照（stats()/takeStats() 透出，供状态接口/周期统计用）；id 为空时用 type 标识
struct RuleStat {
    std::string id;
    std::string type;
    long cross_ab = 0;
    long cross_ba = 0;
    int crowd_now = 0;      // 当前区域内人数（瞬时值，不清零）
    int abandoned_now = 0;  // 当前静止遗留目标数（瞬时值，不清零）
};

class EventEngine {
public:
    explicit EventEngine(const class AppConfig& options);
    ~EventEngine();

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

    // 绊线/区域可视化（黄色绊线 + 半透明青色区域 + 双向计数；命名规则逐条绘制并带 id）
    void drawOverlay(cv::Mat& frame) const;

    // 规则计数快照（线程安全：读最近一次 update 后的缓存，可跨线程调用）
    std::vector<RuleStat> stats() const;
    // 取快照并清零累计穿越计数（周期统计语义；crowd/abandoned 为瞬时值不清零）
    std::vector<RuleStat> takeStats();
    // 规则的 LLM 复检提示（命名规则 YAML 的 llm_hint 字段；仅在输出线程调用）
    std::string ruleHint(const std::string& rule_id) const;

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
