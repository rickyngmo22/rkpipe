#pragma once

// 事件规则引擎（D1 绊线 / D2 区域规则的公共底座，支持命名多规则）。
// detect + tracking 输出线程侧逐帧评估，产出结构化规则事件，由 AlertRuntime 上报。
//
// 两种配置形态（app_config）：
//   1) 扁平键（向后兼容）：event_line/event_line_cross/event_region/event_intrusion/
//      event_dwell/event_absence/event_classes —— 合成为一条匿名规则；
//   2) 命名多规则：event_rules YAML 列表（daemon 侧对应 streams.json 每路 rules 数组）：
//        - {id: "gate",  type: "line_cross", line: "x1,y1,x2,y2", direction: "both|A2B|B2A"}
//        - {id: "yard",  type: "intrusion",  region: "x1,y1,...", classes: "0"}
//        - {id: "loiter",type: "dwell",      region: "...", dwell_seconds: 10}
//        - {id: "post",  type: "absence",    region: "...", absence_seconds: 30}
//        - {id: "crowd", type: "crowd",      region: "...", min_count: 3}   区域内人数超限
//        - {id: "left",  type: "abandoned",  region: "...", seconds: 60}    区域内静止遗留物
//        - {id: "fall",  type: "fall",  [region: "..."], fall_aspect: 1.2, fall_seconds: 2}
//          跌倒：框宽高比（躺倒形态）持续超阈值即告警，region 可选（缺省全画面）；
//          行人场景建议 classes: "0"（只对人形生效，避免车辆等天然宽扁目标误报）。
//        - {id: "spd",   type: "speed", [region: "..."], speed_limit: 300, px_per_meter: 0}
//          测速（G3.1）：跟踪框速度（EMA 平滑）持续超 speed_limit（px/s）告警；
//          px_per_meter>0 时 detail 换算 km/h（像素标定，tools/calibrate_camera.py 可得）；
//          speed_seconds 持续去抖；region 可选（缺省全画面）。
//      direction 过滤即逆行/方向合规：只上报指定方向的穿越（计数仍双向累计）。
//      ref_point（G3.2，line_cross 可选）："center"（默认，框中心）| "bottom"（脚底中心，
//      行人跨线检测的行业惯例——高框+矮绊线时中心判定易漏检/早触发）。
//
// 纯逻辑 + OpenCV core，无硬件依赖，CI 单测可覆盖（几何/状态机全部可测）。

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "detection/simple_object_tracker.h"

struct RuleEvent {
    std::string type;     // "line_cross" | "intrusion" | "dwell" | "absence" | "crowd" | "abandoned" | "fall" | "speed"
    std::string rule_id;  // 命名规则的 id；扁平配置合成的匿名规则为空
    int track_id = 0;     // absence/crowd（区域级事件）为 0
    int cls_id = -1;
    std::string detail;   // "A->B"/"B->A"、滞留/离岗秒数、聚集人数等
};

// 规则计数快照（/status.json 透出 + 周期统计报表 rule_stats 用）；id 为空时用 type 标识
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

    bool active() const { return active_; }
    // 便捷查询：是否存在该类型规则（多规则下按类型聚合判断）
    bool lineCrossEnabled() const;
    bool intrusionEnabled() const;
    bool dwellEnabled() const;
    bool absenceEnabled() const;

    // 旧单规则接口（委托给首条对应规则；无该类规则时返回空/0）
    const std::vector<cv::Point>& line() const;
    long lineCrossAB() const;
    long lineCrossBA() const;

    // 每输出帧调用一次（须在 tracker.update 之后传入本帧 tracked；即使空列表也要喂，
    // absence/crowd/abandoned 计时与轨迹老化依赖连续帧）。
    std::vector<RuleEvent> update(const std::vector<TrackedDetection>& tracked,
                                  std::chrono::steady_clock::time_point now);

    // 绊线/区域可视化（每条规则各自绘制：黄色绊线+计数 / 半透明青色区域+规则名）
    void drawOverlay(cv::Mat& frame) const;

    // 规则计数快照（线程安全：读最近一次 update 后的缓存，可跨线程调用，/status.json 用）
    std::vector<RuleStat> stats() const;
    // 取快照并清零累计计数（周期统计报表用；crowd/abandoned 为瞬时值不清零）
    std::vector<RuleStat> takeStats();
    // 规则的 LLM 复检提示（event_rules 的 llm_hint 字段；仅在输出线程调用）
    std::string ruleHint(const std::string& rule_id) const;

private:
    enum class RuleType { LineCross, Intrusion, Dwell, Absence, Crowd, Abandoned, Fall, Speed };
    enum class RefPoint { Center, Bottom };  // LineCross 判定点（G3.2）

    struct Rule {
        std::string id;
        RuleType type;
        std::vector<cv::Point> line;       // LineCross
        std::vector<cv::Point> region;     // 区域类规则；Fall/Speed 可选（空=全画面）
        std::unordered_set<int> classes;   // 空=全类别
        std::string direction = "both";    // LineCross：both/A2B/B2A（逆行过滤）
        RefPoint ref_point = RefPoint::Center;  // LineCross：center（默认）/bottom（脚底）
        double dwell_seconds = 10.0;
        double absence_seconds = 30.0;
        double seconds = 60.0;             // Abandoned 静止阈值
        int min_count = 3;                 // Crowd 人数阈值
        double fall_seconds = 2.0;         // Fall 躺倒持续阈值（去抖）
        double fall_aspect = 1.2;          // Fall 宽高比阈值（w/h >= 该值视为躺倒）
        double speed_limit = 0.0;          // Speed 阈值（px/s）
        double px_per_meter = 0.0;         // Speed 像素标定（>0 = detail 换算 km/h）
        double speed_seconds = 1.0;        // Speed 超限持续去抖
        std::string llm_hint;              // 规则级 LLM 复检提示（随事件上报）
        long cross_ab = 0;
        long cross_ba = 0;
        // 区域级状态（Absence / Crowd）
        bool region_ever_occupied = false;
        bool absence_fired = false;
        bool region_empty_timing = false;
        std::chrono::steady_clock::time_point region_empty_since{};
        bool crowd_fired = false;
        int crowd_count = 0;  // 本帧区域内已确认目标数（Crowd 规则）
        // Abandoned 区域级：当前静止遗留计数（观测用）
        int abandoned_count = 0;

        struct TrackState {
            cv::Point2f prev_center{0.0f, 0.0f};
            bool has_prev = false;
            cv::Point2f prev_ref{0.0f, 0.0f};  // 绊线参考点上一帧位置（G3.2，可与 center 不同）
            bool has_prev_ref = false;
            bool in_region = false;
            std::chrono::steady_clock::time_point enter_time{};
            bool dwell_fired = false;
            std::chrono::steady_clock::time_point last_seen{};
            int cls_id = -1;
            // Abandoned：静止锚点
            cv::Point2f anchor{0.0f, 0.0f};
            bool has_anchor = false;
            std::chrono::steady_clock::time_point stationary_since{};
            bool abandoned_fired = false;
            // Fall：躺倒形态持续锚点（宽高比回落即复位，可重复触发）
            bool lying = false;
            std::chrono::steady_clock::time_point lying_since{};
            bool fall_fired = false;
            // Speed：EMA 平滑速度（px/s）+ 超限持续锚点（回落即复位，可重复触发）
            float speed_ema = -1.0f;  // <0 = 尚无速度样本
            bool speed_exceeding = false;
            std::chrono::steady_clock::time_point speed_exceeding_since{};
            bool speed_fired = false;
        };
        std::unordered_map<int, TrackState> tracks;
    };

    bool allowedClass(const Rule& rule, int cls_id) const;
    static bool pointInRegion(const std::vector<cv::Point>& region, const cv::Point2f& p);
    void updateRule(Rule& rule, const std::vector<TrackedDetection>& tracked,
                    std::chrono::steady_clock::time_point now, std::vector<RuleEvent>* events);
    void evictStaleTracks(Rule& rule, std::chrono::steady_clock::time_point now);
    void refreshStatsCacheLocked();

    // 计数持久化（可靠性：客流统计跨重启累计）。构造时恢复、update 周期落盘
    void loadPersistedStats();
    void persistStatsLocked() const;   // 需持 stats_mutex_
    std::string stats_path_;           // 空=不持久化
    std::chrono::steady_clock::time_point last_persist_{};
    bool active_ = false;
    double stale_seconds_ = 5.0;  // 轨迹超过该时长未见即清除状态
    std::vector<Rule> rules_;

    // 计数快照缓存：update() 每帧刷新（输出线程写），stats() 跨线程读（web 线程）
    mutable std::mutex stats_mutex_;
    std::vector<RuleStat> stats_cache_;
};
