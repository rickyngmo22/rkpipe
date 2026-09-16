#include "core/event_engine.h"
#include "daemon/mini_json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "config/app_config.h"

namespace {

// 解析 "x1,y1,x2,y2,..." 坐标序列为点列表（数量在 [min_pts, max_pts] 内才有效）
std::vector<cv::Point> parsePoints(const std::string& spec, std::size_t min_pts, std::size_t max_pts) {
    std::vector<cv::Point> pts;
    std::string token;
    std::vector<int> values;
    auto flush = [&]() {
        if (!token.empty()) {
            values.push_back(std::atoi(token.c_str()));
            token.clear();
        }
    };
    for (const char ch : spec) {
        if (ch == ',' || ch == ';') {
            flush();
        } else if (isdigit(static_cast<unsigned char>(ch)) || ch == '-') {
            token.push_back(ch);
        }
    }
    flush();
    if (values.size() % 2 != 0) {
        values.pop_back();
    }
    if (values.size() / 2 < min_pts || values.size() / 2 > max_pts) {
        return {};
    }
    for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
        pts.emplace_back(values[i], values[i + 1]);
    }
    return pts;
}

std::unordered_set<int> parseClassList(const std::string& spec) {
    std::unordered_set<int> out;
    std::string token;
    auto flush = [&]() {
        if (!token.empty()) {
            out.insert(std::atoi(token.c_str()));
            token.clear();
        }
    };
    for (const char ch : spec) {
        if (ch == ',') {
            flush();
        } else if (isdigit(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
        }
    }
    flush();
    return out;
}

// 点 p 相对有向直线 a->b 的叉积符号（>0 左侧，<0 右侧，0 共线）
double sideOfLine(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) {
    return static_cast<double>(b.x - a.x) * (p.y - a.y) - static_cast<double>(b.y - a.y) * (p.x - a.x);
}

double orientation(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
    return static_cast<double>(b.x - a.x) * (c.y - a.y) - static_cast<double>(b.y - a.y) * (c.x - a.x);
}

// 两线段是否严格相交（共线重叠不算，避免沿绊线滑行的轨迹重复计数）
bool segmentsCross(const cv::Point2f& p0, const cv::Point2f& p1,
                   const cv::Point2f& q0, const cv::Point2f& q1) {
    const double d1 = orientation(q0, q1, p0);
    const double d2 = orientation(q0, q1, p1);
    const double d3 = orientation(p0, p1, q0);
    const double d4 = orientation(p0, p1, q1);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

std::string formatSeconds(double seconds) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.0fs", seconds);
    return buf;
}

void warnRule(const std::string& message) {
    std::fprintf(stderr, "[rk_pipe][event] %s\n", message.c_str());
}

// 规则类型名（统计快照/标识用；与 RuleEvent.type 字符串一致）
const char* ruleTypeName(int type) {
    static const char* kNames[] = {"line_cross", "intrusion", "dwell",
                                   "absence",    "crowd",     "abandoned",
                                   "fall",       "speed"};
    if (type < 0 || type >= static_cast<int>(sizeof(kNames) / sizeof(kNames[0]))) {
        return "unknown";  // 新增枚举忘同步本表时防越界（段错误→可定位的占位名）
    }
    return kNames[type];
}

// 区域几何签名（去重绘制 / 同几何规则识别用）
std::string regionSignature(const std::vector<cv::Point>& region) {
    std::string sig;
    for (const auto& p : region) {
        sig += std::to_string(p.x) + "," + std::to_string(p.y) + ";";
    }
    return sig;
}

}  // namespace

EventEngine::EventEngine(const AppConfig& options) {
    if (!options.event_rules.empty()) {
        // 命名多规则（非空时优先，扁平键不再合成）
        int auto_idx = 1;
        for (const auto& rc : options.event_rules) {
            Rule rule;
            rule.id = rc.id.empty() ? ("rule" + std::to_string(auto_idx)) : rc.id;
            ++auto_idx;
            const std::string where = "event_rules[" + rc.type + "/" + rule.id + "]";
            if (rc.type == "line_cross") {
                rule.type = RuleType::LineCross;
                rule.line = parsePoints(rc.line, 2, 2);
                if (rule.line.size() != 2) {
                    warnRule(where + " 绊线格式非法（需 x1,y1,x2,y2），规则已跳过");
                    continue;
                }
                rule.direction = rc.direction.empty() ? "both" : rc.direction;
                if (rule.direction != "both" && rule.direction != "A2B" && rule.direction != "B2A") {
                    warnRule(where + " direction 非法（both/A2B/B2A），回退 both");
                    rule.direction = "both";
                }
                rule.ref_point = rc.ref_point == "bottom" ? RefPoint::Bottom
                                                          : RefPoint::Center;
            } else if (rc.type == "intrusion" || rc.type == "dwell" || rc.type == "absence" ||
                       rc.type == "crowd" || rc.type == "abandoned") {
                rule.region = parsePoints(rc.region, 3, 1024);
                if (rule.region.size() < 3) {
                    warnRule(where + " 区域格式非法（至少 3 个点），规则已跳过");
                    continue;
                }
                if (rc.type == "intrusion") {
                    rule.type = RuleType::Intrusion;
                } else if (rc.type == "dwell") {
                    rule.type = RuleType::Dwell;
                } else if (rc.type == "absence") {
                    rule.type = RuleType::Absence;
                } else if (rc.type == "crowd") {
                    rule.type = RuleType::Crowd;
                } else {
                    rule.type = RuleType::Abandoned;
                }
            } else if (rc.type == "fall") {
                rule.type = RuleType::Fall;
                rule.fall_seconds = rc.fall_seconds > 0.0 ? rc.fall_seconds : 2.0;
                rule.fall_aspect = rc.fall_aspect > 0.0 ? rc.fall_aspect : 1.2;
                // region 可选：配置了则只在区域内检测跌倒，缺省全画面
                if (!rc.region.empty()) {
                    rule.region = parsePoints(rc.region, 3, 1024);
                    if (rule.region.size() < 3) {
                        warnRule(where + " 区域格式非法（至少 3 个点），规则已跳过");
                        continue;
                    }
                }
            } else if (rc.type == "speed") {
                // 测速（G3.1）：跟踪框 EMA 速度持续超阈值告警（region 可选，缺省全画面）
                if (rc.speed_limit <= 0.0) {
                    warnRule(where + " speed_limit 必须 > 0（px/s），规则已跳过");
                    continue;
                }
                rule.type = RuleType::Speed;
                rule.speed_limit = rc.speed_limit;
                rule.px_per_meter = rc.px_per_meter > 0.0 ? rc.px_per_meter : 0.0;
                rule.speed_seconds = rc.speed_seconds > 0.0 ? rc.speed_seconds : 1.0;
                if (!rc.region.empty()) {
                    rule.region = parsePoints(rc.region, 3, 1024);
                    if (rule.region.size() < 3) {
                        warnRule(where + " 区域格式非法（至少 3 个点），规则已跳过");
                        continue;
                    }
                }
            } else {
                warnRule(where + " 类型未知，规则已跳过");
                continue;
            }
            rule.classes = parseClassList(rc.classes);
            rule.llm_hint = rc.llm_hint;
            rule.dwell_seconds = rc.dwell_seconds > 0.0 ? rc.dwell_seconds : 10.0;
            rule.absence_seconds = rc.absence_seconds > 0.0 ? rc.absence_seconds : 30.0;
            rule.seconds = rc.seconds > 0.0 ? rc.seconds : 60.0;
            rule.min_count = rc.min_count > 0 ? rc.min_count : 3;
            rules_.push_back(std::move(rule));
        }
    } else {
        // 扁平键合成匿名规则（向后兼容，语义与旧版逐帧一致）
        const auto classes = parseClassList(options.event_classes);
        if (options.event_line_cross) {
            Rule rule;
            rule.type = RuleType::LineCross;
            rule.line = parsePoints(options.event_line, 2, 2);
            rule.classes = classes;
            if (rule.line.size() != 2) {
                warnRule("event_line_cross=1 但 event_line 未配置或格式非法（需 x1,y1,x2,y2），规则已禁用");
            } else {
                rules_.push_back(std::move(rule));
            }
        }
        auto region = parsePoints(options.event_region, 3, 1024);
        if (region.size() >= 3) {
            if (options.event_intrusion) {
                Rule rule;
                rule.type = RuleType::Intrusion;
                rule.region = region;
                rule.classes = classes;
                rules_.push_back(std::move(rule));
            }
            if (options.event_dwell) {
                Rule rule;
                rule.type = RuleType::Dwell;
                rule.region = region;
                rule.classes = classes;
                rule.dwell_seconds =
                    options.event_dwell_seconds > 0.0 ? options.event_dwell_seconds : 10.0;
                rules_.push_back(std::move(rule));
            }
            if (options.event_absence) {
                Rule rule;
                rule.type = RuleType::Absence;
                rule.region = region;
                rule.classes = classes;
                rule.absence_seconds =
                    options.event_absence_seconds > 0.0 ? options.event_absence_seconds : 30.0;
                rules_.push_back(std::move(rule));
            }
        } else if (options.event_intrusion || options.event_dwell || options.event_absence) {
            warnRule("入侵/滞留/离岗规则已配置但 event_region 未配置或格式非法（需至少 3 个点），相关规则已禁用");
        }
    }
    active_ = !rules_.empty();
    refreshStatsCacheLocked();

    // 计数持久化（可靠性）：恢复跨重启累计的穿越计数
    stats_path_ = options.event_stats_path;
    if (!stats_path_.empty()) {
        loadPersistedStats();
    }
}

bool EventEngine::allowedClass(const Rule& rule, int cls_id) const {
    return rule.classes.empty() || rule.classes.count(cls_id) > 0;
}

bool EventEngine::pointInRegion(const std::vector<cv::Point>& region, const cv::Point2f& p) {
    return cv::pointPolygonTest(region, p, false) >= 0;
}

bool EventEngine::lineCrossEnabled() const {
    return std::any_of(rules_.begin(), rules_.end(),
                       [](const Rule& r) { return r.type == RuleType::LineCross; });
}

bool EventEngine::intrusionEnabled() const {
    return std::any_of(rules_.begin(), rules_.end(),
                       [](const Rule& r) { return r.type == RuleType::Intrusion; });
}

bool EventEngine::dwellEnabled() const {
    return std::any_of(rules_.begin(), rules_.end(),
                       [](const Rule& r) { return r.type == RuleType::Dwell; });
}

bool EventEngine::absenceEnabled() const {
    return std::any_of(rules_.begin(), rules_.end(),
                       [](const Rule& r) { return r.type == RuleType::Absence; });
}

const std::vector<cv::Point>& EventEngine::line() const {
    static const std::vector<cv::Point> kEmpty;
    for (const auto& r : rules_) {
        if (r.type == RuleType::LineCross) {
            return r.line;
        }
    }
    return kEmpty;
}

long EventEngine::lineCrossAB() const {
    for (const auto& r : rules_) {
        if (r.type == RuleType::LineCross) {
            return r.cross_ab;
        }
    }
    return 0;
}

long EventEngine::lineCrossBA() const {
    for (const auto& r : rules_) {
        if (r.type == RuleType::LineCross) {
            return r.cross_ba;
        }
    }
    return 0;
}

void EventEngine::updateRule(Rule& rule, const std::vector<TrackedDetection>& tracked,
                             std::chrono::steady_clock::time_point now, std::vector<RuleEvent>* events) {
    const bool is_line = rule.type == RuleType::LineCross;
    const bool is_region = !is_line && rule.type != RuleType::Fall && rule.type != RuleType::Speed;
    if (is_region) {
        rule.crowd_count = 0;
    }
    // 本帧区域占用（Absence 用）：以"本帧检测列表"为准，不能读轨迹残留状态
    // ——目标从列表消失后其 st.in_region 要到老化才清除，用它会把离岗计时永远压住
    bool region_occupied_this_frame = false;

    for (const auto& td : tracked) {
        if (td.track_id <= 0) {
            continue;
        }
        const cv::Point2f center((td.det.box.left + td.det.box.right) * 0.5f,
                                 (td.det.box.top + td.det.box.bottom) * 0.5f);
        // G3.2：绊线判定参考点（center=框中心默认；bottom=脚底中心，行人跨线惯例）
        const cv::Point2f ref_pt =
            rule.ref_point == RefPoint::Bottom
                ? cv::Point2f((td.det.box.left + td.det.box.right) * 0.5f,
                              static_cast<float>(td.det.box.bottom))
                : center;
        const bool class_ok = allowedClass(rule, td.det.cls_id);
        Rule::TrackState& st = rule.tracks[td.track_id];

        // 绊线穿越：上一帧参考点与本帧参考点分居直线两侧且轨迹线段与绊线相交。
        // 计数始终双向累计；direction 非法值已在构造时归一为 both
        if (is_line && class_ok && td.is_confirmed && st.has_prev) {
            const double s0 = sideOfLine(rule.line[0], rule.line[1], st.prev_ref);
            const double s1 = sideOfLine(rule.line[0], rule.line[1], ref_pt);
            if (s0 != 0.0 && s1 != 0.0 && ((s0 > 0) != (s1 > 0)) &&
                segmentsCross(st.prev_ref, ref_pt, rule.line[0], rule.line[1])) {
                const bool a2b = s1 > 0;
                if (a2b) {
                    ++rule.cross_ab;
                } else {
                    ++rule.cross_ba;
                }
                // 方向合规过滤：direction=A2B/B2A 时只上报指定方向（逆行检测）
                if (rule.direction == "both" || (rule.direction == "A2B") == a2b) {
                    RuleEvent ev;
                    ev.type = "line_cross";
                    ev.rule_id = rule.id;
                    ev.track_id = td.track_id;
                    ev.cls_id = td.det.cls_id;
                    ev.detail = a2b ? "A->B" : "B->A";
                    events->push_back(std::move(ev));
                }
            }
        }

        // 跌倒（M1）：框宽高比（躺倒形态）持续超过阈值即告警，region 可选限定区域。
        // 宽高比回落（起身）即复位，可重复触发；行人场景建议 classes 只配 person，
        // 避免车辆等天然宽扁目标误报。
        if (rule.type == RuleType::Fall && class_ok && td.is_confirmed) {
            const bool in_zone = rule.region.empty() || pointInRegion(rule.region, center);
            const float bw = td.det.box.right - td.det.box.left;
            const float bh = td.det.box.bottom - td.det.box.top;
            const bool lying = bh > 1.0f && (bw / bh) >= rule.fall_aspect && in_zone;
            if (!lying) {
                st.lying = false;
                st.fall_fired = false;
            } else if (!st.lying) {
                st.lying = true;
                st.lying_since = now;
            } else if (!st.fall_fired) {
                const double lying_s =
                    std::chrono::duration<double>(now - st.lying_since).count();
                if (lying_s >= rule.fall_seconds) {
                    st.fall_fired = true;  // 起身前不重复触发
                    RuleEvent ev;
                    ev.type = "fall";
                    ev.rule_id = rule.id;
                    ev.track_id = td.track_id;
                    ev.cls_id = td.det.cls_id;
                    ev.detail = formatSeconds(lying_s);
                    events->push_back(std::move(ev));
                }
            }
        }

        // 测速（G3.1）：跟踪框速度（EMA 平滑，px/s）持续超阈值告警，region 可选限定区域。
        // 速度回落即复位，可重复触发；px_per_meter>0 时 detail 换算 km/h。
        if (rule.type == RuleType::Speed && class_ok && td.is_confirmed && st.has_prev) {
            const bool in_zone = rule.region.empty() || pointInRegion(rule.region, center);
            const double dt = std::chrono::duration<double>(now - st.last_seen).count();
            if (dt > 1e-3) {
                const float dx = center.x - st.prev_center.x;
                const float dy = center.y - st.prev_center.y;
                const float inst = std::sqrt(dx * dx + dy * dy) / static_cast<float>(dt);
                st.speed_ema =
                    st.speed_ema < 0.0f ? inst : 0.4f * inst + 0.6f * st.speed_ema;  // EMA 去抖
                const bool over = in_zone && st.speed_ema >= rule.speed_limit;
                if (!over) {
                    st.speed_exceeding = false;
                    st.speed_fired = false;
                } else if (!st.speed_exceeding) {
                    st.speed_exceeding = true;
                    st.speed_exceeding_since = now;
                } else if (!st.speed_fired) {
                    const double over_s =
                        std::chrono::duration<double>(now - st.speed_exceeding_since).count();
                    if (over_s >= rule.speed_seconds) {
                        st.speed_fired = true;  // 回落前不重复触发
                        RuleEvent ev;
                        ev.type = "speed";
                        ev.rule_id = rule.id;
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        char buf[32];
                        if (rule.px_per_meter > 0.0) {
                            std::snprintf(buf, sizeof(buf), "%.1fkm/h",
                                          st.speed_ema / rule.px_per_meter * 3.6);
                        } else {
                            std::snprintf(buf, sizeof(buf), "%.0fpx/s", st.speed_ema);
                        }
                        ev.detail = buf;
                        events->push_back(std::move(ev));
                    }
                }
            }
        }

        if (is_region && class_ok && !rule.region.empty()) {
            const bool in_reg = pointInRegion(rule.region, center);

            // 入侵：区域外 → 区域内（已确认轨迹）
            if (rule.type == RuleType::Intrusion && td.is_confirmed && in_reg && !st.in_region) {
                RuleEvent ev;
                ev.type = "intrusion";
                ev.rule_id = rule.id;
                ev.track_id = td.track_id;
                ev.cls_id = td.det.cls_id;
                events->push_back(std::move(ev));
            }
            // 滞留：进入后停留超过阈值，每次进入只触发一次
            if (rule.type == RuleType::Dwell && in_reg) {
                if (!st.in_region) {
                    st.enter_time = now;
                    st.dwell_fired = false;
                } else if (!st.dwell_fired) {
                    const double stayed = std::chrono::duration<double>(now - st.enter_time).count();
                    if (stayed >= rule.dwell_seconds) {
                        st.dwell_fired = true;
                        RuleEvent ev;
                        ev.type = "dwell";
                        ev.rule_id = rule.id;
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        ev.detail = formatSeconds(stayed);
                        events->push_back(std::move(ev));
                    }
                }
            }
            // 聚集：区域内（已确认、生效类别）人数计数，post-loop 判定阈值
            if (rule.type == RuleType::Crowd && td.is_confirmed && in_reg) {
                ++rule.crowd_count;
            }
            // 遗留物：区域内目标近乎静止超过阈值，每次静止期只触发一次
            if (rule.type == RuleType::Abandoned && td.is_confirmed && in_reg) {
                const float dx = center.x - st.anchor.x;
                const float dy = center.y - st.anchor.y;
                const float diag = std::sqrt(static_cast<float>(
                    (td.det.box.right - td.det.box.left) * (td.det.box.right - td.det.box.left) +
                    (td.det.box.bottom - td.det.box.top) * (td.det.box.bottom - td.det.box.top)));
                const float move_thresh = std::max(8.0f, diag * 0.1f);
                const float moved = std::sqrt(dx * dx + dy * dy);
                if (!st.has_anchor || moved > move_thresh) {
                    // 新出现或发生位移：重置静止计时
                    st.anchor = center;
                    st.stationary_since = now;
                    st.has_anchor = true;
                    st.abandoned_fired = false;
                } else if (!st.abandoned_fired) {
                    const double still = std::chrono::duration<double>(now - st.stationary_since).count();
                    if (still >= rule.seconds) {
                        st.abandoned_fired = true;  // 移动/离开/老化前不再重复
                        RuleEvent ev;
                        ev.type = "abandoned";
                        ev.rule_id = rule.id;
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        ev.detail = formatSeconds(still);
                        events->push_back(std::move(ev));
                    }
                }
            }
            if (rule.type == RuleType::Absence && in_reg) {
                rule.region_ever_occupied = true;
                region_occupied_this_frame = true;
            }
            st.in_region = in_reg;
        }

        st.prev_center = center;
        st.prev_ref = ref_pt;
        st.has_prev = true;
        st.has_prev_ref = true;
        st.last_seen = now;
        st.cls_id = td.det.cls_id;
    }

    // 离岗：区域出现过人后持续无人超过阈值，重新有人前只触发一次
    if (rule.type == RuleType::Absence) {
        if (region_occupied_this_frame) {
            rule.absence_fired = false;
            rule.region_empty_timing = false;
        } else if (rule.region_ever_occupied) {
            if (!rule.region_empty_timing) {
                rule.region_empty_timing = true;
                rule.region_empty_since = now;
            } else if (!rule.absence_fired) {
                const double empty_s = std::chrono::duration<double>(now - rule.region_empty_since).count();
                if (empty_s >= rule.absence_seconds) {
                    rule.absence_fired = true;
                    // 复位区域内目标状态：之后任何目标（含同 ID）再进入都视为再次入侵。
                    // legacy 单区域共享一份轨迹状态；多规则下同几何的兄弟规则一并复位，
                    // 否则入侵规则的残留 in_region 会吞掉重进入事件
                    for (auto& other : rules_) {
                        if (&other == &rule || other.type == RuleType::LineCross ||
                            other.region.empty()) {
                            continue;
                        }
                        if (regionSignature(other.region) != regionSignature(rule.region)) {
                            continue;
                        }
                        for (auto& kv : other.tracks) {
                            kv.second.in_region = false;
                        }
                    }
                    for (auto& kv : rule.tracks) {
                        kv.second.in_region = false;
                    }
                    RuleEvent ev;
                    ev.type = "absence";
                    ev.rule_id = rule.id;
                    ev.track_id = 0;
                    ev.cls_id = -1;
                    ev.detail = formatSeconds(empty_s);
                    events->push_back(std::move(ev));
                }
            }
        }
    }
    // 聚集：人数越过阈值触发一次，回落后重新武装（迟滞防抖）
    if (rule.type == RuleType::Crowd) {
        if (rule.crowd_count >= rule.min_count) {
            if (!rule.crowd_fired) {
                rule.crowd_fired = true;
                RuleEvent ev;
                ev.type = "crowd";
                ev.rule_id = rule.id;
                ev.track_id = 0;
                ev.cls_id = -1;
                ev.detail = std::to_string(rule.crowd_count) + "人";
                events->push_back(std::move(ev));
            }
        } else {
            rule.crowd_fired = false;
        }
    }
    // 遗留物观测计数：当前处于静止锚定状态的目标数
    if (rule.type == RuleType::Abandoned) {
        rule.abandoned_count = static_cast<int>(std::count_if(
            rule.tracks.begin(), rule.tracks.end(),
            [](const auto& kv) { return kv.second.has_anchor && kv.second.in_region; }));
    }
}

void EventEngine::evictStaleTracks(Rule& rule, std::chrono::steady_clock::time_point now) {
    // 轨迹老化：长时间未见的目标清除状态，防止 map 无界增长
    for (auto it = rule.tracks.begin(); it != rule.tracks.end();) {
        const double unseen = std::chrono::duration<double>(now - it->second.last_seen).count();
        if (unseen > stale_seconds_) {
            it = rule.tracks.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<RuleEvent> EventEngine::update(const std::vector<TrackedDetection>& tracked,
                                           std::chrono::steady_clock::time_point now) {
    std::vector<RuleEvent> events;
    if (!active_) {
        return events;
    }
    for (auto& rule : rules_) {
        updateRule(rule, tracked, now, &events);
        evictStaleTracks(rule, now);
    }
    refreshStatsCacheLocked();

    // 计数周期落盘（30s 一次；临时文件+rename 原子替换）
    if (!stats_path_.empty()) {
        if (std::chrono::duration<double>(now - last_persist_).count() >= 30.0) {
            last_persist_ = now;
            std::lock_guard<std::mutex> lock(stats_mutex_);
            persistStatsLocked();
        }
    }
    return events;
}

void EventEngine::refreshStatsCacheLocked() {
    std::vector<RuleStat> cache;
    cache.reserve(rules_.size());
    for (const auto& rule : rules_) {
        RuleStat st;
        st.id = rule.id.empty() ? ruleTypeName(static_cast<int>(rule.type)) : rule.id;
        st.type = ruleTypeName(static_cast<int>(rule.type));
        st.cross_ab = rule.cross_ab;
        st.cross_ba = rule.cross_ba;
        st.crowd_now = rule.crowd_count;
        st.abandoned_now = rule.abandoned_count;
        cache.push_back(std::move(st));
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_cache_ = std::move(cache);
}

std::vector<RuleStat> EventEngine::stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_cache_;
}

std::vector<RuleStat> EventEngine::takeStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::vector<RuleStat> out = stats_cache_;
    // 清零累计穿越计数（周期报表语义：报表间隔内的增量）；瞬时值（crowd/abandoned）保留
    for (auto& rule : rules_) {
        rule.cross_ab = 0;
        rule.cross_ba = 0;
    }
    // 瞬时值取当前规则态（缓存里的瞬时值随每帧刷新，直接读规则即可——
    // takeStats 只在输出线程调用，与 update 串行，安全）
    for (size_t i = 0; i < rules_.size() && i < out.size(); ++i) {
        out[i].crowd_now = rules_[i].crowd_count;
        out[i].abandoned_now = rules_[i].abandoned_count;
    }
    return out;
}

std::string EventEngine::ruleHint(const std::string& rule_id) const {
    for (const auto& rule : rules_) {
        const std::string key = rule.id.empty() ? ruleTypeName(static_cast<int>(rule.type)) : rule.id;
        if (key == rule_id) {
            return rule.llm_hint;
        }
    }
    return std::string();
}

void EventEngine::drawOverlay(cv::Mat& frame) const {
    if (frame.empty()) {
        return;
    }
    std::vector<std::string> drawn_regions;  // 几何相同的区域只画一次（legacy 多规则共用区域）
    for (const auto& rule : rules_) {
        // 区域：半透明青色填充 + 描边 + 规则名
        if (!rule.region.empty()) {
            const std::string sig = regionSignature(rule.region);
            if (std::find(drawn_regions.begin(), drawn_regions.end(), sig) != drawn_regions.end()) {
                continue;
            }
            drawn_regions.push_back(sig);
            cv::Rect bb = cv::boundingRect(rule.region) & cv::Rect(0, 0, frame.cols, frame.rows);
            if (bb.width > 0 && bb.height > 0) {
                std::vector<cv::Point> shifted;
                shifted.reserve(rule.region.size());
                for (const auto& p : rule.region) {
                    shifted.push_back(p - bb.tl());
                }
                cv::Mat mask(bb.size(), CV_8UC1, cv::Scalar(0));
                cv::fillPoly(mask, shifted, cv::Scalar(255));
                cv::Mat tint(bb.size(), frame.type(), cv::Scalar(200, 200, 0));  // BGR 青色
                cv::Mat blended;
                cv::addWeighted(frame(bb), 0.82, tint, 0.18, 0, blended);
                blended.copyTo(frame(bb), mask);
                // 描边用原始坐标（shifted 只用于 bb 子图内的填充；画全帧会跑到左上角）
                std::vector<std::vector<cv::Point>> outline{rule.region};
                cv::polylines(frame, outline, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                const std::string label = rule.id.empty() ? "zone" : rule.id;
                cv::putText(frame, label, cv::Point(bb.x + 6, bb.y + 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
        }
        // 绊线：黄线 + 双向计数（带规则名）
        if (rule.type == RuleType::LineCross && rule.line.size() == 2) {
            cv::line(frame, rule.line[0], rule.line[1], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            for (const auto& p : rule.line) {
                cv::circle(frame, p, 4, cv::Scalar(0, 255, 255), cv::FILLED);
            }
            char label[96];
            if (rule.id.empty()) {
                std::snprintf(label, sizeof(label), "A->B:%ld B->A:%ld", rule.cross_ab, rule.cross_ba);
            } else {
                std::snprintf(label, sizeof(label), "%s A->B:%ld B->A:%ld", rule.id.c_str(),
                              rule.cross_ab, rule.cross_ba);
            }
            const int lx = std::max(4, std::min(rule.line[0].x, frame.cols - 260));
            const int ly = std::max(22, rule.line[0].y - 8);
            cv::putText(frame, label, cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }
}

// ---- 计数持久化（可靠性：客流统计跨重启累计）----

void EventEngine::loadPersistedStats() {
    std::ifstream in(stats_path_);
    if (!in.is_open()) {
        return;  // 首次运行无文件，正常
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool ok = false;
    mini_json::JsonValue v = mini_json::JsonParser(content).parse(ok);
    if (!ok || v.type != mini_json::JsonValue::Type::Object) {
        std::fprintf(stderr, "[rk_pipe][event] 计数文件解析失败，忽略: %s\n", stats_path_.c_str());
        return;
    }
    const mini_json::JsonValue* rules = v.get("rules");
    if (!rules || rules->type != mini_json::JsonValue::Type::Array) {
        return;
    }
    // 恢复匹配：优先按规则 id，扁平键规则（id 为空）回退按 type——
    // 规则改线/删除后旧计数自然失效
    int restored = 0;
    for (auto& rule : rules_) {
        const std::string type_name = ruleTypeName(static_cast<int>(rule.type));
        for (const auto& item : rules->arr) {
            if (item.type != mini_json::JsonValue::Type::Object) {
                continue;
            }
            const bool id_match = !rule.id.empty() && item.getString("id") == rule.id;
            const bool type_match = item.getString("type") == type_name;
            if (!id_match && !(rule.id.empty() && type_match)) {
                continue;
            }
            const long ab = static_cast<long>(item.getNumber("cross_ab", 0));
            const long ba = static_cast<long>(item.getNumber("cross_ba", 0));
            if (ab > 0 || ba > 0) {
                rule.cross_ab = ab;
                rule.cross_ba = ba;
                ++restored;
            }
            break;
        }
    }
    if (restored > 0) {
        refreshStatsCacheLocked();  // 恢复的计数立即可见于 stats()
        std::fprintf(stderr, "[rk_pipe][event] 恢复 %d 条规则的穿越计数: %s\n", restored,
                     stats_path_.c_str());
    }
}

void EventEngine::persistStatsLocked() const {
    // 临时文件 + rename 原子替换（写一半崩溃不留坏文件）
    const std::string tmp = stats_path_ + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) {
        return;
    }
    out << "{\"saved_ts\":" << std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count()
        << ",\"rules\":[";
    bool first = true;
    for (const auto& rule : rules_) {
        if (rule.cross_ab <= 0 && rule.cross_ba <= 0) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << "{\"id\":\"";
        for (const char ch : rule.id) {
            if (ch == '"' || ch == '\\') {
                out << '\\';
            }
            out << ch;
        }
        out << "\",\"type\":\"" << ruleTypeName(static_cast<int>(rule.type)) << "\",\"cross_ab\":" << rule.cross_ab
            << ",\"cross_ba\":" << rule.cross_ba << "}";
    }
    out << "]}\n";
    out.close();
    std::rename(tmp.c_str(), stats_path_.c_str());
}
