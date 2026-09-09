#include "core/event_engine.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
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

void warnRule(const char* message) {
    std::fprintf(stderr, "[rk_pipe][event] %s\n", message);
}

// ===========================================================================
// 命名多规则（RK_PIPE_EVENT_RULES 环境变量 YAML）。
//
// 布局红线：EventEngine/RuleEvent 与闭源核心按成员布局直接耦合（核心按构建期
// sizeof 分配对象并按偏移读 RuleEvent 字段），本类不可增删数据成员、RuleEvent
// 不可扩展。因此命名规则的全部状态放在下方进程级侧表（一次进程一条流水线，
// 与 RK_PIPE_RESULT_JSONL 同语义），规则身份经 RuleEvent.detail 的 "[id] " 前缀
// 携带；event_rules YAML 配置键与 RuleEvent.rule_id 字段随核心库后续版本协同。
// ===========================================================================

enum class RuleType { LineCross, Intrusion, Dwell, Absence, Crowd, Abandoned, Fall, Speed };
enum class RefPoint { Center, Bottom };  // LineCross 判定点

const char* ruleTypeName(int type) {
    static const char* kNames[] = {"line_cross", "intrusion", "dwell",
                                   "absence",    "crowd",     "abandoned",
                                   "fall",       "speed"};
    if (type < 0 || type >= static_cast<int>(sizeof(kNames) / sizeof(kNames[0]))) {
        return "unknown";  // 新增枚举忘同步本表时防越界（段错误→可定位的占位名）
    }
    return kNames[type];
}

bool pointInRegion(const std::vector<cv::Point>& region, const cv::Point2f& p) {
    return cv::pointPolygonTest(region, p, false) >= 0;
}

std::string regionSignature(const std::vector<cv::Point>& region) {
    std::string sig;
    for (const auto& p : region) {
        sig += std::to_string(p.x) + "," + std::to_string(p.y) + ";";
    }
    return sig;
}

struct NamedRule {
    std::string id;
    RuleType type = RuleType::LineCross;
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
    std::string llm_hint;              // 规则级 LLM 复检提示（ruleHint() 透出）
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
        cv::Point2f prev_ref{0.0f, 0.0f};  // 绊线参考点上一帧位置（可与 center 不同）
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

struct EngineExt {
    std::vector<NamedRule> rules;
    mutable std::mutex stats_mutex;
    std::vector<RuleStat> stats_cache;
};

// 按实例侧表：键为引擎对象地址。构造时写入/重置，析构时清除（旧核心按旧头文件的
// 内联析构销毁对象、不会调用本析构——此时条目留存至进程退出，无正确性影响）
std::map<const EventEngine*, EngineExt>& extTable() {
    static std::map<const EventEngine*, EngineExt> table;
    return table;
}

EngineExt* extFor(const EventEngine* self) {
    auto it = extTable().find(self);
    return it == extTable().end() ? nullptr : &it->second;
}

// ---- 命名规则 YAML 解析(RK_PIPE_EVENT_RULES 指向的文件) ----------------------

std::string nodeString(const cv::FileNode& n, const char* key, const std::string& fallback = {}) {
    const cv::FileNode v = n[key];
    if (v.empty()) {
        return fallback;
    }
    std::string s;
    v >> s;
    return s;
}

double nodeDouble(const cv::FileNode& n, const char* key, double fallback) {
    const cv::FileNode v = n[key];
    return v.empty() ? fallback : static_cast<double>(v);
}

int nodeInt(const cv::FileNode& n, const char* key, int fallback) {
    const cv::FileNode v = n[key];
    return v.empty() ? fallback : static_cast<int>(v);
}

// 解析失败/全部规则非法返回 false（引擎回退扁平键语义）
bool parseNamedRulesFile(const std::string& path, EngineExt& ext) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        warnRule("RK_PIPE_EVENT_RULES 文件打开失败，回退扁平 event_* 配置");
        return false;
    }
    const cv::FileNode list = fs["event_rules"];
    if (list.empty() || !list.isSeq()) {
        warnRule("RK_PIPE_EVENT_RULES 缺少 event_rules 列表，回退扁平 event_* 配置");
        return false;
    }
    ext.rules.clear();
    int auto_idx = 1;
    for (const auto& rc : list) {
        NamedRule rule;
        rule.id = nodeString(rc, "id");
        if (rule.id.empty()) {
            rule.id = "rule" + std::to_string(auto_idx);
        }
        ++auto_idx;
        const std::string type = nodeString(rc, "type");
        const std::string where = "event_rules[" + type + "/" + rule.id + "]";
        if (type == "line_cross") {
            rule.type = RuleType::LineCross;
            rule.line = parsePoints(nodeString(rc, "line"), 2, 2);
            if (rule.line.size() != 2) {
                warnRule((where + " 绊线格式非法（需 x1,y1,x2,y2），规则已跳过").c_str());
                continue;
            }
            rule.direction = nodeString(rc, "direction", "both");
            if (rule.direction != "both" && rule.direction != "A2B" && rule.direction != "B2A") {
                warnRule((where + " direction 非法（both/A2B/B2A），回退 both").c_str());
                rule.direction = "both";
            }
            rule.ref_point = nodeString(rc, "ref_point") == "bottom" ? RefPoint::Bottom
                                                                     : RefPoint::Center;
        } else if (type == "intrusion" || type == "dwell" || type == "absence" ||
                   type == "crowd" || type == "abandoned") {
            rule.region = parsePoints(nodeString(rc, "region"), 3, 1024);
            if (rule.region.size() < 3) {
                warnRule((where + " 区域格式非法（至少 3 个点），规则已跳过").c_str());
                continue;
            }
            if (type == "intrusion") {
                rule.type = RuleType::Intrusion;
            } else if (type == "dwell") {
                rule.type = RuleType::Dwell;
            } else if (type == "absence") {
                rule.type = RuleType::Absence;
            } else if (type == "crowd") {
                rule.type = RuleType::Crowd;
            } else {
                rule.type = RuleType::Abandoned;
            }
        } else if (type == "fall") {
            rule.type = RuleType::Fall;
            rule.fall_seconds = nodeDouble(rc, "fall_seconds", 2.0);
            rule.fall_aspect = nodeDouble(rc, "fall_aspect", 1.2);
            // region 可选：配置了则只在区域内检测跌倒，缺省全画面
            const std::string region = nodeString(rc, "region");
            if (!region.empty()) {
                rule.region = parsePoints(region, 3, 1024);
                if (rule.region.size() < 3) {
                    warnRule((where + " 区域格式非法（至少 3 个点），规则已跳过").c_str());
                    continue;
                }
            }
        } else if (type == "speed") {
            // 测速：跟踪框 EMA 速度持续超阈值告警（region 可选，缺省全画面）
            rule.speed_limit = nodeDouble(rc, "speed_limit", 0.0);
            if (rule.speed_limit <= 0.0) {
                warnRule((where + " speed_limit 必须 > 0（px/s），规则已跳过").c_str());
                continue;
            }
            rule.type = RuleType::Speed;
            rule.px_per_meter = nodeDouble(rc, "px_per_meter", 0.0);
            rule.speed_seconds = nodeDouble(rc, "speed_seconds", 1.0);
            const std::string region = nodeString(rc, "region");
            if (!region.empty()) {
                rule.region = parsePoints(region, 3, 1024);
                if (rule.region.size() < 3) {
                    warnRule((where + " 区域格式非法（至少 3 个点），规则已跳过").c_str());
                    continue;
                }
            }
        } else {
            warnRule((where + " 类型未知，规则已跳过").c_str());
            continue;
        }
        rule.classes = parseClassList(nodeString(rc, "classes"));
        rule.llm_hint = nodeString(rc, "llm_hint");
        rule.dwell_seconds = nodeDouble(rc, "dwell_seconds", 10.0);
        rule.absence_seconds = nodeDouble(rc, "absence_seconds", 30.0);
        rule.seconds = nodeDouble(rc, "seconds", 60.0);
        rule.min_count = nodeInt(rc, "min_count", 3);
        ext.rules.push_back(std::move(rule));
    }
    return !ext.rules.empty();
}

void refreshStatsCache(EngineExt& ext) {
    std::vector<RuleStat> cache;
    cache.reserve(ext.rules.size());
    for (const auto& rule : ext.rules) {
        RuleStat st;
        st.id = rule.id.empty() ? ruleTypeName(static_cast<int>(rule.type)) : rule.id;
        st.type = ruleTypeName(static_cast<int>(rule.type));
        st.cross_ab = rule.cross_ab;
        st.cross_ba = rule.cross_ba;
        st.crowd_now = rule.crowd_count;
        st.abandoned_now = rule.abandoned_count;
        cache.push_back(std::move(st));
    }
    std::lock_guard<std::mutex> lock(ext.stats_mutex);
    ext.stats_cache = std::move(cache);
}

// 规则身份进 detail（RuleEvent 布局耦合不可扩展字段，前缀即身份）
std::string tagDetail(const NamedRule& rule, const std::string& detail) {
    if (rule.id.empty()) {
        return detail;
    }
    return "[" + rule.id + "] " + detail;
}

// 单条命名规则逐帧评估（逻辑与扁平语义对齐，支持 8 类规则）
void updateNamedRule(NamedRule& rule, EngineExt& ext, double stale_seconds,
                     const std::vector<TrackedDetection>& tracked,
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
        // 绊线判定参考点（center=框中心默认；bottom=脚底中心，行人跨线惯例）
        const cv::Point2f ref_pt =
            rule.ref_point == RefPoint::Bottom
                ? cv::Point2f((td.det.box.left + td.det.box.right) * 0.5f,
                              static_cast<float>(td.det.box.bottom))
                : center;
        const bool class_ok = rule.classes.empty() || rule.classes.count(td.det.cls_id) > 0;
        NamedRule::TrackState& st = rule.tracks[td.track_id];

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
                    ev.track_id = td.track_id;
                    ev.cls_id = td.det.cls_id;
                    ev.detail = tagDetail(rule, a2b ? "A->B" : "B->A");
                    events->push_back(std::move(ev));
                }
            }
        }

        // 跌倒：框宽高比（躺倒形态）持续超过阈值即告警，region 可选限定区域。
        // 宽高比回落（起身）即复位，可重复触发
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
                    ev.track_id = td.track_id;
                    ev.cls_id = td.det.cls_id;
                    ev.detail = tagDetail(rule, formatSeconds(lying_s));
                    events->push_back(std::move(ev));
                }
            }
        }

        // 测速：跟踪框速度（EMA 平滑，px/s）持续超阈值告警，region 可选限定区域。
        // 速度回落即复位，可重复触发；px_per_meter>0 时 detail 换算 km/h
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
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        char buf[48];
                        if (rule.px_per_meter > 0.0) {
                            std::snprintf(buf, sizeof(buf), "%.1fkm/h",
                                          st.speed_ema / rule.px_per_meter * 3.6);
                        } else {
                            std::snprintf(buf, sizeof(buf), "%.0fpx/s", st.speed_ema);
                        }
                        ev.detail = tagDetail(rule, buf);
                        events->push_back(std::move(ev));
                    }
                }
            }
        }

        if (is_region && class_ok && !rule.region.empty()) {
            const bool in_reg = cv::pointPolygonTest(rule.region, center, false) >= 0;

            // 入侵：区域外 → 区域内（已确认轨迹）
            if (rule.type == RuleType::Intrusion && td.is_confirmed && in_reg && !st.in_region) {
                RuleEvent ev;
                ev.type = "intrusion";
                ev.track_id = td.track_id;
                ev.cls_id = td.det.cls_id;
                ev.detail = tagDetail(rule, "");
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
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        ev.detail = tagDetail(rule, formatSeconds(stayed));
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
                    const double still =
                        std::chrono::duration<double>(now - st.stationary_since).count();
                    if (still >= rule.seconds) {
                        st.abandoned_fired = true;  // 移动/离开/老化前不再重复
                        RuleEvent ev;
                        ev.type = "abandoned";
                        ev.track_id = td.track_id;
                        ev.cls_id = td.det.cls_id;
                        ev.detail = tagDetail(rule, formatSeconds(still));
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
                const double empty_s =
                    std::chrono::duration<double>(now - rule.region_empty_since).count();
                if (empty_s >= rule.absence_seconds) {
                    rule.absence_fired = true;
                    // 复位区域内目标状态：之后任何目标（含同 ID）再进入都视为再次入侵。
                    // 多规则下同几何的兄弟规则一并复位，否则入侵规则的残留 in_region
                    // 会吞掉重进入事件
                    const std::string sig = regionSignature(rule.region);
                    for (auto& other : ext.rules) {
                        if (&other == &rule || other.type == RuleType::LineCross ||
                            other.region.empty()) {
                            continue;
                        }
                        if (regionSignature(other.region) != sig) {
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
                    ev.track_id = 0;
                    ev.cls_id = -1;
                    ev.detail = tagDetail(rule, formatSeconds(empty_s));
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
                ev.track_id = 0;
                ev.cls_id = -1;
                ev.detail = tagDetail(rule, std::to_string(rule.crowd_count) + "人");
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

void drawNamedOverlay(EngineExt& ext, cv::Mat& frame) {
    std::vector<std::string> drawn_regions;  // 几何相同的区域只画一次（多规则共用区域）
    for (const auto& rule : ext.rules) {
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
                std::snprintf(label, sizeof(label), "A->B:%ld B->A:%ld", rule.cross_ab,
                              rule.cross_ba);
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

}  // namespace

EventEngine::EventEngine(const AppConfig& options) {
    // 命名多规则优先（RK_PIPE_EVENT_RULES 指向 YAML；规则文件进程内共享，按实例重置状态）
    if (const char* rules_yaml = std::getenv("RK_PIPE_EVENT_RULES")) {
        EngineExt& ext = extTable()[this];
        if (*rules_yaml != '\0' && parseNamedRulesFile(rules_yaml, ext)) {
            active_ = true;  // 扁平成员保持默认，update/draw 走命名路径
            refreshStatsCache(ext);
            return;
        }
        extTable().erase(this);  // 解析失败:回退扁平键,不留空条目
    }

    // 扁平键合成匿名规则（向后兼容，语义与旧版逐帧一致）
    region_ = parsePoints(options.event_region, 3, 1024);
    line_ = parsePoints(options.event_line, 2, 2);
    classes_ = parseClassList(options.event_classes);
    dwell_seconds_ = options.event_dwell_seconds > 0.0 ? options.event_dwell_seconds : 10.0;
    absence_seconds_ = options.event_absence_seconds > 0.0 ? options.event_absence_seconds : 30.0;

    line_cross_enabled_ = options.event_line_cross;
    intrusion_enabled_ = options.event_intrusion;
    dwell_enabled_ = options.event_dwell;
    absence_enabled_ = options.event_absence;

    if (line_cross_enabled_ && line_.size() != 2) {
        warnRule("event_line_cross=1 但 event_line 未配置或格式非法（需 x1,y1,x2,y2），规则已禁用");
        line_cross_enabled_ = false;
    }
    if ((intrusion_enabled_ || dwell_enabled_ || absence_enabled_) && region_.size() < 3) {
        warnRule("入侵/滞留/离岗规则已配置但 event_region 未配置或格式非法（需至少 3 个点），相关规则已禁用");
        intrusion_enabled_ = false;
        dwell_enabled_ = false;
        absence_enabled_ = false;
    }
    active_ = line_cross_enabled_ || intrusion_enabled_ || dwell_enabled_ || absence_enabled_;
}

EventEngine::~EventEngine() {
    extTable().erase(this);
}

bool EventEngine::allowedClass(int cls_id) const {
    return classes_.empty() || classes_.count(cls_id) > 0;
}

bool EventEngine::pointInRegion(const cv::Point2f& p) const {
    return cv::pointPolygonTest(region_, p, false) >= 0;
}

std::vector<RuleEvent> EventEngine::update(const std::vector<TrackedDetection>& tracked,
                                           std::chrono::steady_clock::time_point now) {
    std::vector<RuleEvent> events;
    if (!active_) {
        return events;
    }
    if (EngineExt* ext = extFor(this)) {
        for (auto& rule : ext->rules) {
            updateNamedRule(rule, *ext, stale_seconds_, tracked, now, &events);
            // 轨迹老化：长时间未见的目标清除状态，防止 map 无界增长
            for (auto it = rule.tracks.begin(); it != rule.tracks.end();) {
                const double unseen =
                    std::chrono::duration<double>(now - it->second.last_seen).count();
                if (unseen > stale_seconds_) {
                    it = rule.tracks.erase(it);
                } else {
                    ++it;
                }
            }
        }
        refreshStatsCache(*ext);
        return events;
    }

    bool region_occupied = false;
    for (const auto& td : tracked) {
        if (td.track_id <= 0) {
            continue;
        }
        TrackState& st = tracks_[td.track_id];
        const cv::Point2f center((td.det.box.left + td.det.box.right) * 0.5f,
                                 (td.det.box.top + td.det.box.bottom) * 0.5f);
        const bool class_ok = allowedClass(td.det.cls_id);
        const bool in_reg = class_ok && !region_.empty() && pointInRegion(center);

        // 绊线穿越：上一帧中心与本帧中心分居直线两侧且轨迹线段与绊线相交
        if (line_cross_enabled_ && class_ok && td.is_confirmed && st.has_prev) {
            const double s0 = sideOfLine(line_[0], line_[1], st.prev_center);
            const double s1 = sideOfLine(line_[0], line_[1], center);
            if (s0 != 0.0 && s1 != 0.0 && ((s0 > 0) != (s1 > 0)) &&
                segmentsCross(st.prev_center, center, line_[0], line_[1])) {
                RuleEvent ev;
                ev.type = "line_cross";
                ev.track_id = td.track_id;
                ev.cls_id = td.det.cls_id;
                ev.detail = s1 > 0 ? "A->B" : "B->A";
                if (s1 > 0) {
                    ++cross_ab_;
                } else {
                    ++cross_ba_;
                }
                events.push_back(std::move(ev));
            }
        }

        if (class_ok && !region_.empty()) {
            // 入侵：区域外 → 区域内（已确认轨迹）
            if (intrusion_enabled_ && td.is_confirmed && in_reg && !st.in_region) {
                events.push_back({"intrusion", td.track_id, td.det.cls_id, ""});
            }
            // 滞留：进入后停留超过阈值，每次进入只触发一次
            if (dwell_enabled_ && in_reg) {
                if (!st.in_region) {
                    st.enter_time = now;
                    st.dwell_fired = false;
                } else if (!st.dwell_fired) {
                    const double stayed = std::chrono::duration<double>(now - st.enter_time).count();
                    if (stayed >= dwell_seconds_) {
                        st.dwell_fired = true;
                        events.push_back({"dwell", td.track_id, td.det.cls_id, formatSeconds(stayed)});
                    }
                }
            }
            if (in_reg) {
                region_occupied = true;
            }
        }

        st.in_region = in_reg;
        st.prev_center = center;
        st.has_prev = true;
        st.last_seen = now;
        st.cls_id = td.det.cls_id;
    }

    // 离岗：区域出现过人后持续无人超过阈值，重新有人前只触发一次
    if (absence_enabled_ && !region_.empty()) {
        if (region_occupied) {
            region_ever_occupied_ = true;
            absence_fired_ = false;
            region_empty_timing_ = false;
        } else if (region_ever_occupied_) {
            if (!region_empty_timing_) {
                region_empty_timing_ = true;
                region_empty_since = now;
            } else if (!absence_fired_) {
                const double empty_s = std::chrono::duration<double>(now - region_empty_since).count();
                if (empty_s >= absence_seconds_) {
                    absence_fired_ = true;
                    // 复位区域内目标状态：之后任何目标（含同 ID）再进入都视为再次入侵
                    for (auto& kv : tracks_) {
                        kv.second.in_region = false;
                    }
                    events.push_back({"absence", 0, -1, formatSeconds(empty_s)});
                }
            }
        }
    }

    // 轨迹老化：长时间未见的目标清除状态，防止 map 无界增长
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        const double unseen = std::chrono::duration<double>(now - it->second.last_seen).count();
        if (unseen > stale_seconds_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
    return events;
}

std::vector<RuleStat> EventEngine::stats() const {
    if (EngineExt* ext = extFor(this)) {
        std::lock_guard<std::mutex> lock(ext->stats_mutex);
        return ext->stats_cache;
    }
    // 扁平模式：从冻结成员合成单条快照
    std::vector<RuleStat> out;
    if (!active_) {
        return out;
    }
    RuleStat st;
    st.cross_ab = cross_ab_;
    st.cross_ba = cross_ba_;
    if (line_cross_enabled_) {
        st.type = "line_cross";
    } else if (intrusion_enabled_) {
        st.type = "intrusion";
    } else if (dwell_enabled_) {
        st.type = "dwell";
    } else {
        st.type = "absence";
    }
    out.push_back(std::move(st));
    return out;
}

std::vector<RuleStat> EventEngine::takeStats() {
    std::vector<RuleStat> out = stats();
    EngineExt* ext = extFor(this);
    if (ext) {
        // 清零累计穿越计数（周期报表语义：报表间隔内的增量）；瞬时值保留。
        // 返回的快照携带清零前的累计值；缓存由下次 update() 刷新
        std::lock_guard<std::mutex> lock(ext->stats_mutex);
        for (auto& rule : ext->rules) {
            rule.cross_ab = 0;
            rule.cross_ba = 0;
        }
        for (size_t i = 0; i < ext->rules.size() && i < out.size(); ++i) {
            out[i].crowd_now = ext->rules[i].crowd_count;
            out[i].abandoned_now = ext->rules[i].abandoned_count;
        }
    }
    return out;
}

std::string EventEngine::ruleHint(const std::string& rule_id) const {
    EngineExt* ext = extFor(this);
    if (!ext) {
        return std::string();
    }
    for (const auto& rule : ext->rules) {
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
    if (EngineExt* ext = extFor(this)) {
        drawNamedOverlay(*ext, frame);
        return;
    }
    // 区域：半透明青色填充 + 描边
    if (!region_.empty()) {
        cv::Rect bb = cv::boundingRect(region_) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (bb.width > 0 && bb.height > 0) {
            std::vector<cv::Point> shifted;
            shifted.reserve(region_.size());
            for (const auto& p : region_) {
                shifted.push_back(p - bb.tl());
            }
            cv::Mat mask(bb.size(), CV_8UC1, cv::Scalar(0));
            cv::fillPoly(mask, shifted, cv::Scalar(255));
            cv::Mat tint(bb.size(), frame.type(), cv::Scalar(200, 200, 0));  // BGR 青色
            cv::Mat blended;
            cv::addWeighted(frame(bb), 0.82, tint, 0.18, 0, blended);
            blended.copyTo(frame(bb), mask);
            std::vector<std::vector<cv::Point>> outline{shifted};
            cv::polylines(frame, outline, true, cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, "zone", cv::Point(bb.x + 6, bb.y + 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }
    // 绊线：黄线 + 双向计数
    if (line_.size() == 2) {
        cv::line(frame, line_[0], line_[1], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        for (const auto& p : line_) {
            cv::circle(frame, p, 4, cv::Scalar(0, 255, 255), cv::FILLED);
        }
        char label[48];
        std::snprintf(label, sizeof(label), "A->B:%ld B->A:%ld", cross_ab_, cross_ba_);
        const int lx = std::max(4, std::min(line_[0].x, frame.cols - 200));
        const int ly = std::max(22, line_[0].y - 8);
        cv::putText(frame, label, cv::Point(lx, ly),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
}
