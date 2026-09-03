#include "core/event_engine.h"

#include <cctype>
#include <cmath>
#include <cstdio>
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

void warnRule(const char* message) {
    std::fprintf(stderr, "[rk_pipe][event] %s\n", message);
}

}  // namespace

EventEngine::EventEngine(const AppConfig& options) {
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

void EventEngine::drawOverlay(cv::Mat& frame) const {
    if (frame.empty()) {
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
