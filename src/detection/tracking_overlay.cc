#include "../../include/detection/tracking_overlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "../../include/utils/draw_utils.h"
#include "../../include/postprocess/postprocess.h"

namespace {

float calcOverlayScaleFactor(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 1.0f;
    }
    float scale_w = static_cast<float>(width) / 1280.0f;
    float scale_h = static_cast<float>(height) / 720.0f;
    float scale = std::min(scale_w, scale_h);
    if (scale < 0.5f) {
        scale = 0.5f;
    }
    return scale;
}

int calcOverlayTextScale(int width, int height) {
    float scale = calcOverlayScaleFactor(width, height);
    return std::max(1, static_cast<int>(std::round(scale)));
}

int calcOverlaySize(int width, int height, int base) {
    float scale = calcOverlayScaleFactor(width, height);
    return std::max(1, static_cast<int>(std::round(base * scale)));
}

void drawCountOverlay(image_buffer_t& frame,
                      const std::unordered_map<int, int>& total_counts) {
    if (total_counts.empty()) {
        return;
    }
    std::vector<std::pair<int, int>> entries(total_counts.begin(), total_counts.end());
    std::sort(entries.begin(), entries.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    int cumulative_total = 0;
    for (const auto& kv : total_counts) {
        cumulative_total += kv.second;
    }
    int text_scale = calcOverlayTextScale(frame.width, frame.height);
    int start_x = calcOverlaySize(frame.width, frame.height, 10);
    int start_y = calcOverlaySize(frame.width, frame.height, 45);
    int line_step = calcOverlaySize(frame.width, frame.height, 14);
    const cv::Scalar kStatBg(45, 45, 45);  // 深灰底条,白字可读
    drawOverlayText(frame, "TOTAL " + std::to_string(cumulative_total), start_x, start_y,
                    cv::Scalar(255, 255, 255), text_scale, kStatBg);
    int max_lines = std::min(5, static_cast<int>(entries.size()));
    for (int i = 0; i < max_lines; ++i) {
        std::string line = classNameFromId(entries[i].first) + " " + std::to_string(entries[i].second);
        drawOverlayText(frame, line, start_x, start_y + (i + 1) * line_step,
                        cv::Scalar(255, 255, 255), text_scale, kStatBg);
    }
}

}  // namespace

std::string classNameFromId(int cls_id) {
    char* name = coco_cls_to_name(cls_id);
    if (name && std::strlen(name) > 0) {
        return std::string(name);
    }
    return "CLS" + std::to_string(cls_id);
}

void drawTrackingOverlay(image_buffer_t& frame,
                         const std::vector<TrackedDetection>& tracked,
                         const std::unordered_map<int, int>& total_counts,
                         bool show_count_overlay) {
    int text_scale = calcOverlayTextScale(frame.width, frame.height);
    int text_offset = calcOverlaySize(frame.width, frame.height, 12);
    static const bool dbg_track = []() {
        const char* v = getenv("RK_PIPE_DEBUG_TRACK");
        return v && *v && strcmp(v, "0") != 0;
    }();
    for (const auto& td : tracked) {
        if (dbg_track) {
            fprintf(stderr, "[OVERLAY] id=%d p=%.2f thresh=%.2f conf=%d pred=%d -> %s\n",
                    td.track_id, td.det.prop, displayThreshold(),
                    td.is_confirmed ? 1 : 0, td.is_predicted ? 1 : 0,
                    (td.det.prop < displayThreshold()) ? "SKIP" : "DRAW");
        }
        if (td.det.prop < displayThreshold()) {
            continue;
        }
        // 未确认目标画框（零拷贝路径由 drawDetectionResultsZeroCopy 画），此处只给确认目标叠 ID
        if (!td.is_confirmed) {
            continue;
        }
        int x = std::max(0, td.det.box.left);
        int y = std::min(frame.height - 1, std::max(0, td.det.box.bottom + text_offset));
        std::string id_text = td.is_predicted ? ("ID " + std::to_string(td.track_id) + "*") : ("ID " + std::to_string(td.track_id));
        // ID 条与轨迹绑定同色(Ultralytics 风格:白字 + 同色底),跨帧稳定
        drawOverlayText(frame, id_text, x, y, cv::Scalar(255, 255, 255), text_scale,
                        classColor(td.track_id));
    }
    if (show_count_overlay) {
        drawCountOverlay(frame, total_counts);
    }
}

void drawTrackingOverlay(cv::Mat& frame,
                         const std::vector<TrackedDetection>& tracked,
                         const std::unordered_map<int, int>& total_counts,
                         bool show_count_overlay) {
    if (frame.empty()) {
        return;
    }
    int thickness = std::max(1, std::min(frame.cols, frame.rows) / 320);
    double font_scale = std::max(0.4, std::min(frame.cols, frame.rows) / 900.0);
    // 调试：RK_PIPE_DEBUG_TRACK=1 打印每个 tracked 目标是否显示及被过滤原因
    static const bool dbg_track = []() {
        const char* v = getenv("RK_PIPE_DEBUG_TRACK");
        return v && *v && strcmp(v, "0") != 0;
    }();
    for (const auto& td : tracked) {
        if (dbg_track) {
            fprintf(stderr, "[OVERLAY] id=%d p=%.2f thresh=%.2f conf=%d pred=%d -> %s\n",
                    td.track_id, td.det.prop, displayThreshold(),
                    td.is_confirmed ? 1 : 0, td.is_predicted ? 1 : 0,
                    (td.det.prop < displayThreshold()) ? "SKIP" : "DRAW");
        }
        if (!td.is_confirmed) {
            continue;  // 框/类别由 drawDetectionResultsBGR(调用方)统一画,此处只叠已确认 ID
        }
        int x1 = std::max(0, td.det.box.left);
        int y2 = std::min(frame.rows - 1, td.det.box.bottom);
        std::string id_text = td.is_predicted ? ("ID " + std::to_string(td.track_id) + "*") : ("ID " + std::to_string(td.track_id));
        putClassLabelBGR(frame, id_text, x1, std::min(frame.rows - 1, y2 + 16),
                         classColor(td.track_id), font_scale, std::max(1, thickness - 1));
    }
    if (!show_count_overlay || total_counts.empty()) {
        return;
    }
    std::vector<std::pair<int, int>> entries(total_counts.begin(), total_counts.end());
    std::sort(entries.begin(), entries.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    int cumulative_total = 0;
    for (const auto& kv : total_counts) {
        cumulative_total += kv.second;
    }
    int y = 24;
    putClassLabelBGR(frame, "TOTAL " + std::to_string(cumulative_total), 10, y - 12,
                     cv::Scalar(30, 30, 30), font_scale, thickness);
    int max_lines = std::min(5, static_cast<int>(entries.size()));
    for (int i = 0; i < max_lines; ++i) {
        y += 20;
        putClassLabelBGR(frame, classNameFromId(entries[i].first) + " " + std::to_string(entries[i].second),
                         10, y - 12, cv::Scalar(30, 30, 30), font_scale, thickness);
    }
}

void drawTrackedIdsBGR(cv::Mat& frame, const std::vector<TrackedDetection>& tracked) {
    if (frame.empty()) {
        return;
    }
    int thickness = std::max(1, std::min(frame.cols, frame.rows) / 320);
    double font_scale = std::max(0.4, std::min(frame.cols, frame.rows) / 900.0);
    for (const auto& td : tracked) {
        if (td.det.prop < displayThreshold() || !td.is_confirmed) {
            continue;
        }
        int x = std::max(0, td.det.box.left);
        int y = std::min(frame.rows - 1, std::max(0, td.det.box.bottom + 16));
        std::string id_text = td.is_predicted ? ("ID " + std::to_string(td.track_id) + "*")
                                              : ("ID " + std::to_string(td.track_id));
        putClassLabelBGR(frame, id_text, x, y, classColor(td.track_id), font_scale,
                         std::max(1, thickness - 1));
    }
}
