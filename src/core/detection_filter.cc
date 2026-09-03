#include "core/detection_filter.h"

#include <cstdio>
#include <sstream>

namespace {

std::vector<int> parseIntList(const std::string& spec) {
    std::vector<int> out;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        out.push_back(std::atoi(token.c_str()));
    }
    return out;
}

// 解析 "x1,y1,x2,y2,..." 为多边形顶点（至少 3 点）
std::vector<cv::Point> parseRoiPolygon(const std::string& spec) {
    std::vector<cv::Point> polygon;
    std::vector<int> values = parseIntList(spec);
    if (values.size() >= 6 && (values.size() % 2) == 0) {
        for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
            polygon.emplace_back(values[i], values[i + 1]);
        }
    }
    return polygon;
}

}  // namespace

DetectionFilter::DetectionFilter(const AppConfig& options) {
    roi_polygon_ = parseRoiPolygon(options.roi);
    has_roi_ = roi_polygon_.size() >= 3;
    mask_classes_ = parseIntList(options.mask_classes);
}

void DetectionFilter::apply(cv::Mat* frame_bgr, object_detect_result_list* results) {
    if (!results) {
        return;
    }

    // 1) ROI 过滤：框中心在 ROI 多边形外的结果移除（含边界）
    if (has_roi_) {
        int kept = 0;
        for (int i = 0; i < results->count; ++i) {
            const object_detect_result& det = results->results[i];
            const cv::Point2f center(
                (det.box.left + det.box.right) / 2.0f,
                (det.box.top + det.box.bottom) / 2.0f);
            const bool inside = cv::pointPolygonTest(roi_polygon_, center, false) >= 0;
            if (inside) {
                if (kept != i) {
                    results->results[kept] = det;
                }
                ++kept;
            }
        }
        if (kept != results->count) {
            std::fprintf(stderr, "[rk_pipe][roi] 过滤 %d 个目标（ROI 外）\n", results->count - kept);
            results->count = kept;
        }
    }

    // 2) 隐私遮蔽：对 mask_classes 类别的框内部区域做高斯模糊（inset 保护框线）
    if (!mask_classes_.empty() && frame_bgr && !frame_bgr->empty() && results->count > 0) {
        for (int i = 0; i < results->count; ++i) {
            const object_detect_result& det = results->results[i];
            bool is_masked = false;
            for (int cls : mask_classes_) {
                if (det.cls_id == cls) {
                    is_masked = true;
                    break;
                }
            }
            if (!is_masked) {
                continue;
            }
            cv::Rect r(det.box.left, det.box.top,
                       det.box.right - det.box.left, det.box.bottom - det.box.top);
            // 向内收缩，避免糊掉检测框线/文字
            r.x += 2;
            r.y += 2;
            r.width -= 4;
            r.height -= 4;
            r &= cv::Rect(0, 0, frame_bgr->cols, frame_bgr->rows);
            if (r.width <= 0 || r.height <= 0) {
                continue;
            }
            cv::Mat roi = (*frame_bgr)(r);
            cv::GaussianBlur(roi, roi, cv::Size(0, 0), 12.0);
        }
    }
}
