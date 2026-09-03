#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config/app_config.h"
#include "postprocess/postprocess.h"

// detect 结果后处理（在叠加/追踪/计数/告警之前应用）：
//   - ROI 过滤：检测框中心不在 ROI 多边形内的结果被移除
//   - 隐私遮蔽：对 mask_classes 指定的类别，在 BGR 帧上对其框内部区域做高斯模糊
class DetectionFilter {
public:
    explicit DetectionFilter(const AppConfig& options);

    bool active() const { return has_roi_ || !mask_classes_.empty(); }

    // 原地过滤 results（ROI），并可选对 frame_bgr 做遮蔽（模糊 mask 类别框内区域）。
    // frame_bgr 可为空（零拷贝路径），此时只做 ROI 过滤，遮蔽跳过。
    void apply(cv::Mat* frame_bgr, object_detect_result_list* results);

private:
    bool has_roi_ = false;
    std::vector<cv::Point> roi_polygon_;
    std::vector<int> mask_classes_;
};
