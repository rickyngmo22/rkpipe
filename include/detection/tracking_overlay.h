#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../utils/common.h"
#include "simple_object_tracker.h"

std::string classNameFromId(int cls_id);

void drawTrackingOverlay(image_buffer_t& frame,
                         const std::vector<TrackedDetection>& tracked,
                         const std::unordered_map<int, int>& total_counts,
                         bool show_count_overlay);

void drawTrackingOverlay(cv::Mat& frame,
                         const std::vector<TrackedDetection>& tracked,
                         const std::unordered_map<int, int>& total_counts,
                         bool show_count_overlay);

// 只在目标框下方叠加 ID 文字（不重复画框/类别标签），供 pose/obb/seg 的 tracking 渲染使用
void drawTrackedIdsBGR(cv::Mat& frame, const std::vector<TrackedDetection>& tracked);
