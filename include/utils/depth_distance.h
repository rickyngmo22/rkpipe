#pragma once

// D3 检测+单目测距：depth 8bit 反相图 → 检测框估计距离。
// 纯函数（仅依赖 OpenCV core），无硬件依赖，CI 单测可直接覆盖。
//
// depth 约定（见 postprocess/yolov26_depth.cc）：CV_8UC1，近=亮。
// v=255 ↔ 距离 depth_lo（最近），v=0 ↔ 距离 depth_hi（最远）；
// lo/hi 为该帧归一化范围（米），超出范围的深度被钳到 lo/hi 端点。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/postprocess.h"

// 近距目标（近距告警 JSON 与叠加绘制共用）
struct DepthDistanceTarget {
    int cls_id = -1;
    float meters = 0.0f;
};

// 8bit 反相像素值 → 米；scale 为标定系数（实测米数 / 模型输出米数）
inline float depthPixelToMeters(int v, float depth_lo, float depth_hi, float scale = 1.0f) {
    if (depth_hi <= depth_lo) {
        return 0.0f;
    }
    const float meters = depth_lo + (255.0f - static_cast<float>(v)) * (depth_hi - depth_lo) / 255.0f;
    return meters * scale;
}

// 原帧坐标检测框 → 深度子图坐标（roi 为深度子图在原帧中的位置），越界裁剪。
// 返回 false 表示框与深度子图无有效重叠。
inline bool mapBoxToDepthRect(const image_rect_t& box, const cv::Rect& roi,
                              const cv::Size& depth_size, cv::Rect* out) {
    if (out == nullptr || roi.width <= 0 || roi.height <= 0 ||
        depth_size.width <= 0 || depth_size.height <= 0) {
        return false;
    }
    const int dw = depth_size.width;
    const int dh = depth_size.height;
    const int x1 = static_cast<int>((box.left - roi.x) * dw / roi.width);
    const int y1 = static_cast<int>((box.top - roi.y) * dh / roi.height);
    const int x2 = static_cast<int>((box.right - roi.x) * dw / roi.width);
    const int y2 = static_cast<int>((box.bottom - roi.y) * dh / roi.height);
    const int cx1 = std::max(0, std::min(dw - 1, x1));
    const int cy1 = std::max(0, std::min(dh - 1, y1));
    const int cx2 = std::max(0, std::min(dw - 1, x2));
    const int cy2 = std::max(0, std::min(dh - 1, y2));
    if (cx2 <= cx1 || cy2 <= cy1) {
        return false;
    }
    *out = cv::Rect(cx1, cy1, cx2 - cx1 + 1, cy2 - cy1 + 1);
    return true;
}

// 框内深度中值 → 米：抽样 + nth_element 中值，抗个别噪点/遮挡离群值。
// 返回 false 表示框无效或框内无有效深度像素。
inline bool boxDepthMeters(const cv::Mat& depth, const cv::Rect& roi, const image_rect_t& box,
                           float depth_lo, float depth_hi, float scale, float* meters) {
    if (meters == nullptr || depth.empty() || depth_hi <= depth_lo) {
        return false;
    }
    cv::Rect sub;
    if (!mapBoxToDepthRect(box, roi, depth.size(), &sub)) {
        return false;
    }
    // 抽样步长：单框最多约 4k 样本（1080p 大框时避免逐像素遍历）
    const int step = std::max(1, static_cast<int>(
                                    std::sqrt(static_cast<double>(sub.width) * sub.height / 4096.0)));
    std::vector<int> samples;
    samples.reserve((sub.width / step + 1) * (sub.height / step + 1));
    for (int y = sub.y; y < sub.y + sub.height; y += step) {
        const uint8_t* row = depth.ptr<uint8_t>(y);
        for (int x = sub.x; x < sub.x + sub.width; x += step) {
            samples.push_back(row[x]);
        }
    }
    if (samples.empty()) {
        return false;
    }
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    *meters = depthPixelToMeters(samples[mid], depth_lo, depth_hi, scale);
    return *meters >= 0.0f && *meters <= depth_hi * scale + 1e-3f;
}
