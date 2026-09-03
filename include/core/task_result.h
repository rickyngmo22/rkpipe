#pragma once

#include <array>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/postprocess.h"
#include "postprocess/detect3d_decode.h"

struct DetectTaskResult {
    object_detect_result_list data = {};
};

struct PoseTaskResult {
    pose_detect_result_list data = {};
};

struct OBBTaskResult {
    obb_detect_result_list data = {};
};

struct SegTaskResult {
    seg_detect_result_list data;
    // 与 data.boxes 对齐的 tracking ID（0=未跟踪）
    std::vector<int> track_ids;
};

// 深度图结果：CV_8UC1，近=亮。
// depth 为"有效区低分辨率"uint8（约 640 内 letterbox 有效区，不放大到原帧），
// roi 为其在原帧中的目标位置（空 = 整帧），绘制时再放大，省掉 worker 端 1080p 后处理。
// depth_lo/depth_hi 为该 8bit 反相图的归一化范围（米），距离文字据此把像素值还原为米制。
struct DepthTaskResult {
    cv::Mat depth;
    cv::Rect roi = cv::Rect(0, 0, 0, 0);
    float depth_lo = 0.0f;
    float depth_hi = 0.0f;
};

// 语义分割结果：CV_8UC1 类别索引图（低分辨率有效区，不放大到原帧）。
// roi 为其在原帧中的目标位置（空 = 整帧），绘制时再放大 + 伪彩，省掉 worker 端 1080p 后处理。
struct SemTaskResult {
    cv::Mat class_map;
    cv::Rect roi = cv::Rect(0, 0, 0, 0);
    int class_num = 0;
};

struct OCRPolygon {
    std::array<cv::Point2f, 4> points = {};
    float score = 0.0f;
};

struct OCRDetectTaskResult {
    std::vector<OCRPolygon> polygons;
};

struct OCRTextLine {
    std::array<cv::Point2f, 4> points = {};
    std::string text;
    float score = 0.0f;
};

struct OCRTaskResult {
    std::vector<OCRTextLine> lines;
};

// 单目 3D 检测（YOLO26-Detect3D）：每目标 2D 框 + 8 个三维量（Detect3DItem 见 detect3d_decode.h）
struct Detect3DTaskResult {
    std::vector<Detect3DItem> items;
};

using TaskResult = std::variant<std::monostate,
                                DetectTaskResult,
                                PoseTaskResult,
                                OBBTaskResult,
                                SegTaskResult,
                                DepthTaskResult,
                                SemTaskResult,
                                OCRDetectTaskResult,
                                OCRTaskResult,
                                Detect3DTaskResult>;

inline const DetectTaskResult* getDetectTaskResult(const TaskResult& result) {
    return std::get_if<DetectTaskResult>(&result);
}

inline DetectTaskResult* getMutableDetectTaskResult(TaskResult* result) {
    return result ? std::get_if<DetectTaskResult>(result) : nullptr;
}

inline const object_detect_result_list* getDetectResultList(const TaskResult& result) {
    const DetectTaskResult* detect_result = getDetectTaskResult(result);
    return detect_result ? &detect_result->data : nullptr;
}

inline const SemTaskResult* getSemTaskResult(const TaskResult& result) {
    return std::get_if<SemTaskResult>(&result);
}

inline const Detect3DTaskResult* getDetect3DTaskResult(const TaskResult& result) {
    return std::get_if<Detect3DTaskResult>(&result);
}
