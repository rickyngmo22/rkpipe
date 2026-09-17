#pragma once

#include <array>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/postprocess.h"

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

struct OCRTextLine {
    std::array<cv::Point2f, 4> points = {};
    std::string text;
    float score = 0.0f;
};

struct OCRDetectTaskResult {
    std::vector<OCRPolygon> polygons;
    // 文字识别（M6）：配置 ocr_rec_model_path 时逐多边形透视矫正 → rec 推理 → CTC 解码填充；
    // 未配置 rec 时保持为空（仅检测），lines[i].points 与 polygons[i] 一一对应
    std::vector<OCRTextLine> lines;
};

struct OCRTaskResult {
    std::vector<OCRTextLine> lines;
};

// M8 人脸检测（RetinaFace）：人脸框 + 5 点 landmark（模型坐标系，绘制侧已逆映射到原图）
struct FaceItem {
    image_rect_t box{};
    float score = 0.0f;
    cv::Point2f landmarks[5] = {};
};

struct FaceTaskResult {
    std::vector<FaceItem> faces;
};

// M0 两阶级联（detect → crop → cls）：一级检测结果 + 与 results[] 一一对应的二级 top-1。
// cls_labels 为展示文本（标签文件行或 "cls N"），与 results[] 一一对应；未检出框时为空
struct CompositeClsTaskResult {
    object_detect_result_list data{};
    std::vector<int> cls_ids;
    std::vector<float> cls_scores;
    std::vector<std::string> cls_labels;
};

// M13 动作识别（pose→ST-GCN 时序级联）：动作结果附着在 pose 主任务上（同 M0 惯例：
// 内嵌一级结果原样保留 + 附加项）。动作级联激活时（action_model_path 已配置）pose 帧
// 的 result 即本类型；actions 仅在推理帧非空（每轨迹每窗口一次，action_interval 节流）
struct ActionItem {
    int track_id = 0;  // 对齐 pose data 中同 track_id 的框（标签锚点）
    int action_id = -1;
    float score = 0.0f;
};

struct ActionTaskResult {
    pose_detect_result_list data{};
    std::vector<ActionItem> actions;  // 本帧新鲜推理结果；攒窗中/节流间隔内为空
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
                                CompositeClsTaskResult,
                                FaceTaskResult,
                                ActionTaskResult>;

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

// pose 主结果双类型访问：动作级联激活时为 ActionTaskResult（内嵌 pose 数据，M0 惯例），
// 否则为 PoseTaskResult。pose 消费方（导出/payload）统一走这里，免逐处感知级联开关
inline const pose_detect_result_list* getPoseResultList(const TaskResult& result) {
    if (const PoseTaskResult* pose = std::get_if<PoseTaskResult>(&result)) {
        return &pose->data;
    }
    if (const ActionTaskResult* action = std::get_if<ActionTaskResult>(&result)) {
        return &action->data;
    }
    return nullptr;
}

inline ActionTaskResult* getMutableActionTaskResult(TaskResult* result) {
    return result ? std::get_if<ActionTaskResult>(result) : nullptr;
}
