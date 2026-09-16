#pragma once

#include "detection/detector.h"

// YOLO11 detector：yolo11 的 detector 逻辑单独承载在本文件
// （include/detection/yolo11_detector.h + src/detection/yolo11_detector.cc）。
//
// yolo11 的 head 输出布局与 yolov8 相同（3 分支 + DFL 解码，pose 多 1 个
// kpt 分支，seg 多 proto 分支），因此当前直接复用 yolov8 的模型封装
// （init/inference_yolov8_*）与后处理（post_process_yolov8 等）；
// 后续若出现 yolo11 专属的解码差异，在本文件的实现里改，不影响 yolov8。

// YOLO11 目标检测（yolo11n.rknn 等）
class YOLO11Detector : public Detector {
public:
    YOLO11Detector();
    ~YOLO11Detector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
};

// YOLO11 姿态（yolo11n_pose.rknn：4 输出，3 检测分支 + kpt 分支）
class YOLO11PoseDetector : public Detector {
public:
    YOLO11PoseDetector();
    ~YOLO11PoseDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsPose() const override { return true; }
};

// YOLO11 分割（yolo11n_seg.rknn：4 输出，3 检测分支 + proto 分支）
class YOLO11SegDetector : public Detector {
public:
    YOLO11SegDetector();
    ~YOLO11SegDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsSeg() const override { return true; }
};
