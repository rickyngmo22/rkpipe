#include "detection/yolo11_detector.h"
#include "model/yolov8.h"

// YOLO11 的模型封装与后处理复用 yolov8（head 输出布局一致），见头文件说明。
// 这里只做 Detector 接口的适配与结果类型转换。

YOLO11Detector::YOLO11Detector() : Detector(0.4f) {}

int YOLO11Detector::initModel(const std::string& path) {
    return init_yolov8_model(path.c_str(), &appCtx_);
}

int YOLO11Detector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLO11Detector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<object_detect_result_list*>(results);
    return inference_yolov8_model(&appCtx_, src, lb, res, conf_, nms_);
}

YOLO11PoseDetector::YOLO11PoseDetector() : Detector(0.4f) {}

int YOLO11PoseDetector::initModel(const std::string& path) {
    return init_yolov8_model(path.c_str(), &appCtx_);
}

int YOLO11PoseDetector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLO11PoseDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<pose_detect_result_list*>(results);
    return inference_yolov8_pose_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLO11PoseDetector::extractResultCount(void* results) const {
    return static_cast<pose_detect_result_list*>(results)->count;
}

YOLO11SegDetector::YOLO11SegDetector() : Detector(0.4f) {}

int YOLO11SegDetector::initModel(const std::string& path) {
    return init_yolov8_seg_model(path.c_str(), &appCtx_);
}

int YOLO11SegDetector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLO11SegDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<seg_detect_result_list*>(results);
    return inference_yolov8_seg_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLO11SegDetector::extractResultCount(void* results) const {
    auto* res = static_cast<seg_detect_result_list*>(results);
    return static_cast<int>(res->boxes.size());
}
