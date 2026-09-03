#include "detection/detector.h"
#include "detection/detection.h"
#include "ocr_det.h"
#include "yolov5.h"
#include "yolov8.h"
#include "yolov26.h"
#include "postprocess/postprocess.h"
#include "core/thread_local_memory_pool.h"
#include "core/performance.h"
#include "utils.h"
#include <filesystem>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cctype>

std::unique_ptr<Detector> Detector::create() {
    return std::make_unique<YOLOv8Detector>();
}

static std::string toLowerCopy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

static std::string buildPreprocessImagePath(const std::string& dir, int index) {
    char name[64];
    snprintf(name, sizeof(name), "preprocess_%06d.jpg", index);
    if (dir.empty()) {
        return std::string(name);
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::unique_ptr<Detector> Detector::createForModel(const std::string& model_path, const std::string& task) {
    std::string lower_path = toLowerCopy(model_path);
    std::string lower_task = toLowerCopy(task);
    const bool is_yolo26 = lower_path.find("yolo26") != std::string::npos ||
                           lower_path.find("yolov26") != std::string::npos;
    // rtmpose 两阶段：model_path 为 RTMPose(SimCC) 模型，第一阶段人体检测模型
    // 经 setSecondModelPath 注入（见 detector_runtime.cc）。放在最前，
    // 避免 is_yolo26 文件名启发式误判。
    if (lower_task == "rtmpose" || lower_task == "rtm-pose") {
        return std::make_unique<RtmposeDetector>();
    }
    if (lower_task == "pose") {
        if (is_yolo26) {
            return std::make_unique<YOLOv26PoseDetector>();
        }
        return std::make_unique<YOLOv8PoseDetector>();
    }
    if (lower_task == "obb") {
        if (is_yolo26) {
            return std::make_unique<YOLOv26OBBDetector>();
        }
        return std::make_unique<YOLOv8OBBDetector>();
    }
    if (lower_task == "seg") {
        if (is_yolo26) {
            return std::make_unique<YOLOv26SegDetector>();
        }
        return std::make_unique<YOLOv8SegDetector>();
    }
    if (lower_task == "ocr_det" || lower_task == "ocr-det" || lower_task == "text_det") {
        return std::make_unique<OCRDetectDetector>();
    }
    if (lower_task == "depth") {
        return std::make_unique<YOLOv26DepthDetector>();
    }
    if (lower_task == "sem") {
        return std::make_unique<YOLOv26SemDetector>();
    }
    if (lower_task == "detect3d" || lower_task == "detect_3d") {
        return std::make_unique<YOLOv26Detect3DDetector>();
    }
    if (lower_task == "yolo26" || lower_task == "yolov26" || is_yolo26) {
        return std::make_unique<YOLOv26Detector>();
    }
    if (lower_path.find("yolov5") != std::string::npos) {
        return std::make_unique<YOLOv5Detector>();
    }
    return std::make_unique<YOLOv8Detector>();
}

Detector::Detector(float conf_default) : conf_(conf_default) {
    performance_ = std::make_unique<PerformanceMetrics>();
}

Detector::~Detector() = default;

bool Detector::load(const std::string& path) {
    if (ok_) return true;

    int ret = initModel(path);
    if (ret < 0) {
        std::cerr << "[Detector] Load failed: " << path << std::endl;
        return false;
    }

    mi_.name = path;
    mi_.width = appCtx_.model_width;
    mi_.height = appCtx_.model_height;
    mi_.isPose = modelIsPose();

    ok_ = true;
    std::cout << "[Detector] " << path << " (" << mi_.width << "x" << mi_.height << ")" << std::endl;
    return true;
}

bool Detector::detect(const cv::Mat& img, void* results) {
    if (!ok_ || img.empty()) return false;

    performance_->startTotal();

    image_buffer_t src{};
    letterbox_t lb{};

    Detection det;
    det.setModelContext(&appCtx_);

    performance_->startPreprocessing();
    if (!det.preprocess(img, src, lb, mi_.width, mi_.height)) {
        performance_->stopPreprocessing();
        performance_->stopTotal();
        return false;
    }
    performance_->stopPreprocessing();

    if (savePreprocess_ && !savePreprocessDir_.empty() && src.virt_addr) {
        int index = preprocessIndex_.fetch_add(1);
        std::string path = buildPreprocessImagePath(savePreprocessDir_, index);
        write_image(path.c_str(), &src);
    }

    performance_->startInference();
    int ret = runInference(&src, &lb, results);
    if (ret == 0) {
        resultCount_ = extractResultCount(results);
    }
    performance_->stopInference();

    performance_->startPostprocessing();
    if (src.virt_addr && !src.priv_data) {
        ThreadLocalMemoryManager::deallocate(src.virt_addr);
    }
    performance_->stopPostprocessing();

    performance_->stopTotal();
    performance_->addMeasurement();

    return true;
}

bool Detector::detect(const image_buffer_t& img, void* results) {
    if (!ok_ || (img.virt_addr == nullptr && img.fd <= 0)) return false;

    performance_->startTotal();

    image_buffer_t src{};
    letterbox_t lb{};

    Detection det;
    det.setModelContext(&appCtx_);

    performance_->startPreprocessing();
    if (!det.preprocess(img, src, lb, mi_.width, mi_.height)) {
        performance_->stopPreprocessing();
        performance_->stopTotal();
        return false;
    }
    performance_->stopPreprocessing();

    if (savePreprocess_ && !savePreprocessDir_.empty() && src.virt_addr) {
        int index = preprocessIndex_.fetch_add(1);
        std::string path = buildPreprocessImagePath(savePreprocessDir_, index);
        write_image(path.c_str(), &src);
    }

    performance_->startInference();
    int ret = runInference(&src, &lb, results);
    if (ret == 0) {
        resultCount_ = extractResultCount(results);
    }
    performance_->stopInference();

    performance_->startPostprocessing();
    if (src.virt_addr && !src.priv_data) {
        ThreadLocalMemoryManager::deallocate(src.virt_addr);
    }
    performance_->stopPostprocessing();

    performance_->stopTotal();
    performance_->addMeasurement();

    return true;
}

void Detector::release() {
    releaseModel();
    performance_.reset();
    ok_ = false;
}

void Detector::setCoreMask(int core_mask) {
    if (appCtx_.rknn_ctx) {
        int ret = rknn_set_core_mask(appCtx_.rknn_ctx, (rknn_core_mask)core_mask);
        if (ret < 0) {
            std::cerr << "[Detector] Set core mask failed: " << ret << std::endl;
        } else {
            std::cout << "[Detector] Set core mask to " << core_mask << " successfully" << std::endl;
        }
    }
}

void Detector::setPreprocessDump(bool enable, const std::string& output_dir) {
    savePreprocess_ = enable;
    savePreprocessDir_ = output_dir;
    preprocessIndex_.store(0);
    if (savePreprocess_ && !savePreprocessDir_.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(savePreprocessDir_, ec);
    }
}

void Detector::warmup(int input_width, int input_height, int times) {
    if (!ok_ || times <= 0) {
        return;
    }

    bool prev_save = savePreprocess_;
    std::string prev_dir = savePreprocessDir_;
    savePreprocess_ = false;
    savePreprocessDir_.clear();
    preprocessIndex_.store(0);

    cv::Mat dummy(input_height, input_width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < times; ++i) {
        image_buffer_t src{};
        letterbox_t lb{};

        Detection det;
        det.setModelContext(&appCtx_);

        if (!det.preprocess(dummy, src, lb, mi_.width, mi_.height)) {
            continue;
        }

        if (modelIsPose()) {
            pose_detect_result_list pose_results;
            runInference(&src, &lb, &pose_results);
        } else if (modelIsOBB()) {
            obb_detect_result_list obb_results;
            runInference(&src, &lb, &obb_results);
        } else if (modelIsSeg()) {
            seg_detect_result_list seg_results;
            runInference(&src, &lb, &seg_results);
        } else if (modelIsOCRDet()) {
            OCRDetectTaskResult ocr_results;
            runInference(&src, &lb, &ocr_results);
        } else if (modelIsDepth()) {
            DepthTaskResult depth;
            runInference(&src, &lb, &depth);
        } else if (modelIsSem()) {
            SemTaskResult sem;
            runInference(&src, &lb, &sem);
        } else if (modelIsDetect3D()) {
            Detect3DTaskResult d3;
            runInference(&src, &lb, &d3);
        } else {
            object_detect_result_list obj_results;
            runInference(&src, &lb, &obj_results);
        }

        if (src.virt_addr && !src.priv_data) {
            ThreadLocalMemoryManager::deallocate(src.virt_addr);
        }
    }

    savePreprocess_ = prev_save;
    savePreprocessDir_ = prev_dir;
}

double Detector::getPreprocessingTime() const {
    return performance_->getPreprocessingTime();
}

double Detector::getInferenceTime() const {
    return performance_->getInferenceTime();
}

double Detector::getPostprocessingTime() const {
    return performance_->getPostprocessingTime();
}

double Detector::getTotalTime() const {
    return performance_->getTotalTime();
}

// release() 会 reset performance_，此后调用 getAverage* 返回 0，避免空指针解引用
double Detector::getAveragePreprocessingTime() const {
    return performance_ ? performance_->getAveragePreprocessingTime() : 0.0;
}

double Detector::getAverageInferenceTime() const {
    return performance_ ? performance_->getAverageInferenceTime() : 0.0;
}

double Detector::getAveragePostprocessingTime() const {
    return performance_ ? performance_->getAveragePostprocessingTime() : 0.0;
}

double Detector::getAverageTotalTime() const {
    return performance_ ? performance_->getAverageTotalTime() : 0.0;
}

int Detector::extractResultCount(void* results) const {
    return static_cast<object_detect_result_list*>(results)->count;
}

YOLOv8Detector::YOLOv8Detector() : Detector(0.4f) {}

int YOLOv8Detector::initModel(const std::string& path) {
    return init_yolov8_model(path.c_str(), &appCtx_);
}

int YOLOv8Detector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLOv8Detector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<object_detect_result_list*>(results);
    return inference_yolov8_model(&appCtx_, src, lb, res, conf_, nms_);
}

YOLOv8PoseDetector::YOLOv8PoseDetector() : Detector(0.4f) {}

int YOLOv8PoseDetector::initModel(const std::string& path) {
    return init_yolov8_model(path.c_str(), &appCtx_);
}

int YOLOv8PoseDetector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLOv8PoseDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<pose_detect_result_list*>(results);
    return inference_yolov8_pose_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv8PoseDetector::extractResultCount(void* results) const {
    return static_cast<pose_detect_result_list*>(results)->count;
}

YOLOv8OBBDetector::YOLOv8OBBDetector() : Detector(0.4f) {}

int YOLOv8OBBDetector::initModel(const std::string& path) {
    return init_yolov8_obb_model(path.c_str(), &appCtx_);
}

int YOLOv8OBBDetector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLOv8OBBDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<obb_detect_result_list*>(results);
    return inference_yolov8_obb_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv8OBBDetector::extractResultCount(void* results) const {
    return static_cast<obb_detect_result_list*>(results)->count;
}

YOLOv5Detector::YOLOv5Detector() : Detector(0.5f) {}

int YOLOv5Detector::initModel(const std::string& path) {
    return init_yolov5_model(path.c_str(), &appCtx_);
}

int YOLOv5Detector::releaseModel() {
    return release_yolov5_model(&appCtx_);
}

int YOLOv5Detector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<object_detect_result_list*>(results);
    return inference_yolov5_model(&appCtx_, src, lb, res, conf_, nms_);
}

YOLOv26Detector::YOLOv26Detector() : Detector(0.4f) {}

int YOLOv26Detector::initModel(const std::string& path) {
    return init_yolov26_model(path.c_str(), &appCtx_);
}

int YOLOv26Detector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26Detector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<object_detect_result_list*>(results);
    return inference_yolov26_model(&appCtx_, src, lb, res, conf_, nms_);
}

YOLOv26PoseDetector::YOLOv26PoseDetector() : Detector(0.4f) {}

int YOLOv26PoseDetector::initModel(const std::string& path) {
    return init_yolov26_pose_model(path.c_str(), &appCtx_);
}

int YOLOv26PoseDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26PoseDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<pose_detect_result_list*>(results);
    return inference_yolov26_pose_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv26PoseDetector::extractResultCount(void* results) const {
    return static_cast<pose_detect_result_list*>(results)->count;
}

YOLOv26OBBDetector::YOLOv26OBBDetector() : Detector(0.4f) {}

int YOLOv26OBBDetector::initModel(const std::string& path) {
    return init_yolov26_obb_model(path.c_str(), &appCtx_);
}

int YOLOv26OBBDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26OBBDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<obb_detect_result_list*>(results);
    return inference_yolov26_obb_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv26OBBDetector::extractResultCount(void* results) const {
    return static_cast<obb_detect_result_list*>(results)->count;
}

YOLOv26SegDetector::YOLOv26SegDetector() : Detector(0.4f) {}

int YOLOv26SegDetector::initModel(const std::string& path) {
    return init_yolov26_seg_model(path.c_str(), &appCtx_);
}

int YOLOv26SegDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26SegDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<seg_detect_result_list*>(results);
    return inference_yolov26_seg_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv26SegDetector::extractResultCount(void* results) const {
    return static_cast<seg_detect_result_list*>(results)->boxes.size();
}

YOLOv26DepthDetector::YOLOv26DepthDetector() : Detector(0.4f) {}

int YOLOv26DepthDetector::initModel(const std::string& path) {
    return init_yolov26_depth_model(path.c_str(), &appCtx_);
}

int YOLOv26DepthDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26DepthDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<DepthTaskResult*>(results);
    const int ret = inference_yolov26_depth_model(&appCtx_, src, lb, res, conf_, nms_);
    // depth 有效区子图在原帧中的目标位置（绘制时放大到该区域）
    if (lb && lb->scale > 0.0f) {
        res->roi = cv::Rect(lb->crop_x, lb->crop_y, lb->crop_w, lb->crop_h);
    } else if (src && src->width > 0) {
        res->roi = cv::Rect(0, 0, src->width, src->height);
    } else {
        res->roi = cv::Rect(0, 0, 0, 0);
    }
    return ret;
}

int YOLOv26DepthDetector::extractResultCount(void* results) const {
    auto* res = static_cast<cv::Mat*>(results);
    return res->empty() ? 0 : 1;
}

YOLOv26SemDetector::YOLOv26SemDetector() : Detector(0.4f) {}

int YOLOv26SemDetector::initModel(const std::string& path) {
    return init_yolov26_sem_model(path.c_str(), &appCtx_);
}

int YOLOv26SemDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26SemDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<SemTaskResult*>(results);
    const int ret = inference_yolov26_sem_model(&appCtx_, src, lb, &res->class_map, conf_, nms_);
    res->class_num = appCtx_.class_num;
    // 类别索引图有效区子图在原帧中的目标位置（绘制时放大到该区域）
    if (lb && lb->scale > 0.0f) {
        res->roi = cv::Rect(lb->crop_x, lb->crop_y, lb->crop_w, lb->crop_h);
    } else if (src && src->width > 0) {
        res->roi = cv::Rect(0, 0, src->width, src->height);
    } else {
        res->roi = cv::Rect(0, 0, 0, 0);
    }
    return ret;
}

int YOLOv26SemDetector::extractResultCount(void* results) const {
    auto* res = static_cast<SemTaskResult*>(results);
    return res->class_map.empty() ? 0 : 1;
}

YOLOv26Detect3DDetector::YOLOv26Detect3DDetector() : Detector(0.25f) {}

int YOLOv26Detect3DDetector::initModel(const std::string& path) {
    return init_yolov26_detect3d_model(path.c_str(), &appCtx_);
}

int YOLOv26Detect3DDetector::releaseModel() {
    return release_yolov26_model(&appCtx_);
}

int YOLOv26Detect3DDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<Detect3DTaskResult*>(results);
    const int ret = inference_yolov26_detect3d_model(&appCtx_, src, lb, res, conf_, nms_);
    if (ret == 0 && !res->items.empty()) {
        // 后处理保持模型输入坐标，这里做 letterbox 逆映射。
        // 注意：src 是预处理后的模型尺寸 buffer，原帧尺寸必须从 letterbox 的 crop 字段取
        // （crop_x/crop_y 为有效区在原帧中的偏移，crop_w/crop_h 为原帧有效区宽高）
        const int model_w = mi_.width > 0 ? mi_.width : appCtx_.model_width;
        const int model_h = mi_.height > 0 ? mi_.height : appCtx_.model_height;
        int frame_w = model_w;
        int frame_h = model_h;
        if (lb && lb->crop_w > 0 && lb->crop_h > 0) {
            frame_w = lb->crop_x + lb->crop_w;
            frame_h = lb->crop_y + lb->crop_h;
        } else if (src && src->width > 0) {
            frame_w = src->width;
            frame_h = src->height;
        }
        static const bool dbg = []() {
            const char* e = getenv("RK_PIPE_DEBUG_D3D");
            return e && *e && strcmp(e, "0") != 0;
        }();
        if (dbg) {
            std::printf("[d3d-map] model=%dx%d frame=%dx%d lb(scale=%.4f pad=%.1f,%.1f crop=%d,%d,%d,%d) items=%zu\n",
                        model_w, model_h, frame_w, frame_h,
                        lb ? lb->scale : -1.f, lb ? lb->x_pad : -1.f, lb ? lb->y_pad : -1.f,
                        lb ? lb->crop_x : -1, lb ? lb->crop_y : -1, lb ? lb->crop_w : -1, lb ? lb->crop_h : -1,
                        res->items.size());
            for (size_t k = 0; k < res->items.size() && k < 2; ++k) {
                const auto& it = res->items[k];
                std::printf("[d3d-map]   pre : box=(%d,%d,%d,%d) d=%.2f hwl=(%.2f,%.2f,%.2f) conf=%.3f cls=%d\n",
                            it.box.left, it.box.top, it.box.right, it.box.bottom,
                            (double)it.depth_m, (double)it.h3, (double)it.w3, (double)it.l3,
                            (double)it.conf, it.cls_id);
            }
        }
        for (auto& item : res->items) {
            detect3dMapToFrame(item, lb, model_w, model_h, frame_w, frame_h);
        }
        if (dbg) {
            for (size_t k = 0; k < res->items.size() && k < 2; ++k) {
                const auto& it = res->items[k];
                std::printf("[d3d-map]   post: box=(%d,%d,%d,%d)\n",
                            it.box.left, it.box.top, it.box.right, it.box.bottom);
            }
        }
    }
    return ret;
}

int YOLOv26Detect3DDetector::extractResultCount(void* results) const {
    auto* res = static_cast<Detect3DTaskResult*>(results);
    return static_cast<int>(res->items.size());
}

YOLOv8SegDetector::YOLOv8SegDetector() : Detector(0.4f) {}

int YOLOv8SegDetector::initModel(const std::string& path) {
    return init_yolov8_seg_model(path.c_str(), &appCtx_);
}

int YOLOv8SegDetector::releaseModel() {
    return release_yolov8_model(&appCtx_);
}

int YOLOv8SegDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<seg_detect_result_list*>(results);
    return inference_yolov8_seg_model(&appCtx_, src, lb, res, conf_, nms_);
}

int YOLOv8SegDetector::extractResultCount(void* results) const {
    auto* res = static_cast<seg_detect_result_list*>(results);
    return static_cast<int>(res->boxes.size());
}

OCRDetectDetector::OCRDetectDetector() : Detector(0.3f) {}

int OCRDetectDetector::initModel(const std::string& path) {
    return init_ocr_det_model(path.c_str(), &appCtx_);
}

int OCRDetectDetector::releaseModel() {
    return release_ocr_det_model(&appCtx_);
}

int OCRDetectDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<OCRDetectTaskResult*>(results);
    return inference_ocr_det_model(&appCtx_, src, lb, res, conf_, nms_);
}

int OCRDetectDetector::extractResultCount(void* results) const {
    auto* res = static_cast<OCRDetectTaskResult*>(results);
    return static_cast<int>(res->polygons.size());
}
