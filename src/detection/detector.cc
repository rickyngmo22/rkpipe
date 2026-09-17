#include "detection/detector.h"
#include "detection/detection.h"
#include "detection/yolo11_detector.h"
#include "model/ocr_det.h"
#include "model/yolov5.h"
#include "model/yolov8.h"
#include "model/yolov26.h"
#include "model/ocr_rec.h"
#include "model/composite_cls.h"
#include "model/retinaface.h"
#include "postprocess/cls_top1.h"
#include "postprocess/postprocess.h"
#include "core/thread_local_memory_pool.h"
#include "core/performance.h"
#include "preprocess.h"
#include "utils/utils.h"
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
    const bool is_yolo11 = lower_path.find("yolo11") != std::string::npos ||
                           lower_path.find("yolov11") != std::string::npos;
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
        if (is_yolo11) {
            return std::make_unique<YOLO11PoseDetector>();
        }
        return std::make_unique<YOLOv8PoseDetector>();
    }
    if (lower_task == "obb") {
        if (is_yolo26) {
            return std::make_unique<YOLOv26OBBDetector>();
        }
        // yolo11 无 obb 分支，落回 yolov8 obb（head 布局一致）
        return std::make_unique<YOLOv8OBBDetector>();
    }
    if (lower_task == "seg") {
        if (is_yolo26) {
            return std::make_unique<YOLOv26SegDetector>();
        }
        if (is_yolo11) {
            return std::make_unique<YOLO11SegDetector>();
        }
        return std::make_unique<YOLOv8SegDetector>();
    }
    if (lower_task == "ocr_det" || lower_task == "ocr-det" || lower_task == "text_det") {
        return std::make_unique<OCRDetectDetector>();
    }
    if (lower_task == "composite_cls" || lower_task == "detect_cls") {
        return std::make_unique<CompositeClsDetector>();
    }
    if (lower_task == "retinaface" || lower_task == "face") {
        return std::make_unique<RetinaFaceDetector>();
    }
    if (lower_task == "depth") {
        return std::make_unique<YOLOv26DepthDetector>();
    }
    if (lower_task == "sem") {
        return std::make_unique<YOLOv26SemDetector>();
    }
    if (lower_task == "yolo26" || lower_task == "yolov26" || is_yolo26) {
        return std::make_unique<YOLOv26Detector>();
    }
    if (lower_path.find("yolov5") != std::string::npos) {
        return std::make_unique<YOLOv5Detector>();
    }
    if (is_yolo11) {
        return std::make_unique<YOLO11Detector>();
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
    // 调试：RK_PIPE_DUMP_PREPROCESS=<目录> 时把 letterbox 后的模型输入落盘
    // （排查预处理通道序/填充/尺寸问题；warmup 帧自动跳过）
    if (const char* dump_dir = std::getenv("RK_PIPE_DUMP_PREPROCESS")) {
        if (dump_dir[0] != '\0') {
            setPreprocessDump(true, dump_dir);
        }
    }
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

RetinaFaceDetector::RetinaFaceDetector() : Detector(0.5f) {}

int RetinaFaceDetector::initModel(const std::string& path) {
    return init_retinaface_model(path.c_str(), &appCtx_);
}

int RetinaFaceDetector::releaseModel() {
    return release_retinaface_model(&appCtx_);
}

int RetinaFaceDetector::runInference(image_buffer_t* src, letterbox_t* lb, void* results) {
    auto* res = static_cast<FaceTaskResult*>(results);
    return inference_retinaface_model(&appCtx_, src, lb, res, conf_, nms_);
}

int RetinaFaceDetector::extractResultCount(void* results) const {
    auto* res = static_cast<FaceTaskResult*>(results);
    return static_cast<int>(res->faces.size());
}

CompositeClsDetector::CompositeClsDetector() : Detector(0.4f) {}

int CompositeClsDetector::initModel(const std::string& path) {
    // 一级：组合现有 detect Detector（自动路由 v5/v8/11/26 后处理）
    stage1_ = Detector::createForModel(path, "detect");
    if (!stage1_) {
        std::fprintf(stderr, "[composite-cls] create stage1 detector failed\n");
        return -1;
    }
    if (!stage1_->load(path)) {
        std::fprintf(stderr, "[composite-cls] stage1 load failed: %s\n", path.c_str());
        return -1;
    }
    mi_ = stage1_->info();  // 元信息/checkModelMetadata 以一级检测模型为准
    return 0;
}

int CompositeClsDetector::releaseModel() {
    // stage1_ 由 unique_ptr 释放；这里只清二级
    return 0;
}

int CompositeClsDetector::runInference(image_buffer_t* /*src*/, letterbox_t* /*lb*/, void* /*results*/) {
    return -1;  // 占位：detect 已完全重载
}

int CompositeClsDetector::extractResultCount(void* results) const {
    auto* res = static_cast<CompositeClsTaskResult*>(results);
    return static_cast<int>(res->data.count);
}

void CompositeClsDetector::release() {
    if (cls_ctx_.rknn_ctx) {
        release_composite_cls_model(&cls_ctx_);
    }
    cls_ready_ = false;
    labels_.clear();
    stage1_.reset();
    orig_bgr_.release();
    Detector::release();
}

void CompositeClsDetector::setCoreMask(int core_mask) {
    if (stage1_) {
        stage1_->setCoreMask(core_mask);
    }
    Detector::setCoreMask(core_mask);  // cls 模型用外层 appCtx_
}

void CompositeClsDetector::warmup(int input_width, int input_height, int times) {
    if (stage1_) {
        stage1_->warmup(input_width, input_height, times);
    }
    if (ensureClsLoaded()) {
        cv::Mat warm(cls_ctx_.model_height, cls_ctx_.model_width, CV_8UC3, cv::Scalar(114, 114, 114));
        ClsTop1 top;
        for (int i = 0; i < 2 && i < times; ++i) {
            inference_composite_cls_model(&cls_ctx_, warm, &top);
        }
    }
}

double CompositeClsDetector::getAveragePreprocessingTime() const {
    return stage1_ ? stage1_->getAveragePreprocessingTime() : 0.0;
}
double CompositeClsDetector::getAverageInferenceTime() const {
    return stage1_ ? stage1_->getAverageInferenceTime() : 0.0;
}
double CompositeClsDetector::getAveragePostprocessingTime() const {
    return stage1_ ? stage1_->getAveragePostprocessingTime() : 0.0;
}
double CompositeClsDetector::getAverageTotalTime() const {
    return stage1_ ? stage1_->getAverageTotalTime() : 0.0;
}
double CompositeClsDetector::getPreprocessingTime() const {
    return stage1_ ? stage1_->getPreprocessingTime() : 0.0;
}
double CompositeClsDetector::getInferenceTime() const {
    return stage1_ ? stage1_->getInferenceTime() : 0.0;
}
double CompositeClsDetector::getPostprocessingTime() const {
    return stage1_ ? stage1_->getPostprocessingTime() : 0.0;
}
double CompositeClsDetector::getTotalTime() const {
    return stage1_ ? stage1_->getTotalTime() : 0.0;
}

bool CompositeClsDetector::ensureClsLoaded() {
    if (cls_ready_) {
        return true;
    }
    if (cls_model_path_.empty()) {
        return false;
    }
    if (labels_.empty() && !labels_path_.empty()) {
        labels_ = loadCtcDict(labels_path_);  // 同款"每行一个"格式
        if (labels_.empty()) {
            std::fprintf(stderr, "[composite-cls] labels empty: %s (展示为 cls N)\n",
                         labels_path_.c_str());
        }
    }
    if (init_composite_cls_model(cls_model_path_.c_str(), &cls_ctx_) != 0) {
        std::fprintf(stderr, "[composite-cls] load cls model failed: %s, cls disabled\n",
                     cls_model_path_.c_str());
        cls_model_path_.clear();  // 不再重复加载
        return false;
    }
    cls_ready_ = true;
    std::fprintf(stderr, "[composite-cls] cls ready: %s (input %dx%d, %d classes, %zu labels)\n",
                 cls_model_path_.c_str(), cls_ctx_.model_width, cls_ctx_.model_height,
                 cls_ctx_.class_num, labels_.size());
    return true;
}

bool CompositeClsDetector::runClsStage(const cv::Mat& frame_bgr, void* results) {
    auto* res = static_cast<CompositeClsTaskResult*>(results);
    if (!res || res->data.count <= 0 || frame_bgr.empty()) {
        return true;
    }
    const int n = std::min<int>(res->data.count, 128);  // 与检测结构容量对齐
    res->cls_ids.assign(n, -1);
    res->cls_scores.assign(n, 0.0f);
    res->cls_labels.assign(n, std::string());
    int cls_hits = 0;
    for (int i = 0; i < n; ++i) {
        const auto& box = res->data.results[i].box;
        const int bw = box.right - box.left;
        const int bh = box.bottom - box.top;
        if (bw < 8 || bh < 8) {
            continue;  // 过小框 crop 无意义
        }
        const int x0 = std::max(0, box.left);
        const int y0 = std::max(0, box.top);
        const int x1 = std::min(frame_bgr.cols, box.right);
        const int y1 = std::min(frame_bgr.rows, box.bottom);
        if (x1 - x0 < 8 || y1 - y0 < 8) {
            continue;
        }
        cv::Mat crop = frame_bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        ClsTop1 top;
        if (inference_composite_cls_model(&cls_ctx_, crop, &top) != 0 || top.cls_id < 0) {
            continue;
        }
        res->cls_ids[i] = top.cls_id;
        res->cls_scores[i] = top.score;
        res->cls_labels[i] =
            (top.cls_id < static_cast<int>(labels_.size()) ? labels_[top.cls_id]
                                                           : "cls " + std::to_string(top.cls_id)) +
            " " + std::to_string(static_cast<int>(top.score * 100)) + "%";
        ++cls_hits;
    }
    if (const char* dbg = std::getenv("RK_PIPE_DEBUG_COMPOSITE"); dbg && *dbg == '1') {
        std::printf("[composite-cls][debug] boxes=%d cls_hits=%d\n", n, cls_hits);
        for (int i = 0; i < n; ++i) {
            if (!res->cls_labels[i].empty()) {
                std::printf("[composite-cls][debug] box %d det_cls=%d -> \"%s\"\n", i,
                            res->data.results[i].cls_id, res->cls_labels[i].c_str());
            }
        }
    }
    return true;
}

bool CompositeClsDetector::detect(const cv::Mat& img, void* results) {
    if (!ensureClsLoaded()) {
        return stage1_->detect(img, results) &&
               false;  // 未配二级时不产出 composite 结果（调用方走 detect 任务分支）
    }
    orig_bgr_ = img;
    auto* res = static_cast<CompositeClsTaskResult*>(results);
    DetectTaskResult det_view;
    if (!stage1_->detect(img, &det_view)) {
        return false;
    }
    res->data = det_view.data;
    resultCount_ = static_cast<int>(res->data.count);
    return runClsStage(orig_bgr_, res);
}

bool CompositeClsDetector::detect(const image_buffer_t& img, void* results) {
    if (!ensureClsLoaded()) {
        DetectTaskResult det_view;
        if (!stage1_->detect(img, &det_view)) {
            return false;
        }
        auto* res0 = static_cast<CompositeClsTaskResult*>(results);
        res0->data = det_view.data;
        return true;
    }
    // NV12 优先 RGA 硬转（与 M6/rtmpose 同路径），失败退 CPU
    if (img.virt_addr) {
        if (img.format == IMAGE_FORMAT_YUV420SP_NV12 || img.format == IMAGE_FORMAT_YUV420SP_NV21) {
            if (rga_nv12_to_bgr(img, orig_bgr_, 1.0) != 0 || orig_bgr_.empty()) {
                orig_bgr_ = cv::Mat();
            }
        }
        if (orig_bgr_.empty()) {
            const int w = img.width;
            const int h = img.height;
            const int hs = img.height_stride > 0 ? img.height_stride : h;
            const size_t step = img.width_stride > 0 ? img.width_stride : static_cast<size_t>(w);
            if (img.format == IMAGE_FORMAT_YUV420SP_NV12 || img.format == IMAGE_FORMAT_YUV420SP_NV21) {
                cv::Mat yuv(hs * 3 / 2, w, CV_8UC1, img.virt_addr, step);
                cv::cvtColor(yuv, orig_bgr_, img.format == IMAGE_FORMAT_YUV420SP_NV12
                                                ? cv::COLOR_YUV2BGR_NV12
                                                : cv::COLOR_YUV2BGR_NV21);
            } else if (img.format == IMAGE_FORMAT_BGR888) {
                orig_bgr_ = cv::Mat(h, w, CV_8UC3, img.virt_addr, step);
            } else if (img.format == IMAGE_FORMAT_RGB888) {
                cv::Mat src(h, w, CV_8UC3, img.virt_addr, step);
                cv::cvtColor(src, orig_bgr_, cv::COLOR_RGB2BGR);
            }
        }
    }
    if (orig_bgr_.empty()) {
        DetectTaskResult det_view;
        if (!stage1_->detect(img, &det_view)) {
            return false;
        }
        auto* res0 = static_cast<CompositeClsTaskResult*>(results);
        res0->data = det_view.data;
        return true;
    }
    auto* res = static_cast<CompositeClsTaskResult*>(results);
    DetectTaskResult det_view;
    if (!stage1_->detect(img, &det_view)) {
        return false;
    }
    res->data = det_view.data;
    resultCount_ = static_cast<int>(res->data.count);
    return runClsStage(orig_bgr_, res);
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

void OCRDetectDetector::release() {
    if (rec_ctx_.rknn_ctx) {
        release_ocr_rec_model(&rec_ctx_);
    }
    rec_ready_ = false;
    dict_.clear();
    orig_bgr_.release();
    Detector::release();
}

bool OCRDetectDetector::ensureRecLoaded() {
    if (rec_ready_) {
        return true;
    }
    if (rec_model_path_.empty()) {
        return false;
    }
    // 字典可能经 setOcrDictPath 在 load 之后注入，每次尝试重读（文件不存在打一次警告）
    if (dict_.empty() && !dict_path_.empty()) {
        dict_ = loadCtcDict(dict_path_);
        if (dict_.empty()) {
            return false;
        }
    }
    if (dict_.empty()) {
        std::fprintf(stderr, "[rk_pipe][ocr-rec] dict not set (ocr_dict_path), rec disabled\n");
        rec_model_path_.clear();  // 不再重复告警
        return false;
    }
    if (init_ocr_rec_model(rec_model_path_.c_str(), &rec_ctx_) != 0) {
        std::fprintf(stderr, "[rk_pipe][ocr-rec] load rec model failed: %s, rec disabled\n",
                     rec_model_path_.c_str());
        rec_model_path_.clear();
        return false;
    }
    rec_ready_ = true;
    std::fprintf(stderr, "[rk_pipe][ocr-rec] rec ready: %s (dict %zu chars, blank=%d, input %dx%d)\n",
                 rec_model_path_.c_str(), dict_.size(), rec_blank_index_,
                 rec_ctx_.model_width, rec_ctx_.model_height);
    return true;
}

cv::Mat OCRDetectDetector::frameBgrForRec(const cv::Mat& img, const image_buffer_t* buf) {
    if (img.empty() && buf && buf->virt_addr) {
        // NV12/NV21 优先走 RGA 硬件转换（与 rtmpose stage2 同路径），失败退 CPU
        if (buf->format == IMAGE_FORMAT_YUV420SP_NV12 || buf->format == IMAGE_FORMAT_YUV420SP_NV21) {
            cv::Mat rga_bgr;
            if (rga_nv12_to_bgr(*buf, rga_bgr, 1.0) == 0 && !rga_bgr.empty()) {
                return rga_bgr;
            }
        }
        const int w = buf->width;
        const int h = buf->height;
        const int hs = buf->height_stride > 0 ? buf->height_stride : h;
        cv::Mat bgr;
        if (buf->format == IMAGE_FORMAT_YUV420SP_NV12 || buf->format == IMAGE_FORMAT_YUV420SP_NV21) {
            const size_t step = buf->width_stride > 0 ? buf->width_stride : static_cast<size_t>(w);
            cv::Mat yuv(hs * 3 / 2, w, CV_8UC1, buf->virt_addr, step);
            cv::cvtColor(yuv, bgr, buf->format == IMAGE_FORMAT_YUV420SP_NV12 ? cv::COLOR_YUV2BGR_NV12
                                                                             : cv::COLOR_YUV2BGR_NV21);
        } else if (buf->format == IMAGE_FORMAT_BGR888) {
            const size_t step = buf->width_stride > 0 ? buf->width_stride : static_cast<size_t>(w) * 3;
            bgr = cv::Mat(h, w, CV_8UC3, buf->virt_addr, step);
        } else if (buf->format == IMAGE_FORMAT_RGB888) {
            const size_t step = buf->width_stride > 0 ? buf->width_stride : static_cast<size_t>(w) * 3;
            cv::Mat src(h, w, CV_8UC3, buf->virt_addr, step);
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
        }
        return bgr;
    }
    return img;
}

bool OCRDetectDetector::runRecStage(const cv::Mat& frame_bgr, void* results) {
    auto* res = static_cast<OCRDetectTaskResult*>(results);
    if (!res || res->polygons.empty() || frame_bgr.empty()) {
        return true;  // 无多边形不算失败（空帧合法）
    }
    res->lines.clear();
    res->lines.reserve(res->polygons.size());
    const bool ocr_dbg = [] { const char* d = std::getenv("RK_PIPE_DEBUG_OCR_DET"); return d && *d == '1'; }();
    for (const auto& poly : res->polygons) {
        // 透视矫正：多边形（TL,TR,BR,BL 语义序）→ 正置矩形，高 48 保持长宽比
        // 外扩沿"多边形各自中心"做：先算本多边形外心，再按比例外推四点
        cv::Point2f pts[4] = {poly.points[0], poly.points[1], poly.points[2], poly.points[3]};
        if (rec_crop_pad_ > 1.0) {
            cv::Point2f c(0, 0);
            for (const auto& p : pts) {
                c += p;
            }
            c *= 0.25f;
            for (auto& p : pts) {
                p = c + (p - c) * static_cast<float>(rec_crop_pad_);
            }
        }
        const float w_top = std::hypot(pts[1].x - pts[0].x,
                                       pts[1].y - pts[0].y);
        const float w_bot = std::hypot(pts[2].x - pts[1].x,
                                       pts[2].y - pts[1].y);
        const float h_left = std::hypot(pts[3].x - pts[0].x,
                                        pts[3].y - pts[0].y);
        int crop_w = std::max(8, static_cast<int>(std::max(w_top, w_bot)));
        int crop_h = std::max(8, static_cast<int>(h_left));
        const cv::Point2f src_pts[4] = {pts[0], pts[1], pts[2], pts[3]};
        const cv::Point2f dst_pts[4] = {{0.0f, 0.0f},
                                        {static_cast<float>(crop_w - 1), 0.0f},
                                        {static_cast<float>(crop_w - 1), static_cast<float>(crop_h - 1)},
                                        {0.0f, static_cast<float>(crop_h - 1)}};
        const cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat crop;
        cv::warpPerspective(frame_bgr, crop, M, cv::Size(crop_w, crop_h));

        OCRTextLine line;
        line.points = poly.points;  // lines 展示用原始多边形（未外扩）
        line.score = poly.score;
        if (inference_ocr_rec_model(&rec_ctx_, crop, dict_, rec_blank_index_, &line) != 0) {
            std::fprintf(stderr, "[rk_pipe][ocr-rec] rec inference failed (skip text)\n");
        }
        if (ocr_dbg) {
            std::printf("[ocr-rec][debug] box %zu score=%.2f text=\"%s\"\n", res->lines.size(),
                        line.score, line.text.c_str());
        }
        res->lines.push_back(std::move(line));
    }
    return true;
}

bool OCRDetectDetector::detect(const cv::Mat& img, void* results) {
    if (!ensureRecLoaded()) {
        return Detector::detect(img, results);  // 未配置 rec：纯检测，行为不变
    }
    orig_bgr_ = img;  // 浅拷贝（refcount），worker 内单线程生命周期安全
    if (!Detector::detect(img, results)) {
        return false;
    }
    return runRecStage(orig_bgr_, results);
}

bool OCRDetectDetector::detect(const image_buffer_t& img, void* results) {
    if (!ensureRecLoaded()) {
        return Detector::detect(img, results);
    }
    orig_bgr_ = frameBgrForRec({}, &img);
    if (orig_bgr_.empty()) {
        // 无法取到 BGR（罕见格式）：退纯检测，不阻塞管线
        return Detector::detect(img, results);
    }
    if (!Detector::detect(img, results)) {
        return false;
    }
    return runRecStage(orig_bgr_, results);
}
