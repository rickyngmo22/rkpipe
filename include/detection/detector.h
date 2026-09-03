#pragma once

#include <string>
#include <memory>
#include <algorithm>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "core/rknn_context.h"
#include "core/task_result.h"
#include "postprocess/postprocess.h"
#include "model/rtmpose.h"
#include "model/rtmpose_stage2_pool.h"

class PerformanceMetrics;

struct ModelInfo {
    std::string name;
    int width = 0;
    int height = 0;
    bool isPose = false;
};

class Detector {
public:
    virtual ~Detector();
    virtual bool load(const std::string& path);
    virtual bool detect(const cv::Mat& img, void* results);
    virtual bool detect(const image_buffer_t& img, void* results);
    virtual int getResultCount() const { return resultCount_; }
    virtual ModelInfo info() const { return mi_; }
    virtual bool ready() const { return ok_; }
    virtual void release();
    virtual void setCoreMask(int core_mask);
    virtual void setConfThreshold(float t) { conf_ = std::max(0.f, std::min(1.f, t)); }
    virtual void setNMSThreshold(float t) { nms_ = std::max(0.f, std::min(1.f, t)); }
    virtual void setPreprocessDump(bool enable, const std::string& output_dir);
    virtual void warmup(int input_width, int input_height, int times);

    // 两阶段模型（task="rtmpose"）的第二阶段模型路径；单阶段模型忽略
    virtual void setSecondModelPath(const std::string& path) { (void)path; }

    // rtmpose 拆分部署（feat/head 两段）的 head 模型路径；非拆分部署忽略
    virtual void setRtmposeHeadModelPath(const std::string& path) { (void)path; }

    // rtmpose 两阶段核拆分：stage1(人体检测) 固定 det_core、stage2(pose) 固定 pose_core
    // （RKNN core mask，如 CORE_0=1 / CORE_1_2=6；-1 = 该侧跟随 worker 级 core_mask）。单阶段模型忽略
    virtual void setStageCoreSplit(int det_core, int pose_core) {
        (void)det_core;
        (void)pose_core;
    }

    // rtmpose 单帧工作预算（ms）：单帧处理超过预算时只做 top-K 人（按置信度），
    // 保证帧率稳定在目标值、密集场景平滑降级而不掉帧。<=0 = 不限制。单阶段模型忽略
    virtual void setFrameBudgetMs(float budget_ms) { (void)budget_ms; }

    // rtmpose 二阶段全局工作池线程数（0=内联）。单阶段模型忽略
    virtual void setStage2ThreadCount(int n) { (void)n; }

    // rtmpose 仿射裁剪前框外扩系数（1.0=RTMDet 训练分布；YOLO 框偏紧可调 1.1~1.25）。单阶段模型忽略
    virtual void setRtmposeBoxPad(float pad) { (void)pad; }

    // 模型元信息（用于配置自检：输入尺寸 / 类别数 / 量化类型）
    int modelWidth() const { return mi_.width; }
    int modelHeight() const { return mi_.height; }
    int modelClassNum() const { return appCtx_.class_num; }
    bool modelQuantized() const { return appCtx_.is_quant; }

    virtual double getPreprocessingTime() const;
    virtual double getInferenceTime() const;
    virtual double getPostprocessingTime() const;
    virtual double getTotalTime() const;

    // 平均单帧耗时（累积 / 总帧数，用于 pipeline 多 worker 聚合）
    virtual double getAveragePreprocessingTime() const;
    virtual double getAverageInferenceTime() const;
    virtual double getAveragePostprocessingTime() const;
    virtual double getAverageTotalTime() const;

    static std::unique_ptr<Detector> create();
    static std::unique_ptr<Detector> createForModel(const std::string& model_path, const std::string& task);

protected:
    explicit Detector(float conf_default);

    virtual int initModel(const std::string& path) = 0;
    virtual int releaseModel() = 0;
    virtual int runInference(image_buffer_t* src, letterbox_t* lb, void* results) = 0;
    virtual int extractResultCount(void* results) const;
    virtual bool modelIsPose() const { return false; }
    virtual bool modelIsOBB() const { return false; }
    virtual bool modelIsSeg() const { return false; }
    virtual bool modelIsOCRDet() const { return false; }
    virtual bool modelIsDepth() const { return false; }
    virtual bool modelIsSem() const { return false; }
    virtual bool modelIsDetect3D() const { return false; }

    rknn_app_context_t appCtx_{};
    ModelInfo mi_{};
    bool ok_ = false;
    float conf_ = 0.f;
    float nms_ = NMS_THRESH;
    int resultCount_ = 0;
    std::unique_ptr<PerformanceMetrics> performance_{};
    bool savePreprocess_ = false;
    std::string savePreprocessDir_;
    std::atomic<int> preprocessIndex_{0};
};

class YOLOv8Detector : public Detector {
public:
    YOLOv8Detector();
    ~YOLOv8Detector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
};

class YOLOv8PoseDetector : public Detector {
public:
    YOLOv8PoseDetector();
    ~YOLOv8PoseDetector() override { release(); }

protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsPose() const override { return true; }
};

class YOLOv8OBBDetector : public Detector {
public:
    YOLOv8OBBDetector();
    ~YOLOv8OBBDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsOBB() const override { return true; }
};

class YOLOv5Detector : public Detector {
public:
    YOLOv5Detector();
    ~YOLOv5Detector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
};

// YOLO26 detect（非 end2end，one2one 头）：直接距离回归 + logits + 免 NMS
class YOLOv26Detector : public Detector {
public:
    YOLOv26Detector();
    ~YOLOv26Detector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
};

// YOLO26 pose（非 end2end）：3 x [1,56,H,W]，kpt 按 Pose26.kpts_decode 解码 + 免 NMS
class YOLOv26PoseDetector : public Detector {
public:
    YOLOv26PoseDetector();
    ~YOLOv26PoseDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsPose() const override { return true; }
};

// YOLO26 OBB（非 end2end）：3 x [1,4+nc+1,H,W]，dist2rbox 解码 + angle 弧度 + 免 NMS
class YOLOv26OBBDetector : public Detector {
public:
    YOLOv26OBBDetector();
    ~YOLOv26OBBDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsOBB() const override { return true; }
};

// YOLO26 seg（非 end2end）：3 x [1,4+nc+nm,H,W] + proto，mask = sigmoid(proto @ coeff)
class YOLOv26SegDetector : public Detector {
public:
    YOLOv26SegDetector();
    ~YOLOv26SegDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsSeg() const override { return true; }
};

// YOLO26 depth：单输出 [1,1,H,W]，结果写入 cv::Mat（CV_8UC1，原帧分辨率）
class YOLOv26DepthDetector : public Detector {
public:
    YOLOv26DepthDetector();
    ~YOLOv26DepthDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsDepth() const override { return true; }
};

// YOLO26 sem（语义分割）：单输出 [1,C,H,W]，argmax 得类别索引图（CV_8UC1）
class YOLOv26SemDetector : public Detector {
public:
    YOLOv26SemDetector();
    ~YOLOv26SemDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsSem() const override { return true; }
};

// YOLO26 Detect3D（单目 3D）：单输出 [1,300,14]（2D 框 + 深度/朝向/3D 尺寸），免 NMS
class YOLOv26Detect3DDetector : public Detector {
public:
    YOLOv26Detect3DDetector();
    ~YOLOv26Detect3DDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsDetect3D() const override { return true; }
};

// RTMPose 两阶段姿态（task="rtmpose"）：
//   model_path = RTMPose(SimCC) 模型（外层 appCtx_，供 setCoreMask/元信息）
//   person_model_path = 第一阶段人体检测模型（复用通用 detect Detector，setSecondModelPath 注入）
//   流程：整帧 person 检测 → 每人仿射裁剪 256x192 → SimCC 解码（逆仿射映射回原图）
class RtmposeDetector : public Detector {
public:
    RtmposeDetector();
    ~RtmposeDetector() override { release(); }

    bool detect(const cv::Mat& img, void* results) override;
    bool detect(const image_buffer_t& img, void* results) override;
    void setSecondModelPath(const std::string& path) override { person_model_path_ = path; }
    void setRtmposeHeadModelPath(const std::string& path) override { rtmpose_head_path_ = path; }
    void setCoreMask(int core_mask) override;
    void setStageCoreSplit(int det_core, int pose_core) override;
    void setFrameBudgetMs(float budget_ms) override;
    void setStage2ThreadCount(int n) override;
    void setRtmposeBoxPad(float pad) override;
    void setConfThreshold(float t) override;
    void setNMSThreshold(float t) override;
    void setPreprocessDump(bool enable, const std::string& output_dir) override;
    void warmup(int input_width, int input_height, int times) override;

protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    // 两阶段不走基类"整帧预处理→runInference"通路（detect 已完全重载），仅作占位
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsPose() const override { return true; }
    // 打印 stage1(人体检测) 与 stage2(feat+head) 的耗时分布（每 worker 独立统计）
    void printStageTiming() const;

private:
    bool detectFrame(const cv::Mat& bgr, void* results);
    bool detectFrameBuffer(const image_buffer_t& img, void* results);
    // 逐人两阶段推理（crop 已裁好）或带 buf 的 RGA ROI 直裁路径；buf 为空回退 bgr CPU warpAffine。
    // stage1_ms 用于单帧预算：超预算只处理 top-K 人。
    bool runPersons(const cv::Mat& bgr, const image_buffer_t* buf, float stage1_ms,
                    const object_detect_result_list& od, pose_detect_result_list* out);
    bool runPersonCrop(const cv::Mat& crop_rgb, const rtmpose_affine_t& aff, const image_rect_t& box,
                       float box_conf, pose_detect_result_list* out);
    // 单人推理（feat→head→SimCC 解码，按 head 部署模式分派）+ stage2 计时统计
    int inferPersonInline(const cv::Mat& crop_rgb, const rtmpose_affine_t& aff,
                          rtmpose_person_result_t* person);
    bool runPerson(const cv::Mat& bgr, const image_rect_t& box, float box_conf, pose_detect_result_list* out);
    cv::Mat affineCrop(const cv::Mat& bgr, const rtmpose_affine_t& aff) const;

    std::string person_model_path_;
    std::string rtmpose_head_path_;
    std::unique_ptr<Detector> stage1_;
    rknn_app_context_t head_ctx_{};
    rtmpose_head_cpu_t head_cpu_{};  // CPU 复刻 head（rtmpose_head_path 指向权重目录时使用）
    bool dump_enable_ = false;
    std::string dump_dir_;
    // 两阶段核拆分（setStageCoreSplit 注入；-1 = 该侧跟随 worker 级 core_mask）
    int det_core_ = -1;
    int pose_core_ = -1;
    // 单帧工作预算（ms，setFrameBudgetMs 注入；<=0 = 不限制）与 stage2 人均耗时 EMA（用于预算换算 K）
    float frame_budget_ms_ = 0.0f;
    double stage2_ema_us_ = 0.0;
    // 二阶段全局工作池（setStage2ThreadCount >0 且 CPU head 部署时创建；0=内联）。
    // 懒创建：initModel 时 setter 还没注入线程数，改在首次 runPersons 时建。
    int stage2_threads_ = 0;
    rtmpose_stage2_pool_t* stage2_pool_ = nullptr;
    std::string feat_model_path_;  // 懒创建池时需要（initModel 的 path）
    float box_pad_ = 1.0f;         // 仿射裁剪前框外扩系数（setRtmposeBoxPad 注入）
    // 分阶段耗时统计（微秒累加）
    mutable std::atomic<int64_t> stage1_accum_us_{0};
    mutable std::atomic<int64_t> stage2_accum_us_{0};
    mutable std::atomic<int64_t> frame_count_{0};
    mutable std::atomic<int64_t> person_count_{0};       // 实际送检人数（预算截断后）
    mutable std::atomic<int64_t> detected_count_{0};     // 检测到的人数（预算截断前）
};

class YOLOv8SegDetector : public Detector {
public:
    YOLOv8SegDetector();
    ~YOLOv8SegDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsSeg() const override { return true; }
};

class OCRDetectDetector : public Detector {
public:
    OCRDetectDetector();
    ~OCRDetectDetector() override { release(); }
protected:
    int initModel(const std::string& path) override;
    int releaseModel() override;
    int runInference(image_buffer_t* src, letterbox_t* lb, void* results) override;
    int extractResultCount(void* results) const override;
    bool modelIsOCRDet() const override { return true; }
};
