#pragma once

#include <string>
#include <memory>
#include <algorithm>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "core/rknn_context.h"
#include "core/task_result.h"
#include "postprocess/postprocess.h"

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
    virtual bool modelIsDepth() const { return false; }

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

