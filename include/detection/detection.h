#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>
#include "postprocess/postprocess.h"
#include "core/performance.h"

// 前向声明Logger类，避免Qt依赖
class Logger;

// 释放 Detection::preprocess 产出的目标缓冲。缓冲来源由 image_buffer_t::priv_data 约定：
//   零拷贝（priv_data == rknn input_mem）→ 不释放，归 RKNN 上下文所有；
//   priv_data == nullptr                  → 线程本地内存池分配；
//   其他哨兵值                            → malloc 回退分配。
// 必须用它释放：预处理的 malloc 回退指针交给内存池释放是未定义行为。
void releasePreprocessBuffer(image_buffer_t &buf);

class Detection {
public:
    Detection();
    ~Detection();

    // 核心检测函数
    bool runDetection(image_buffer_t *processedImage, letterbox_t *letter_box, object_detect_result_list &results, rknn_app_context_t *rknnAppCtx);

    // Pose推理函数
    bool runPoseDetection(image_buffer_t *processedImage, letterbox_t *letter_box, pose_detect_result_list &results, rknn_app_context_t *rknnAppCtx);

    // 预处理函数
    bool preprocess(const cv::Mat &image, image_buffer_t &processedImage, letterbox_t &letter_box, int model_width, int model_height);
    bool preprocess(const image_buffer_t &src_image, image_buffer_t &dst_image, letterbox_t &letter_box, int model_width, int model_height);

    // 设置函数
    void setModelContext(rknn_app_context_t *ctx);
    void setLogger(Logger *log);
    void setModelInitialized(bool initialized);

    // 性能指标函数
    PerformanceMetrics *getPerformanceMetrics();
    void printPerformanceStats();

    // 辅助函数
    cv::Scalar getDefectColor(int cls_id);

private:
    rknn_app_context_t *rknn_app_ctx;
    Logger *logger;
    PerformanceMetrics *performance;
    bool modelInitialized;
};

#endif // DETECTION_H
