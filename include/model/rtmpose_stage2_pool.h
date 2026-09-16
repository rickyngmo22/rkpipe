#ifndef _RKNN_DEMO_RTMPOSE_STAGE2_POOL_H_
#define _RKNN_DEMO_RTMPOSE_STAGE2_POOL_H_

#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "core/rknn_context.h"
#include "model/rtmpose.h"

// RTMPose 二阶段全局工作池：把逐人 pose（feat NPU + CPU head）从 frame worker 线程剥离，
// 由专用池线程并行执行，实现一阶段(人体检测)与二阶段(pose)的解耦并行。
//
// 线程安全约束：rknn_context 非线程安全，每个池线程持有独立的 feat context 与
// CPU head 实例（权重各自加载），互不共享，可并发推理。
//
// 使用方式（批量）：
//   rtmpose_stage2_batch_t batch;
//   rtmpose_stage2_pool_batch_begin(pool, &batch, capacity);
//   for (...) rtmpose_stage2_pool_batch_add(&batch, crop, aff, &person_out_i);
//   rtmpose_stage2_pool_batch_run(&batch);   // 一次性提交并等待全部完成
//   rtmpose_stage2_pool_batch_end(&batch);
//
// 批内任务并行执行；crop 为引用计数共享（不复制像素），调用方须保证 batch_run 返回前
// crop/aff/person_out 均有效（本流程中即 runPersons 栈上对象，安全）。

typedef struct rtmpose_stage2_pool rtmpose_stage2_pool_t;

struct rtmpose_stage2_batch;

struct rtmpose_stage2_pool_job {
    cv::Mat crop;              // 引用计数共享，不复制像素
    rtmpose_affine_t aff;
    rtmpose_person_result_t* person_out = nullptr;
    rtmpose_stage2_batch* batch = nullptr;  // 完成时用于递减剩余计数
    int ret = -1;              // 推理返回码（batch_run 后可查）
};

struct rtmpose_stage2_batch {
    rtmpose_stage2_pool_t* pool = nullptr;
    std::vector<rtmpose_stage2_pool_job> jobs;  // 内部使用；batch_run 后按序查 ret
    std::atomic<int> remaining{0};
    std::mutex mtx;
    std::condition_variable cv;
};
typedef struct rtmpose_stage2_batch rtmpose_stage2_batch_t;

// 获取全局池（首次调用创建）。num_threads<=0 或加载失败返回 nullptr（调用方走内联路径）。
rtmpose_stage2_pool_t* rtmpose_stage2_pool_get(int num_threads, const std::string& feat_model_path,
                                               const std::string& head_weight_dir,
                                               int model_w, int model_h,
                                               float kpt_threshold, int use_gpd);

// 批量任务生命周期
int rtmpose_stage2_pool_batch_begin(rtmpose_stage2_pool_t* pool, rtmpose_stage2_batch* batch,
                                    int capacity);
int rtmpose_stage2_pool_batch_add(rtmpose_stage2_batch* batch, const cv::Mat& crop_rgb,
                                  const rtmpose_affine_t& aff, rtmpose_person_result_t* person_out);
// 一次性提交全部并阻塞到完成；返回 0 成功
int rtmpose_stage2_pool_batch_run(rtmpose_stage2_batch* batch);
void rtmpose_stage2_pool_batch_end(rtmpose_stage2_batch* batch);

#endif  // _RKNN_DEMO_RTMPOSE_STAGE2_POOL_H_
