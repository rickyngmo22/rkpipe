#ifndef _RKNN_DEMO_RTMPOSE_H_
#define _RKNN_DEMO_RTMPOSE_H_

#include "core/rknn_context.h"
#include "postprocess/postprocess.h"
#include "model/rtmpose_head_cpu.h"

// RTMPose（SimCC 头，top-down 两阶段的第二阶段）
//
// 模型输入：[1,3,256,192] RGB，归一化已编译进模型（rknn.config mean/std）
// 模型输出：simcc_x [1,17,W*2]、simcc_y [1,17,H*2]（logits，argmax 前不做 sigmoid/softmax）
// 坐标 = argmax / split_ratio(2.0)，为 256x192 裁剪图坐标，需经逆仿射映射回原图。

// 仿射系数：正向 m 把原图坐标映射到模型输入坐标，逆向 im 用于解码后映射回原图。
// 均为 [a, b, tx, c, d, ty]（行优先 2x3）。
typedef struct {
    float m[6];
    float im[6];
} rtmpose_affine_t;

// 单人裁剪推理结果（关键点已映射回原图坐标，conf 低于阈值置 0 便于绘制侧跳过）
typedef struct {
    pose_keypoint keypoints[KEYPOINT_NUM];
} rtmpose_person_result_t;

// SimCC 解码参数（Detector 组装，经 run_rknn_inference 的 results 指针透传给后处理）
typedef struct {
    const rtmpose_affine_t* affine;
    rtmpose_person_result_t* person_out;
    float kpt_threshold;
    int use_gpd;  // 1: score=sigmoid((vx+vy)/2)（GPD 训练模型）；0: score=sigmoid(vx)*sigmoid(vy)（mmpose 默认）
} rtmpose_decode_job_t;

// RTMPose（非 end2end，SimCC 头）：2 输出 simcc_x/simcc_y，单类 person
int init_rtmpose_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_rtmpose_model(rknn_app_context_t* app_ctx);

// 对单人仿射裁剪图（RGB888, 模型输入尺寸, virt_addr 连续）做推理并解码。
// 关键点坐标已按 affine->im 映射回原图。
int inference_rtmpose_model(rknn_app_context_t* app_ctx, image_buffer_t* rgb_crop,
                            const rtmpose_affine_t* affine, rtmpose_person_result_t* person_out,
                            float kpt_threshold, int use_gpd);

// ---- 拆分部署（feat + head 两个 RKNN，PC 端 split 产物）----
// feat：图像 → 中间特征 feat（[1,C,H,W]，NCHW，INT8）
// head：中间特征 feat → simcc_x/simcc_y（SimCC 双输出）
//
// 转换一致性要求（PC 端，务必满足，否则级联结果退化）：
//   1) feat 输出与 head 输入的中间特征必须共享同一量化域：
//      - 推荐：同一脚本内 rknn.split() 拆分（边界张量量化参数保持一致）；或
//      - 边界走 FP16（两段 do_quantization=False / 边界张量标 FP16），避免量化为题；
//   2) head 子图转换时禁止再带图像 mean/std 归一化配置（仅 feat 的图像输入需要）。
//   板端以 float 反量化 + pass_through=0 桥接，driver 按 head 自身 zp/scale 重新量化，
//   因此只要两段边界量化一致，板端即可无损级联。
//
// init：加载 head 模型并校验 SimCC 双输出（simcc_x bins=2*W，simcc_y bins=2*H）
int init_rtmpose_head_model(const char* model_path, rknn_app_context_t* app_ctx);

// 级联推理：feat 推理 → 特征转 head 输入布局（float 桥接）→ head 推理 → SimCC 解码。
// 关键点坐标已按 affine->im 映射回原图。
int inference_rtmpose_split_model(rknn_app_context_t* feat_ctx, rknn_app_context_t* head_ctx,
                                  image_buffer_t* rgb_crop, const rtmpose_affine_t* affine,
                                  rtmpose_person_result_t* person_out,
                                  float kpt_threshold, int use_gpd);

// CPU head 版级联推理：feat(RKNN) → head(CPU 复刻) → SimCC 解码。
// 用于 head 的 RKNN 转换损坏（MatMul/高斯注意力输出退化）时绕过 head RKNN。
// head_cpu 权重目录由 init_rtmpose_head_cpu 加载。
int inference_rtmpose_split_model_cpu(rknn_app_context_t* feat_ctx, rtmpose_head_cpu_t* head_cpu,
                                      image_buffer_t* rgb_crop, const rtmpose_affine_t* affine,
                                      rtmpose_person_result_t* person_out,
                                      float kpt_threshold, int use_gpd);

#endif //_RKNN_DEMO_RTMPOSE_H_
