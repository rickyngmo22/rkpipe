#pragma once

// M13 动作识别（ST-GCN on pose）模型封装：
//   输入：PoseSequenceBuffer 组装的 [C=3, T, V] float 骨架序列（归一化已在组装侧完成，
//         不像图像模型走 mean/std 入图），运行时以 RKNN_TENSOR_FLOAT32 直接写入
//   输出：[1, num_actions]（raw logits 或已 softmax 自动判别，行和≈1 视为概率）
//         → clsTop1 概率模式 → 动作 id/score
// 模式复用 ocr_rec/composite_cls 惯例；动作标签文件每行一个动作名，索引对齐模型输出。
// 转换侧注意：图卷积 matmul 概率聚簇，INT8 易塌缩，默认 FP16（convert_stgcn.py）。

#include <cstdint>
#include <vector>

#include "core/rknn_context.h"
#include "postprocess/cls_top1.h"

int init_action_rec_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_action_rec_model(rknn_app_context_t* app_ctx);

// 对一个骨架窗口做动作推理。window 为 [C,T,V] float 行主序（PoseWindowTensor.data），
// dims 为 rknn 维度向量（{1,3,T,V}）。结果写入 out（cls_id/score，标签由调用方映射）。
// 返回 0 成功。
int inference_action_rec_model(rknn_app_context_t* app_ctx,
                               const float* window,
                               const std::vector<uint32_t>& dims,
                               ClsTop1* out);
