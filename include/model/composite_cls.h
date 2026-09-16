#pragma once

// M0 二级分类模型封装（MobileNetV2 等 ImageNet 分类头）：
//   输入：一级检测框 crop（BGR 任意尺寸），内部拉伸 resize 到模型输入尺寸
//   输出：[1, N]（raw logits 或已 softmax 自动判别）→ clsTop1
//   归一化已入图（转换脚本 mean/std），运行时喂 UINT8 BGR

#include <opencv2/core.hpp>

#include "core/rknn_context.h"
#include "postprocess/cls_top1.h"

int init_composite_cls_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_composite_cls_model(rknn_app_context_t* app_ctx);

// 对一张 crop 做分类，结果写入 out。返回 0 成功。
int inference_composite_cls_model(rknn_app_context_t* app_ctx,
                                  const cv::Mat& crop_bgr,
                                  ClsTop1* out);
