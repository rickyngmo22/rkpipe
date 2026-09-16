#pragma once

// PP-OCRv4 Rec（M6，RKNN Model Zoo ppocr_rec 对接）文字识别模型封装：
//   输入：文本行 BGR crop（透视矫正后的任意尺寸矩形），内部按模型宽高比
//         缩放（高对齐 model_height，保持宽高比，右侧补黑到 model_width）
//   输出：[1, T, C]（或 [T, C]）CTC logits → ctcGreedyDecode → 文本 + 分数
//   归一化 (x/255-0.5)/0.5 已在转换脚本经 mean/std_values 入图，运行时喂 UINT8。
//
// 纯 CPU 前后处理 + NPU 推理；字典与 blank 由调用方注入。

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/rknn_context.h"
#include "core/task_result.h"
#include "postprocess/ctc_decode.h"

int init_ocr_rec_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_ocr_rec_model(rknn_app_context_t* app_ctx);

// 对一张文本行 crop 做识别，结果写入 out（text/score）。返回 0 成功。
int inference_ocr_rec_model(rknn_app_context_t* app_ctx,
                            const cv::Mat& crop_bgr,
                            const std::vector<std::string>& dict,
                            int blank_index,
                            OCRTextLine* out);
