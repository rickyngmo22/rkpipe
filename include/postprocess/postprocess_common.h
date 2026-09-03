#ifndef _RKNN_YOLOV8_POSTPROCESS_COMMON_H_
#define _RKNN_YOLOV8_POSTPROCESS_COMMON_H_

#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "postprocess/postprocess.h"

int clamp(float val, int min, int max);
float sigmoidf(float x);

float fp16_to_float(uint16_t fp16_val);

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);
int nms(int validCount, std::vector<float> &outputLocations, const std::vector<int>& classIds, std::vector<int> &order,
        int filterId, float threshold, int box_stride);
int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);

int32_t __clip(float val, float min, float max);
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
void compute_dfl(float* tensor, int dfl_len, float* box);
void compute_dfl_contiguous_f32(const float* tensor, int dfl_len, float* box);
void compute_dfl_strided_f32_offset(const float* tensor, int grid_len, int dfl_len, int offset, float* box);
void compute_dfl_contiguous_i8(const int8_t* tensor, int dfl_len, int32_t zp, float scale, float* box);
void compute_dfl_strided_i8_offset(const int8_t* tensor, int grid_len, int dfl_len, int offset, int32_t zp, float scale, float* box);

// yolo26 系模型：判定 cls 输出是否已在图上 sigmoid（方案2：cls 入图 sigmoid + INT8）。
// 扫描所有输出的 cls 区间 [cls_start, cls_start+class_count) 通道的反量化值域：
//   ⊂ [-1.0, 1.02] → 返回 1（已 sigmoid，后处理跳过二次 sigmoid）；否则返回 -1（logits）。
// class_count: 类别数；cls_start: 该任务 cls 通道起始（detect/seg/pose=4，obb=5 因 ch4 是 angle）。
int y26_scan_cls_sigmoided(rknn_app_context_t* app_ctx, const rknn_output* outputs,
                           int class_count, int cls_start);

#endif
