#ifndef _RKNN_YOLOV8_POSTPROCESS_COMMON_H_
#define _RKNN_YOLOV8_POSTPROCESS_COMMON_H_

#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "postprocess/postprocess.h"

int clamp(float val, int min, int max);
float sigmoidf(float x);

// letterbox 逆映射的统一实现(此前散落在各任务后处理中各复制一份,约定靠人肉同步)。
// 框:坐标减 pad → 逐边钳到 [0, model_in] → 除 scale → 加裁剪偏移 → 钳回裁剪区(像素)。
// 入参为模型坐标系原始值(未减 pad)。
void map_box_to_frame(float x1, float y1, float x2, float y2,
                      const letterbox_t* letter_box, int model_in_w, int model_in_h,
                      image_rect_t* box);

// 点(关键点/中心点):减 pad → 除 scale → 钳位 → 加裁剪偏移(浮点)。
// content_clamp=true 钳到 letterbox 内容区(yolov8-pose 关键点行为,不落填充区);
// false 钳到 [0, model_in/scale](yolov26 系行为)。
void map_point_to_frame(float x, float y, const letterbox_t* letter_box,
                        int model_in_w, int model_in_h, bool content_clamp,
                        float* out_x, float* out_y);

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

// yolo26 系输出张量的实际类型:非量化模型 runtime 已转 fp32 直读;量化模型按 attr.type 分派
enum class Y26TensorType { kInt8, kFp16, kFp32 };
Y26TensorType y26_tensor_type_from_attr(bool is_quant, rknn_tensor_type type);

// yolo26 系输出张量元素读取:int8 返回原始量化值(反量化由调用方结合 zp/scale 完成),
// fp16 转 float,fp32 直读
float y26_tensor_at(const void* tensor, Y26TensorType type, int idx);

// 置信度阈值统一换算:cls 已 sigmoid → 概率域直传;logits → logit 域 ln(p/(1-p)),
// p<=0 → -1e9f,p>=1 → +1e9f
float y26_conf_to_logit_threshold(bool cls_sigmoided, float conf_threshold);

// yolo26 系模型：判定 cls 输出是否已在图上 sigmoid（方案2：cls 入图 sigmoid + INT8）。
// 扫描所有输出的 cls 区间 [cls_start, cls_start+class_count) 通道的反量化值域：
//   ⊂ [-1.0, 1.02] → 返回 1（已 sigmoid，后处理跳过二次 sigmoid）；否则返回 -1（logits）。
// class_count: 类别数；cls_start: 该任务 cls 通道起始（detect/seg/pose=4，obb=5 因 ch4 是 angle）。
int y26_scan_cls_sigmoided(rknn_app_context_t* app_ctx, const rknn_output* outputs,
                           int class_count, int cls_start);

#endif
