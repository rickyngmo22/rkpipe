#pragma once

// RetinaFace（M8，RKNN Model Zoo RetinaFace_mobile320）模型封装：
//   输入 320×320（caffe BGR mean [104,117,123] 已入图，板端喂 UINT8 BGR）
//   输出 3 头：box(1,4200,4) / cls(1,4200,2) 图内已 Softmax / landm(1,4200,10)
//   letterbox 逆映射在本封装内完成（输出原图坐标）

#include "core/rknn_context.h"
#include "core/task_result.h"

int init_retinaface_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_retinaface_model(rknn_app_context_t* app_ctx);
int inference_retinaface_model(rknn_app_context_t* app_ctx,
                               image_buffer_t* preprocessed_img,
                               letterbox_t* letter_box,
                               FaceTaskResult* face_results,
                               float conf_threshold,
                               float nms_threshold);
