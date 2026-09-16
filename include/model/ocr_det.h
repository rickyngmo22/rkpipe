#pragma once

#include "core/rknn_context.h"
#include "core/task_result.h"

int init_ocr_det_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_ocr_det_model(rknn_app_context_t* app_ctx);
int inference_ocr_det_model(rknn_app_context_t* app_ctx,
                            image_buffer_t* preprocessed_img,
                            letterbox_t* letter_box,
                            OCRDetectTaskResult* ocr_results,
                            float conf_threshold,
                            float nms_threshold);
