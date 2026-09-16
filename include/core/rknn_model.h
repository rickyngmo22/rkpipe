#pragma once

#include "core/rknn_context.h"
#include "utils/utils.h"

typedef int (*RknnPostprocessFunc)(rknn_app_context_t*, void*, letterbox_t*, float, float, void*);

int init_rknn_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_rknn_model(rknn_app_context_t* app_ctx);
int run_rknn_inference(rknn_app_context_t* app_ctx,
                       image_buffer_t* img,
                       letterbox_t* letter_box,
                       float conf_threshold,
                       float nms_threshold,
                       void* results,
                       RknnPostprocessFunc postprocess);
