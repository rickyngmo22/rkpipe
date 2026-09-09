#ifndef _RKNN_DEMO_YOLOV5_H_
#define _RKNN_DEMO_YOLOV5_H_

#include "core/rknn_context.h"
#include "postprocess/postprocess.h"

int init_yolov5_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolov5_model(rknn_app_context_t* app_ctx);
int inference_yolov5_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, object_detect_result_list* od_results, float conf_threshold, float nms_threshold);
int inference_yolov5_model_batch(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_imgs, letterbox_t* letter_boxes, object_detect_result_list* od_results, int batch_size, float conf_threshold, float nms_threshold);

#endif
