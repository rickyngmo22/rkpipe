

#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

#include "core/rknn_context.h"



#include "postprocess/postprocess.h"


int init_yolov8_model(const char* model_path, rknn_app_context_t* app_ctx);

int init_yolov8_obb_model(const char* model_path, rknn_app_context_t* app_ctx);

int release_yolov8_model(rknn_app_context_t* app_ctx);

int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, object_detect_result_list* od_results, float conf_threshold, float nms_threshold);
int inference_yolov8_model_batch(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_imgs, letterbox_t* letter_boxes, object_detect_result_list* od_results, int batch_size, float conf_threshold, float nms_threshold);

// Pose推理函数
int inference_yolov8_pose_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, pose_detect_result_list* pd_results, float conf_threshold, float nms_threshold);
int inference_yolov8_pose_model_batch(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_imgs, letterbox_t* letter_boxes, pose_detect_result_list* pd_results, int batch_size, float conf_threshold, float nms_threshold);

int inference_yolov8_obb_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, obb_detect_result_list* od_results, float conf_threshold, float nms_threshold);
int inference_yolov8_obb_model_batch(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_imgs, letterbox_t* letter_boxes, obb_detect_result_list* od_results, int batch_size, float conf_threshold, float nms_threshold);

int init_yolov8_seg_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov8_seg_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, seg_detect_result_list* seg_results, float conf_threshold, float nms_threshold);
int inference_yolov8_seg_model_batch(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_imgs, letterbox_t* letter_boxes, seg_detect_result_list* seg_results, int batch_size, float conf_threshold, float nms_threshold);

#endif //_RKNN_DEMO_YOLOV8_H_
