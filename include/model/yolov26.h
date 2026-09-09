#ifndef _RKNN_DEMO_YOLOV26_H_
#define _RKNN_DEMO_YOLOV26_H_

#include "core/rknn_context.h"
#include "core/task_result.h"
#include "postprocess/postprocess.h"

// YOLO26 detect（非 end2end，one2one 头）：3 尺度 [1, 4+nc, H, W]，box 直接距离 + cls logits
int init_yolov26_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolov26_model(rknn_app_context_t* app_ctx);
int inference_yolov26_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, object_detect_result_list* od_results, float conf_threshold, float nms_threshold);

// YOLO26 pose（非 end2end，one2one 头）：3 尺度 [1, 56, H, W]，56 = 4 box + 1 cls + 51 kpt
int init_yolov26_pose_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov26_pose_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, pose_detect_result_list* pd_results, float conf_threshold, float nms_threshold);

// YOLO26 OBB（非 end2end，one2one 头）：3 尺度 [1, 4+nc+1, H, W]，angle 原始弧度
int init_yolov26_obb_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov26_obb_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, obb_detect_result_list* od_results, float conf_threshold, float nms_threshold);

// YOLO26 seg（非 end2end，one2one 头）：3 尺度 [1, 4+nc+nm, H, W] + proto[1, nm, 160, 160]
int init_yolov26_seg_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov26_seg_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, seg_detect_result_list* seg_results, float conf_threshold, float nms_threshold);

// YOLO26 depth：单输出 [1,1,H,W]，depth=exp(head)，结果写入 DepthTaskResult*
//（depth=CV_8UC1 原帧分辨率 + roi + depth_lo/depth_hi 归一化范围）
int init_yolov26_depth_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov26_depth_model(rknn_app_context_t* app_ctx, image_buffer_t* preprocessed_img, letterbox_t* letter_box, DepthTaskResult* depth_out, float conf_threshold, float nms_threshold);



#endif //_RKNN_DEMO_YOLOV26_H_
