#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "yolov8.h"
#include "common.h"
#include "utils.h"
#include "core/rknn_model.h"

int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 3 && app_ctx->output_attrs)
    {
        int score_index = 1;
        int class_num = 0;

        if (app_ctx->output_attrs[score_index].fmt == RKNN_TENSOR_NCHW)
        {
            class_num = app_ctx->output_attrs[score_index].dims[1];
        }
        else
        {
            class_num = app_ctx->output_attrs[score_index].dims[3];
        }

        bool is_pose_model = (app_ctx->io_num.n_output == 4);

        if (is_pose_model)
        {
            app_ctx->class_num = 1;
            printf("Model type: Pose\n");
            printf("Model actual class number: 1\n");
        }
        else
        {
            app_ctx->class_num = class_num;
            printf("Model type: Detect\n");
            printf("Model actual class number: %d\n", class_num);
        }

        printf("Defined class number in code: %d\n", get_obj_class_num());

        int actual_class_num = app_ctx->class_num;
        if (actual_class_num > get_obj_class_num())
        {
            printf("WARNING: Model output class number (%d) is larger than defined OBJ_CLASS_NUM (%d), may cause out of bounds access!\n",
                   actual_class_num, get_obj_class_num());
        }
        else if (actual_class_num < get_obj_class_num())
        {
            printf("INFO: Model output class number (%d) is smaller than defined OBJ_CLASS_NUM (%d)\n",
                   actual_class_num, get_obj_class_num());
        }
    }
    else
    {
        printf("WARNING: Model has less than 3 outputs, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

int init_yolov8_obb_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 3 && app_ctx->output_attrs)
    {
        int feature_index = 0;
        int channel_num = 0;
        const int input_loc_len = 64;

        if (app_ctx->output_attrs[feature_index].fmt == RKNN_TENSOR_NCHW)
        {
            channel_num = app_ctx->output_attrs[feature_index].dims[1];
        }
        else
        {
            channel_num = app_ctx->output_attrs[feature_index].dims[3];
        }

        int class_num = channel_num - input_loc_len;
        if (class_num <= 0)
        {
            class_num = get_obj_class_num();
        }

        app_ctx->class_num = class_num;
        printf("Model type: OBB\n");
        printf("Model actual class number: %d\n", class_num);

        printf("Defined class number in code: %d\n", get_obj_class_num());

        int actual_class_num = app_ctx->class_num;
        if (actual_class_num > get_obj_class_num())
        {
            printf("WARNING: Model output class number (%d) is larger than defined OBJ_CLASS_NUM (%d), may cause out of bounds access!\n",
                   actual_class_num, get_obj_class_num());
        }
        else if (actual_class_num < get_obj_class_num())
        {
            printf("INFO: Model output class number (%d) is smaller than defined OBJ_CLASS_NUM (%d)\n",
                   actual_class_num, get_obj_class_num());
        }
    }
    else
    {
        printf("WARNING: Model has less than 3 outputs, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

int release_yolov8_model(rknn_app_context_t *app_ctx)
{
    return release_rknn_model(app_ctx);
}

int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, object_detect_result_list *od_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, od_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov8(ctx, outputs, lb, conf, nms, static_cast<object_detect_result_list *>(results));
                              });
}

int inference_yolov8_model_batch(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_imgs, letterbox_t *letter_boxes, object_detect_result_list *od_results, int batch_size, float conf_threshold, float nms_threshold)
{
    int ret;

    if ((!app_ctx) || !(preprocessed_imgs) || (!letter_boxes) || (!od_results))
    {
        return -1;
    }

    // Process each image individually (model may not support batch processing)
    for (int i = 0; i < batch_size; i++) {
        ret = inference_yolov8_model(app_ctx, &preprocessed_imgs[i], &letter_boxes[i], &od_results[i], conf_threshold, nms_threshold);
        if (ret < 0) {
            printf("inference_yolov8_model failed for image %d, ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}

int inference_yolov8_pose_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, pose_detect_result_list *pd_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, pd_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_pose(ctx, outputs, lb, conf, nms, static_cast<pose_detect_result_list *>(results));
                              });
}

int inference_yolov8_obb_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, obb_detect_result_list *od_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, od_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_obb(ctx, outputs, lb, conf, nms, static_cast<obb_detect_result_list *>(results));
                              });
}

int inference_yolov8_obb_model_batch(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_imgs, letterbox_t *letter_boxes, obb_detect_result_list *od_results, int batch_size, float conf_threshold, float nms_threshold)
{
    int ret;

    if ((!app_ctx) || !(preprocessed_imgs) || (!letter_boxes) || (!od_results))
    {
        return -1;
    }

    for (int i = 0; i < batch_size; i++) {
        ret = inference_yolov8_obb_model(app_ctx, &preprocessed_imgs[i], &letter_boxes[i], &od_results[i], conf_threshold, nms_threshold);
        if (ret < 0) {
            printf("inference_yolov8_obb_model failed for image %d, ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}

// 初始化YOLOv8分割模型
int init_yolov8_seg_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 3 && app_ctx->output_attrs)
    {
        // 分割模型的类别数通常与标签文件一致，优先使用配置的OBJ_CLASS_NUM
        app_ctx->class_num = get_obj_class_num();
        printf("Model type: Seg\n");
        printf("Model actual class number (by config): %d\n", app_ctx->class_num);
    }
    else
    {
        printf("WARNING: Model has less than 3 outputs, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

// 单张图YOLOv8分割推理
int inference_yolov8_seg_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, seg_detect_result_list *seg_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, seg_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_seg(ctx, outputs, lb, conf, nms, static_cast<seg_detect_result_list *>(results));
                              });
}

// 批量YOLOv8分割推理
int inference_yolov8_seg_model_batch(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_imgs, letterbox_t *letter_boxes, seg_detect_result_list *seg_results, int batch_size, float conf_threshold, float nms_threshold)
{
    int ret;

    if ((!app_ctx) || !(preprocessed_imgs) || (!letter_boxes) || (!seg_results))
    {
        return -1;
    }

    for (int i = 0; i < batch_size; i++)
    {
        ret = inference_yolov8_seg_model(app_ctx, &preprocessed_imgs[i], &letter_boxes[i], &seg_results[i], conf_threshold, nms_threshold);
        if (ret < 0)
        {
            printf("inference_yolov8_seg_model failed for image %d, ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}

int inference_yolov8_pose_model_batch(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_imgs, letterbox_t *letter_boxes, pose_detect_result_list *pd_results, int batch_size, float conf_threshold, float nms_threshold)
{
    int ret;

    if ((!app_ctx) || !(preprocessed_imgs) || (!letter_boxes) || (!pd_results))
    {
        return -1;
    }

    // Process each image individually (model may not support batch processing)
    for (int i = 0; i < batch_size; i++) {
        ret = inference_yolov8_pose_model(app_ctx, &preprocessed_imgs[i], &letter_boxes[i], &pd_results[i], conf_threshold, nms_threshold);
        if (ret < 0) {
            printf("inference_yolov8_pose_model failed for image %d, ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}
