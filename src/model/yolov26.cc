#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "yolov26.h"
#include "common.h"
#include "utils.h"
#include "core/rknn_model.h"

int init_yolov26_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 1 && app_ctx->output_attrs)
    {
        // 非 end2end 头：每尺度一个张量 [1, 4+nc, H, W]，4 通道为直接距离回归（reg_max=1）
        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
        int class_num = channels - 4;
        if (class_num <= 0)
        {
            class_num = get_obj_class_num();
        }

        app_ctx->class_num = class_num;
        printf("Model type: Detect (YOLO26)\n");
        printf("Model actual class number: %d\n", class_num);
        printf("Defined class number in code: %d\n", get_obj_class_num());

        if (class_num > get_obj_class_num())
        {
            printf("WARNING: Model output class number (%d) is larger than defined OBJ_CLASS_NUM (%d), may cause out of bounds access!\n",
                   class_num, get_obj_class_num());
        }
        else if (class_num < get_obj_class_num())
        {
            printf("INFO: Model output class number (%d) is smaller than defined OBJ_CLASS_NUM (%d)\n",
                   class_num, get_obj_class_num());
        }
    }
    else
    {
        printf("WARNING: Model has no output, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

int release_yolov26_model(rknn_app_context_t *app_ctx)
{
    return release_rknn_model(app_ctx);
}

int inference_yolov26_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, object_detect_result_list *od_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, od_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov26(ctx, outputs, lb, conf, nms, static_cast<object_detect_result_list *>(results));
                              });
}

int init_yolov26_pose_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    // 姿态模型单类（person），类数固定为 1
    app_ctx->class_num = 1;
    printf("Model type: Pose (YOLO26)\n");
    printf("Model actual class number: 1\n");
    printf("Defined class number in code: %d\n", get_obj_class_num());

    if (app_ctx->io_num.n_output >= 1 && app_ctx->output_attrs)
    {
        // 非 end2end pose 头：56 = 4(box) + 1(cls) + kpt*3，推导关键点数并告警
        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
        int kpt_num = (channels - 5) / 3;
        if (kpt_num > 0)
        {
            printf("Model pose keypoint number: %d\n", kpt_num);
            if (kpt_num != KEYPOINT_NUM)
            {
                printf("WARNING: Model keypoint number (%d) != code KEYPOINT_NUM (%d)\n",
                       kpt_num, KEYPOINT_NUM);
            }
        }
    }

    return ret;
}

int inference_yolov26_pose_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, pose_detect_result_list *pd_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, pd_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov26_pose(ctx, outputs, lb, conf, nms, static_cast<pose_detect_result_list *>(results));
                              });
}

int init_yolov26_obb_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 1 && app_ctx->output_attrs)
    {
        // 非 end2end OBB 头：channels = 4(box) + nc(cls) + 1(angle)，类数 = channels - 5
        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
        int class_num = channels - 5;
        if (class_num <= 0)
        {
            class_num = get_obj_class_num();
        }

        app_ctx->class_num = class_num;
        printf("Model type: OBB (YOLO26)\n");
        printf("Model actual class number: %d\n", class_num);
        printf("Defined class number in code: %d\n", get_obj_class_num());

        if (class_num > get_obj_class_num())
        {
            printf("WARNING: Model output class number (%d) is larger than defined OBJ_CLASS_NUM (%d), may cause out of bounds access!\n",
                   class_num, get_obj_class_num());
        }
        else if (class_num < get_obj_class_num())
        {
            printf("INFO: Model output class number (%d) is smaller than defined OBJ_CLASS_NUM (%d)\n",
                   class_num, get_obj_class_num());
        }
    }
    else
    {
        printf("WARNING: Model has no output, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

int inference_yolov26_obb_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, obb_detect_result_list *od_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, od_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov26_obb(ctx, outputs, lb, conf, nms, static_cast<obb_detect_result_list *>(results));
                              });
}

int init_yolov26_seg_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 2 && app_ctx->output_attrs)
    {
        // seg 头：3 尺度 [4 box + nc cls + nm mask]，最后输出是 proto [nm, H, W]
        const int proto_idx = app_ctx->io_num.n_output - 1;
        const rknn_tensor_attr &proto_attr = app_ctx->output_attrs[proto_idx];
        int mask_dim = (proto_attr.fmt == RKNN_TENSOR_NCHW) ? proto_attr.dims[1] : proto_attr.dims[3];

        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
        int class_num = channels - 4 - mask_dim;
        if (class_num <= 0)
        {
            class_num = get_obj_class_num();
        }

        app_ctx->class_num = class_num;
        printf("Model type: Seg (YOLO26)\n");
        printf("Model actual class number: %d, mask dim: %d\n", class_num, mask_dim);
        printf("Defined class number in code: %d\n", get_obj_class_num());

        if (class_num > get_obj_class_num())
        {
            printf("WARNING: Model output class number (%d) is larger than defined OBJ_CLASS_NUM (%d), may cause out of bounds access!\n",
                   class_num, get_obj_class_num());
        }
        else if (class_num < get_obj_class_num())
        {
            printf("INFO: Model output class number (%d) is smaller than defined OBJ_CLASS_NUM (%d)\n",
                   class_num, get_obj_class_num());
        }
    }
    else
    {
        printf("WARNING: Model has less than 2 outputs, cannot determine class number!\n");
        app_ctx->class_num = get_obj_class_num();
    }

    return ret;
}

int inference_yolov26_seg_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, seg_detect_result_list *seg_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, seg_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov26_seg(ctx, outputs, lb, conf, nms, static_cast<seg_detect_result_list *>(results));
                              });
}

int init_yolov26_depth_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    // 深度模型无类别，class_num 置 1 避免配置自检告警
    app_ctx->class_num = 1;
    printf("Model type: Depth (YOLO26)\n");
    if (app_ctx->io_num.n_output >= 1 && app_ctx->output_attrs)
    {
        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int h = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[2] : attr.dims[1];
        int w = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[3] : attr.dims[2];
        printf("Depth output: %dx%d\n", w, h);
    }
    return ret;
}

int inference_yolov26_depth_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, DepthTaskResult *depth_out, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, depth_out,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  DepthTaskResult *r = static_cast<DepthTaskResult *>(results);
                                  return post_process_yolov26_depth(ctx, outputs, lb, conf, nms, &r->depth, &r->depth_lo, &r->depth_hi);
                              });
}

