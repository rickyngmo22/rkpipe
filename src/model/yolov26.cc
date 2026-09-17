#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "model/yolov26.h"
#include "utils/common.h"
#include "utils/utils.h"
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
        // 布局自适应：
        //   融合布局（旧）：每尺度一个张量 [1, 4+nc, H, W]，4 通道为直接距离回归（reg_max=1）
        //   拆分布局（新）：box [1, 4, H, W] 与 cls [1, nc, H, W] 成对，量化时各张量独立 scale
        // 判定：存在 channels==4 的输出张量即认为拆分布局（要求 nc != 4）
        auto y26_ch = [](const rknn_tensor_attr &a) {
            return (a.fmt == RKNN_TENSOR_NCHW) ? a.dims[1] : a.dims[3];
        };
        bool split_layout = false;
        for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i)
        {
            if (y26_ch(app_ctx->output_attrs[i]) == 4)
            {
                split_layout = true;
                break;
            }
        }

        int class_num = 0;
        if (split_layout)
        {
            for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i)
            {
                const int ch = y26_ch(app_ctx->output_attrs[i]);
                if (ch != 4)
                {
                    class_num = ch;
                    break;
                }
            }
            printf("YOLO26 detect: split box/cls layout, %u outputs\n", app_ctx->io_num.n_output);
        }
        else
        {
            class_num = y26_ch(app_ctx->output_attrs[0]) - 4;
        }

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
        // 布局自适应：
        //   融合布局：channels = 4(box) + nc(cls) + 1(angle)，类数 = channels - 5
        //   拆分布局：boxangle [1,5,H,W] 与 cls [1,nc,H,W] 成对，cls 独立 scale
        auto y26o_ch = [](const rknn_tensor_attr &a) {
            return (a.fmt == RKNN_TENSOR_NCHW) ? a.dims[1] : a.dims[3];
        };
        bool split_layout = false;
        for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i)
        {
            if (y26o_ch(app_ctx->output_attrs[i]) == 5)
            {
                split_layout = true;
                break;
            }
        }

        int class_num = 0;
        if (split_layout)
        {
            for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i)
            {
                const int ch = y26o_ch(app_ctx->output_attrs[i]);
                if (ch != 5)
                {
                    class_num = ch;
                    break;
                }
            }
            printf("YOLO26 OBB: split boxangle/cls layout, %u outputs\n", app_ctx->io_num.n_output);
        }
        else
        {
            class_num = y26o_ch(app_ctx->output_attrs[0]) - 5;
        }

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
        // seg 头两种布局（最后输出都是 proto [nm, H, W]）：
        //   融合:   3 尺度 [4 box + nc cls + nm mask] + proto，共 4 输出
        //   拆分:   3 尺度 × (box_i[4] + cls_i[nc] + mask_i[nm]) + proto，共 10 输出
        const int proto_idx = app_ctx->io_num.n_output - 1;
        const rknn_tensor_attr &proto_attr = app_ctx->output_attrs[proto_idx];
        int mask_dim = (proto_attr.fmt == RKNN_TENSOR_NCHW) ? proto_attr.dims[1] : proto_attr.dims[3];

        int class_num;
        if (app_ctx->io_num.n_output >= 7 && app_ctx->io_num.n_output % 3 == 1)
        {
            // 拆分布局：cls 张量独占输出，class_num 直接从 output_attrs[1] 通道数读取
            // （旧逻辑对 box 的 4 通道推出 0，只能回退 get_obj_class_num()，非 80 类模型会错）
            const rknn_tensor_attr &cls_attr = app_ctx->output_attrs[1];
            class_num = (cls_attr.fmt == RKNN_TENSOR_NCHW) ? cls_attr.dims[1] : cls_attr.dims[3];
            printf("Model layout: Seg split (box/cls/mask per scale)\n");
        }
        else
        {
            const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
            int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
            class_num = channels - 4 - mask_dim;
            printf("Model layout: Seg fused (box+cls+mask per scale)\n");
        }
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

int init_yolov26_sem_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0)
    {
        return ret;
    }

    if (app_ctx->io_num.n_output >= 1 && app_ctx->output_attrs)
    {
        // 语义分割头：单输出 [1, C, H, W]，C = 类别数（Cityscapes 19）
        const rknn_tensor_attr &attr = app_ctx->output_attrs[0];
        int channels = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[1] : attr.dims[3];
        int h = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[2] : attr.dims[1];
        int w = (attr.fmt == RKNN_TENSOR_NCHW) ? attr.dims[3] : attr.dims[2];
        if (channels <= 0) {
            channels = 19;
        }
        app_ctx->class_num = channels;
        printf("Model type: Sem (YOLO26)\n");
        printf("Model actual class number: %d, output: %dx%d\n", channels, w, h);
        printf("Defined class number in code: %d\n", get_obj_class_num());
    }
    else
    {
        printf("WARNING: Model has no output, cannot determine class number!\n");
        app_ctx->class_num = 19;
    }

    return ret;
}

int inference_yolov26_sem_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, cv::Mat *class_map_out, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, class_map_out,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov26_sem(ctx, outputs, lb, conf, nms, static_cast<cv::Mat *>(results));
                              });
}

