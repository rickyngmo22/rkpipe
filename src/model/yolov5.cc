#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov5.h"
#include "core/rknn_model.h"

int init_yolov5_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret = init_rknn_model(model_path, app_ctx);
    if (ret < 0) {
        return ret;
    }

    if (app_ctx->io_num.n_output <= 0 || !app_ctx->output_attrs) {
        return ret;
    }

    int channels = 0;
    if (app_ctx->output_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channels = app_ctx->output_attrs[0].dims[1];
    } else {
        channels = app_ctx->output_attrs[0].dims[3];
    }

    int anchor_per_scale = 3;
    int calc_class_num = channels / anchor_per_scale - 5;
    if (channels <= 0 || calc_class_num <= 0) {
        app_ctx->class_num = get_obj_class_num();
    } else {
        app_ctx->class_num = calc_class_num;
    }

    return ret;
}

int release_yolov5_model(rknn_app_context_t *app_ctx)
{
    return release_rknn_model(app_ctx);
}

int inference_yolov5_model(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_img, letterbox_t *letter_box, object_detect_result_list *od_results, float conf_threshold, float nms_threshold)
{
    return run_rknn_inference(app_ctx, preprocessed_img, letter_box, conf_threshold, nms_threshold, od_results,
                              [](rknn_app_context_t *ctx, void *outputs, letterbox_t *lb, float conf, float nms, void *results) {
                                  return post_process_yolov5(ctx, outputs, lb, conf, nms, static_cast<object_detect_result_list *>(results));
                              });
}

int inference_yolov5_model_batch(rknn_app_context_t *app_ctx, image_buffer_t *preprocessed_imgs, letterbox_t *letter_boxes, object_detect_result_list *od_results, int batch_size, float conf_threshold, float nms_threshold)
{
    int ret;

    if ((!app_ctx) || !(preprocessed_imgs) || (!letter_boxes) || (!od_results))
    {
        return -1;
    }

    for (int i = 0; i < batch_size; i++) {
        ret = inference_yolov5_model(app_ctx, &preprocessed_imgs[i], &letter_boxes[i], &od_results[i], conf_threshold, nms_threshold);
        if (ret < 0) {
            printf("inference_yolov5_model failed for image %d, ret=%d\n", i, ret);
            return ret;
        }
    }

    return 0;
}
