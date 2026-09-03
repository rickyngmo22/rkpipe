#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <set>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>

int post_process_yolov5(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    rknn_output *_outputs = (rknn_output *)outputs;
    static thread_local std::vector<float> filterBoxes;
    static thread_local std::vector<float> objProbs;
    static thread_local std::vector<int> classId;
    static thread_local bool initialized = false;

    if (!initialized) {
        filterBoxes.reserve(OBJ_NUMB_MAX_SIZE * 4);
        objProbs.reserve(OBJ_NUMB_MAX_SIZE);
        classId.reserve(OBJ_NUMB_MAX_SIZE);
        initialized = true;
    }

    filterBoxes.clear();
    objProbs.clear();
    classId.clear();

    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    if (!app_ctx || !_outputs || !app_ctx->output_attrs) {
        printf("post_process_yolov5: invalid parameters\n");
        return -1;
    }

    int class_count = app_ctx->class_num;
    if (class_count <= 0) {
        return -1;
    }

    const int anchors[3][6] = {
        {10, 13, 16, 30, 33, 23},
        {30, 61, 62, 45, 59, 119},
        {116, 90, 156, 198, 373, 326}
    };

    int validCount = 0;
    int branch_count = std::min(3, (int)app_ctx->io_num.n_output);
    for (int b = 0; b < branch_count; ++b) {
        int grid_h = 0;
        int grid_w = 0;
        int channels = 0;
        bool nchw = app_ctx->output_attrs[b].fmt == RKNN_TENSOR_NCHW;
        if (nchw) {
            channels = app_ctx->output_attrs[b].dims[1];
            grid_h = app_ctx->output_attrs[b].dims[2];
            grid_w = app_ctx->output_attrs[b].dims[3];
        } else {
            grid_h = app_ctx->output_attrs[b].dims[1];
            grid_w = app_ctx->output_attrs[b].dims[2];
            channels = app_ctx->output_attrs[b].dims[3];
        }

        if (grid_h <= 0 || grid_w <= 0 || channels <= 0) {
            continue;
        }

        int anchor_per_scale = 3;
        if (channels % anchor_per_scale != 0) {
            continue;
        }

        int output_class_num = channels / anchor_per_scale - 5;
        if (output_class_num <= 0) {
            continue;
        }
        int use_class_count = std::min(output_class_num, class_count);
        int stride = model_in_h / grid_h;

        const float *data_f32 = app_ctx->is_quant ? nullptr : (float *)_outputs[b].buf;
        const int8_t *data_i8 = app_ctx->is_quant ? (int8_t *)_outputs[b].buf : nullptr;
        int32_t zp = app_ctx->output_attrs[b].zp;
        float scale = app_ctx->output_attrs[b].scale;

        auto get_value = [&](int anchor, int y, int x, int c) -> float {
            int ch = anchor * (5 + output_class_num) + c;
            int idx = 0;
            if (nchw) {
                idx = ch * grid_h * grid_w + y * grid_w + x;
            } else {
                idx = (y * grid_w + x) * channels + ch;
            }
            if (app_ctx->is_quant) {
                return deqnt_affine_to_f32(data_i8[idx], zp, scale);
            }
            return data_f32[idx];
        };

        for (int i = 0; i < grid_h; ++i) {
            for (int j = 0; j < grid_w; ++j) {
                for (int a = 0; a < anchor_per_scale; ++a) {
                    float bx = get_value(a, i, j, 0);
                    float by = get_value(a, i, j, 1);
                    float bw = get_value(a, i, j, 2);
                    float bh = get_value(a, i, j, 3);
                    float obj = get_value(a, i, j, 4);

                    float obj_conf = obj;
                    if (obj_conf < conf_threshold) {
                        continue;
                    }

                    int best_id = -1;
                    float best_conf = 0.0f;
                    for (int c = 0; c < use_class_count; ++c) {
                        float cls = get_value(a, i, j, 5 + c);
                        float conf = obj_conf * cls;
                        if (conf > best_conf) {
                            best_conf = conf;
                            best_id = c;
                        }
                    }

                    if (best_conf < conf_threshold) {
                        continue;
                    }

                    float cx = (bx * 2.0f - 0.5f + j) * stride;
                    float cy = (by * 2.0f - 0.5f + i) * stride;

                    float aw = anchors[b][a * 2 + 0];
                    float ah = anchors[b][a * 2 + 1];

                    float pw = bw * 2.0f;
                    float ph = bh * 2.0f;
                    float w = pw * pw * aw;
                    float h = ph * ph * ah;

                    float x1 = cx - w / 2.0f;
                    float y1 = cy - h / 2.0f;

                    filterBoxes.push_back(x1);
                    filterBoxes.push_back(y1);
                    filterBoxes.push_back(w);
                    filterBoxes.push_back(h);
                    filterBoxes.push_back(0.0f);
                    objProbs.push_back(best_conf);
                    classId.push_back(best_id);
                    validCount++;
                }
            }
        }
    }

    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    if (validCount > OBJ_NUMB_MAX_SIZE) {
        validCount = OBJ_NUMB_MAX_SIZE;
        indexArray.resize(validCount);
    }

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold, 5);
    }

    int last_count = 0;
    od_results->count = 0;
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 5 + 2];
        float y2 = y1 + filterBoxes[n * 5 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        int crop_left = letter_box->crop_x;
        int crop_top = letter_box->crop_y;
        int crop_right = crop_left + std::max(1, letter_box->crop_w);
        int crop_bottom = crop_top + std::max(1, letter_box->crop_h);

        int left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale) + crop_left;
        int top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale) + crop_top;
        int right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale) + crop_left;
        int bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale) + crop_top;

        od_results->results[last_count].box.left = clamp(left, crop_left, crop_right);
        od_results->results[last_count].box.top = clamp(top, crop_top, crop_bottom);
        od_results->results[last_count].box.right = clamp(right, crop_left, crop_right);
        od_results->results[last_count].box.bottom = clamp(bottom, crop_top, crop_bottom);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}
