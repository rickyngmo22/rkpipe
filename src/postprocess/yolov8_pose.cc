#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>

int post_process_pose(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, pose_detect_result_list *pd_results)
{
    rknn_output *_outputs = (rknn_output *)outputs;
    static thread_local std::vector<float> filterBoxes;
    static thread_local std::vector<float> objProbs;
    static thread_local std::vector<int> classId;
    static thread_local bool initialized = false;

    if (!initialized) {
        filterBoxes.reserve(OBJ_NUMB_MAX_SIZE * 5);
        objProbs.reserve(OBJ_NUMB_MAX_SIZE);
        classId.reserve(OBJ_NUMB_MAX_SIZE);
        initialized = true;
    }

    filterBoxes.clear();
    objProbs.clear();
    classId.clear();
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(pd_results, 0, sizeof(pose_detect_result_list));

    if (!app_ctx || !_outputs || !app_ctx->output_attrs) {
        printf("post_process_pose: invalid parameters\n");
        return -1;
    }

    if (app_ctx->output_attrs[0].dims[1] == 0) {
        printf("post_process_pose: invalid output dims\n");
        return -1;
    }

    int output_dims = app_ctx->output_attrs[0].dims[1];
    int dfl_len = (output_dims - 1) / 4;

    bool has_keypoint_output = app_ctx->io_num.n_output >= 4;

    int keypoint_grid_len[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        int box_idx = i;
        if ((uint32_t)box_idx >= app_ctx->io_num.n_output) {
            keypoint_grid_len[i] = 0;
            continue;
        }
        int branch_grid_h = app_ctx->output_attrs[box_idx].dims[2];
        int branch_grid_w = app_ctx->output_attrs[box_idx].dims[3];
        keypoint_grid_len[i] = branch_grid_h * branch_grid_w;
    }

    int keypoints_index = 0;

    for (int i = 0; i < 3; i++)
    {
        int box_idx = i;

        if ((uint32_t)box_idx >= app_ctx->io_num.n_output) {
            printf("post_process_pose: box_idx out of bounds\n");
            continue;
        }

        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];

        if (grid_h <= 0 || grid_w <= 0) {
            printf("post_process_pose: invalid grid dimensions\n");
            continue;
        }

        stride = model_in_h / grid_h;
        int grid_len = grid_h * grid_w;

        if (app_ctx->is_quant)
        {
            int8_t *box_tensor = (int8_t *)_outputs[box_idx].buf;
            int32_t box_zp = app_ctx->output_attrs[box_idx].zp;
            float box_scale = app_ctx->output_attrs[box_idx].scale;

            int8_t *score_tensor = (int8_t *)_outputs[box_idx].buf;
            int32_t score_zp = app_ctx->output_attrs[box_idx].zp;
            float score_scale = app_ctx->output_attrs[box_idx].scale;

            for (int y = 0; y < grid_h; y++)
            {
                for (int x = 0; x < grid_w; x++)
                {
                    int offset = y * grid_w + x;

                    float conf = deqnt_affine_to_f32(score_tensor[offset + dfl_len * 4 * grid_len], score_zp, score_scale);
                    conf = 1.0f / (1.0f + expf(-conf));
                    if (conf < conf_threshold) {
                        continue;
                    }

                    float box[4];
                    float before_dfl[64];
                    for (int k = 0; k < dfl_len * 4; k++) {
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k * grid_len], box_zp, box_scale);
                    }
                    compute_dfl(before_dfl, dfl_len, box);

                    float x1 = (-box[0] + x + 0.5) * stride;
                    float y1 = (-box[1] + y + 0.5) * stride;
                    float x2 = (box[2] + x + 0.5) * stride;
                    float y2 = (box[3] + y + 0.5) * stride;
                    float w = x2 - x1;
                    float h = y2 - y1;

                    filterBoxes.push_back(x1);
                    filterBoxes.push_back(y1);
                    filterBoxes.push_back(w);
                    filterBoxes.push_back(h);
                    int current_keypoints_index = keypoints_index + offset;
                    filterBoxes.push_back(current_keypoints_index);
                    objProbs.push_back(conf);
                    classId.push_back(0);
                    validCount++;
                }
            }
        }

        keypoints_index += keypoint_grid_len[i];
    }

    if (validCount <= 0)
    {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    if (validCount > OBJ_NUMB_MAX_SIZE) {
        validCount = OBJ_NUMB_MAX_SIZE;
        indexArray.resize(validCount);
    }

    nms(validCount, filterBoxes, classId, indexArray, 0, nms_threshold, 5);

    int last_count = 0;
    pd_results->count = 0;

    int keypoint_output_idx = 3;

    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }

        int n = indexArray[i];

        const float x1 = filterBoxes[n * 5 + 0];
        const float y1 = filterBoxes[n * 5 + 1];
        const float x2 = x1 + filterBoxes[n * 5 + 2];
        const float y2 = y1 + filterBoxes[n * 5 + 3];
        float obj_conf = objProbs[i];
        int keypoints_index = (int)filterBoxes[n * 5 + 4];

        pose_detect_result* result = &pd_results->results[last_count];

        map_box_to_frame(x1, y1, x2, y2, letter_box, model_in_w, model_in_h, &result->box);
        result->box_conf = obj_conf;
        result->cls_id = 0;

        if (has_keypoint_output) {

            uint16_t *keypoint_data = (uint16_t *)_outputs[keypoint_output_idx].buf;

            for (int k = 0; k < KEYPOINT_NUM; k++) {
                int base_idx = k * 3 * 8400 + keypoints_index;

                uint16_t raw_x = keypoint_data[base_idx];
                uint16_t raw_y = keypoint_data[base_idx + 8400];
                uint16_t raw_conf = keypoint_data[base_idx + 2 * 8400];

                float kp_x = fp16_to_float(raw_x);
                float kp_y = fp16_to_float(raw_y);
                float kp_conf = fp16_to_float(raw_conf);

                map_point_to_frame(kp_x, kp_y, letter_box, model_in_w, model_in_h, true,
                                   &result->keypoints[k].x, &result->keypoints[k].y);
                result->keypoints[k].conf = kp_conf;
            }
        }

        last_count++;
    }

    pd_results->count = last_count;
    return 0;
}
