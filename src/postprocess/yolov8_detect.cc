#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <set>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      int class_num, bool nhwc,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(score_sum_threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            if (nhwc) {
                int base = offset * class_num;
                for (int c = 0; c < class_num; c++) {
                    int8_t score = score_tensor[base + c];
                    if (score > score_thres_i8 && score > max_score) {
                        max_score = score;
                        max_class_id = c;
                    }
                }
            } else {
                for (int c= 0; c< class_num; c++){
                    if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                    {
                        max_score = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }
            }

            if (max_score> score_thres_i8){
                float box[4];
                float before_dfl[dfl_len*4];
                if (nhwc) {
                    int base = (i * grid_w + j) * dfl_len * 4;
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[base + k], box_zp, box_scale);
                    }
                } else {
                    offset = i* grid_w + j;
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        int class_num, bool nhwc,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                if (score_sum_tensor[offset] < score_sum_threshold){
                    continue;
                }
            }

            float max_score = 0;
            if (nhwc) {
                int base = offset * class_num;
                for (int c = 0; c < class_num; c++) {
                    float score = score_tensor[base + c];
                    if ((score > threshold) && (score > max_score)) {
                        max_score = score;
                        max_class_id = c;
                    }
                }
            } else {
                for (int c= 0; c< class_num; c++){
                    if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                    {
                        max_score = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }
            }

            // compute box
            if (max_score> threshold){
                float box[4];
                float before_dfl[dfl_len*4];
                if (nhwc) {
                    int base = (i * grid_w + j) * dfl_len * 4;
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = box_tensor[base + k];
                    }
                } else {
                    offset = i* grid_w + j;
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

int post_process_yolov8(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
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

    memset(od_results, 0, sizeof(object_detect_result_list));
    int output_per_branch = app_ctx->io_num.n_output / 3;
    int class_count = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;


        bool nhwc = app_ctx->output_attrs[box_idx].fmt != RKNN_TENSOR_NCHW;
        if (nhwc) {
            grid_h = app_ctx->output_attrs[box_idx].dims[1];
            grid_w = app_ctx->output_attrs[box_idx].dims[2];
        } else {
            grid_h = app_ctx->output_attrs[box_idx].dims[2];
            grid_w = app_ctx->output_attrs[box_idx].dims[3];
        }

        int box_channels = nhwc ? app_ctx->output_attrs[box_idx].dims[3] : app_ctx->output_attrs[box_idx].dims[1];
        int dfl_len = box_channels / 4;

        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                            (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                            (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                            grid_h, grid_w, stride, dfl_len, class_count, nhwc,
                            filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, class_count, nhwc,
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
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

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold, 4);
    }

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        const float x1 = filterBoxes[n * 4 + 0];
        const float y1 = filterBoxes[n * 4 + 1];
        const float x2 = x1 + filterBoxes[n * 4 + 2];
        const float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        map_box_to_frame(x1, y1, x2, y2, letter_box, model_in_w, model_in_h,
                         &od_results->results[last_count].box);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;

    return 0;
}
