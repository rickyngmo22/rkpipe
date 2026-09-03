#include <vector>
#include <set>
#include <cstring>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdlib>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <opencv2/opencv.hpp>
#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

static thread_local double g_last_seg_post_time_ms = 0.0;
static thread_local double g_last_seg_post_decode_time_ms = 0.0;
static thread_local double g_last_seg_post_sort_time_ms = 0.0;
static thread_local double g_last_seg_post_nms_time_ms = 0.0;
static thread_local double g_last_seg_post_mask_time_ms = 0.0;
static thread_local int g_last_seg_candidate_count = 0;
static thread_local int g_last_seg_class_count = 0;
static std::atomic<bool> g_seg_mask_enabled{true};

void set_seg_mask_enabled(bool enabled)
{
    g_seg_mask_enabled.store(enabled, std::memory_order_relaxed);
}

bool get_seg_mask_enabled()
{
    return g_seg_mask_enabled.load(std::memory_order_relaxed);
}

// 掩膜二值化阈值（logit 域，0.0 等价于 sigmoid>0.5）。
// RK_PIPE_SEG_MASK_THRESH 覆盖（默认 -0.5：略微放宽，容忍弱/量化掩膜，
// 避免部分正常目标的掩膜全零、绘制端只画框不画遮罩；设 0.0 恢复原行为）。
static float segMaskLogitThreshold() {
    const char* v = std::getenv("RK_PIPE_SEG_MASK_THRESH");
    if (!v || !*v) {
        return -0.5f;
    }
    return std::atof(v);
}

static inline int find_max_score_index_fp32(const float* scores, int class_num, float threshold, float& out_score)
{
    float max_score = threshold;
    int max_class_id = -1;

#if defined(__ARM_NEON)
    int c = 0;
    float32x4_t vmax = vdupq_n_f32(threshold);
    for (; c + 4 <= class_num; c += 4)
    {
        float32x4_t v = vld1q_f32(scores + c);
        vmax = vmaxq_f32(vmax, v);
    }
    float tmp0 = vgetq_lane_f32(vmax, 0);
    float tmp1 = vgetq_lane_f32(vmax, 1);
    float tmp2 = vgetq_lane_f32(vmax, 2);
    float tmp3 = vgetq_lane_f32(vmax, 3);
    max_score = tmp0;
    if (tmp1 > max_score) max_score = tmp1;
    if (tmp2 > max_score) max_score = tmp2;
    if (tmp3 > max_score) max_score = tmp3;
    for (; c < class_num; ++c)
    {
        float score = scores[c];
        if (score > max_score)
        {
            max_score = score;
        }
    }
#else
    for (int c = 0; c < class_num; ++c)
    {
        float score = scores[c];
        if (score > max_score)
        {
            max_score = score;
        }
    }
#endif

    if (max_score <= threshold)
    {
        out_score = max_score;
        return -1;
    }

    for (int c = 0; c < class_num; ++c)
    {
        float score = scores[c];
        if (score >= max_score && score > threshold)
        {
            max_class_id = c;
            break;
        }
    }

    out_score = max_score;
    return max_class_id;
}

static inline int find_max_score_index_i8(const int8_t* scores, int class_num, int8_t threshold, int8_t& out_score)
{
    int8_t max_score = threshold;
    int max_class_id = -1;

#if defined(__ARM_NEON)
    int c = 0;
    int8x16_t vmax = vdupq_n_s8(threshold);
    for (; c + 16 <= class_num; c += 16)
    {
        int8x16_t v = vld1q_s8(scores + c);
        vmax = vmaxq_s8(vmax, v);
    }
    int8_t tmp[16];
    vst1q_s8(tmp, vmax);
    for (int i = 0; i < 16; ++i)
    {
        if (tmp[i] > max_score)
        {
            max_score = tmp[i];
        }
    }
    for (; c < class_num; ++c)
    {
        int8_t score = scores[c];
        if (score > max_score)
        {
            max_score = score;
        }
    }
#else
    for (int c = 0; c < class_num; ++c)
    {
        int8_t score = scores[c];
        if (score > max_score)
        {
            max_score = score;
        }
    }
#endif

    if (max_score <= threshold)
    {
        out_score = max_score;
        return -1;
    }

    for (int c = 0; c < class_num; ++c)
    {
        int8_t score = scores[c];
        if (score >= max_score && score > threshold)
        {
            max_class_id = c;
            break;
        }
    }

    out_score = max_score;
    return max_class_id;
}

static inline int find_max_score_index_fp32_nchw(const float* const* class_ptrs, int class_num, int offset, float threshold, float& out_score)
{
    float max_score = threshold;
    int max_class_id = -1;
    for (int c = 0; c < class_num; ++c)
    {
        float score = class_ptrs[c][offset];
        if (score > max_score)
        {
            max_score = score;
            max_class_id = c;
        }
    }
    out_score = max_score;
    return max_class_id;
}

static inline int find_max_score_index_i8_nchw(const int8_t* const* class_ptrs, int class_num, int offset, int8_t threshold, int8_t& out_score)
{
    int8_t max_score = threshold;
    int max_class_id = -1;
    for (int c = 0; c < class_num; ++c)
    {
        int8_t score = class_ptrs[c][offset];
        if (score > max_score)
        {
            max_score = score;
            max_class_id = c;
        }
    }
    out_score = max_score;
    return max_class_id;
}

static int process_fp32_seg(float* box_tensor,
                            float* score_tensor,
                            float* score_sum_tensor,
                            float* mask_tensor,
                            int grid_h,
                            int grid_w,
                            int stride,
                            int dfl_len,
                            int class_num,
                            int mask_dim,
                            bool nhwc,
                            bool keep_mask,
                            std::vector<float>& boxes,
                            std::vector<float>& scores,
                            std::vector<int>& classIds,
                            std::vector<float>& maskCoeffs,
                            float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    static thread_local std::vector<const float*> class_ptrs;
    static thread_local std::vector<float> score_buf;
    static thread_local std::vector<float> score_block;
    bool use_score_buf = !nhwc && class_num <= 128;
    bool use_score_block = !nhwc && !use_score_buf;
    if (!nhwc)
    {
        if (use_score_buf)
        {
            score_buf.resize(class_num);
        }
        else if (use_score_block)
        {
            score_block.resize(static_cast<size_t>(class_num) * 8);
        }
        else
        {
            class_ptrs.resize(class_num);
            for (int c = 0; c < class_num; ++c)
            {
                class_ptrs[c] = score_tensor + c * grid_len;
            }
        }
    }

    if (nhwc)
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                float max_score = 0.f;

                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    if (score_sum_tensor[offset] < score_sum_threshold)
                    {
                        continue;
                    }
                }

                int base = offset * (4 * dfl_len);
                int cls_base = offset * class_num;
                max_class_id = find_max_score_index_fp32(score_tensor + cls_base, class_num, threshold, max_score);

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_contiguous_f32(box_tensor + base, dfl_len, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(max_score);
                classIds.push_back(max_class_id);

                int mask_base = offset * mask_dim;
                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(mask_tensor[mask_base + m]);
                    }
                }

                validCount++;
            }
        }
    }
    else if (use_score_block)
    {
        const int block = 8;
        std::vector<uint8_t> pass_mask(block);
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; j += block)
            {
                int blk = std::min(block, grid_w - j);
                int base_offset = i * grid_w + j;
                int pass_count = 0;
                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    for (int k = 0; k < blk; ++k)
                    {
                        int offset = base_offset + k;
                        uint8_t pass = score_sum_tensor[offset] >= score_sum_threshold;
                        pass_mask[k] = pass;
                        pass_count += pass;
                    }
                    if (pass_count == 0)
                    {
                        continue;
                    }
                }

                for (int c = 0; c < class_num; ++c)
                {
                    const float* src = score_tensor + c * grid_len + base_offset;
                    for (int k = 0; k < blk; ++k)
                    {
                        score_block[k * class_num + c] = src[k];
                    }
                }

                for (int k = 0; k < blk; ++k)
                {
                    if (score_sum_tensor != nullptr && pass_mask[k] == 0)
                    {
                        continue;
                    }

                    int offset = base_offset + k;
                    int max_class_id = -1;
                    float max_score = 0.f;
                    max_class_id = find_max_score_index_fp32(score_block.data() + k * class_num, class_num, threshold, max_score);
                    if (max_class_id < 0)
                    {
                        continue;
                    }

                    float box[4];
                    compute_dfl_strided_f32_offset(box_tensor, grid_len, dfl_len, offset, box);

                    float x1 = (-box[0] + j + k + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + k + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;
                    float w = x2 - x1;
                    float h = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    scores.push_back(max_score);
                    classIds.push_back(max_class_id);

                    if (keep_mask)
                    {
                        for (int m = 0; m < mask_dim; ++m)
                        {
                            maskCoeffs.push_back(mask_tensor[m * grid_len + offset]);
                        }
                    }

                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                float max_score = 0.f;

                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    if (score_sum_tensor[offset] < score_sum_threshold)
                    {
                        continue;
                    }
                }

                if (use_score_buf)
                {
                    for (int c = 0; c < class_num; ++c)
                    {
                        score_buf[c] = score_tensor[c * grid_len + offset];
                    }
                    max_class_id = find_max_score_index_fp32(score_buf.data(), class_num, threshold, max_score);
                }
                else
                {
                    max_class_id = find_max_score_index_fp32_nchw(class_ptrs.data(), class_num, offset, threshold, max_score);
                }

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_strided_f32_offset(box_tensor, grid_len, dfl_len, offset, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(max_score);
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(mask_tensor[m * grid_len + offset]);
                    }
                }

                validCount++;
            }
        }
    }

    return validCount;
}

static int process_fp32_seg_combined(float* tensor,
                                     float* score_sum_tensor,
                                     int grid_h,
                                     int grid_w,
                                     int stride,
                                     int dfl_len,
                                     int class_num,
                                     int mask_dim,
                                     bool nhwc,
                                     bool keep_mask,
                                     std::vector<float>& boxes,
                                     std::vector<float>& scores,
                                     std::vector<int>& classIds,
                                     std::vector<float>& maskCoeffs,
                                     float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int channels = 4 * dfl_len + class_num + mask_dim;
    static thread_local std::vector<const float*> class_ptrs;
    static thread_local std::vector<float> score_buf;
    static thread_local std::vector<float> score_block;
    bool use_score_buf = !nhwc && class_num <= 128;
    bool use_score_block = !nhwc && !use_score_buf;
    if (!nhwc)
    {
        if (use_score_buf)
        {
            score_buf.resize(class_num);
        }
        else if (use_score_block)
        {
            score_block.resize(static_cast<size_t>(class_num) * 8);
        }
        else
        {
            class_ptrs.resize(class_num);
            for (int c = 0; c < class_num; ++c)
            {
                class_ptrs[c] = tensor + (4 * dfl_len + c) * grid_len;
            }
        }
    }

    if (nhwc)
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                float max_score = 0.f;

                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    if (score_sum_tensor[offset] < score_sum_threshold)
                    {
                        continue;
                    }
                }

                int base = offset * channels;
                int box_base = base;
                int cls_base = base + 4 * dfl_len;
                int mask_base = cls_base + class_num;
                max_class_id = find_max_score_index_fp32(tensor + cls_base, class_num, threshold, max_score);

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_contiguous_f32(tensor + box_base, dfl_len, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(max_score);
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(tensor[mask_base + m]);
                    }
                }

                validCount++;
            }
        }
    }
    else if (use_score_block)
    {
        const int block = 8;
        std::vector<uint8_t> pass_mask(block);
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; j += block)
            {
                int blk = std::min(block, grid_w - j);
                int base_offset = i * grid_w + j;
                int pass_count = 0;
                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    for (int k = 0; k < blk; ++k)
                    {
                        int offset = base_offset + k;
                        uint8_t pass = score_sum_tensor[offset] >= score_sum_threshold;
                        pass_mask[k] = pass;
                        pass_count += pass;
                    }
                    if (pass_count == 0)
                    {
                        continue;
                    }
                }

                for (int c = 0; c < class_num; ++c)
                {
                    const float* src = tensor + (4 * dfl_len + c) * grid_len + base_offset;
                    for (int k = 0; k < blk; ++k)
                    {
                        score_block[k * class_num + c] = src[k];
                    }
                }

                for (int k = 0; k < blk; ++k)
                {
                    if (score_sum_tensor != nullptr && pass_mask[k] == 0)
                    {
                        continue;
                    }

                    int offset = base_offset + k;
                    int max_class_id = -1;
                    float max_score = 0.f;
                    max_class_id = find_max_score_index_fp32(score_block.data() + k * class_num, class_num, threshold, max_score);
                    if (max_class_id < 0)
                    {
                        continue;
                    }

                    float box[4];
                    compute_dfl_strided_f32_offset(tensor, grid_len, dfl_len, offset, box);

                    float x1 = (-box[0] + j + k + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + k + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;
                    float w = x2 - x1;
                    float h = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    scores.push_back(max_score);
                    classIds.push_back(max_class_id);

                    if (keep_mask)
                    {
                        for (int m = 0; m < mask_dim; ++m)
                        {
                            maskCoeffs.push_back(tensor[(4 * dfl_len + class_num + m) * grid_len + offset]);
                        }
                    }

                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                float max_score = 0.f;

                if (score_sum_tensor != nullptr)
                {
                    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
                    if (score_sum_tensor[offset] < score_sum_threshold)
                    {
                        continue;
                    }
                }

                if (use_score_buf)
                {
                    for (int c = 0; c < class_num; ++c)
                    {
                        score_buf[c] = tensor[(4 * dfl_len + c) * grid_len + offset];
                    }
                    max_class_id = find_max_score_index_fp32(score_buf.data(), class_num, threshold, max_score);
                }
                else
                {
                    max_class_id = find_max_score_index_fp32_nchw(class_ptrs.data(), class_num, offset, threshold, max_score);
                }

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_strided_f32_offset(tensor, grid_len, dfl_len, offset, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(max_score);
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(tensor[(4 * dfl_len + class_num + m) * grid_len + offset]);
                    }
                }

                validCount++;
            }
        }
    }

    return validCount;
}

static int process_i8_seg(int8_t* box_tensor,
                          int32_t box_zp,
                          float box_scale,
                          int8_t* score_tensor,
                          int32_t score_zp,
                          float score_scale,
                          int8_t* score_sum_tensor,
                          int32_t score_sum_zp,
                          float score_sum_scale,
                          int8_t* mask_tensor,
                          int32_t mask_zp,
                          float mask_scale,
                          int grid_h,
                          int grid_w,
                          int stride,
                          int dfl_len,
                          int class_num,
                          int mask_dim,
                          bool nhwc,
                          bool keep_mask,
                          std::vector<float>& boxes,
                          std::vector<float>& scores,
                          std::vector<int>& classIds,
                          std::vector<float>& maskCoeffs,
                          float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(score_sum_threshold, score_sum_zp, score_sum_scale);

    static thread_local std::vector<const int8_t*> class_ptrs;
    static thread_local std::vector<int8_t> score_buf;
    static thread_local std::vector<int8_t> score_block;
    bool use_score_buf = !nhwc && class_num <= 128;
    bool use_score_block = !nhwc && !use_score_buf;
    if (!nhwc)
    {
        if (use_score_buf)
        {
            score_buf.resize(class_num);
        }
        else if (use_score_block)
        {
            score_block.resize(static_cast<size_t>(class_num) * 8);
        }
        else
        {
            class_ptrs.resize(class_num);
            for (int c = 0; c < class_num; ++c)
            {
                class_ptrs[c] = score_tensor + c * grid_len;
            }
        }
    }

    if (nhwc)
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                int8_t max_score = -score_zp;

                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                    {
                        continue;
                    }
                }

                int base = offset * (4 * dfl_len);
                int cls_base = offset * class_num;
                max_class_id = find_max_score_index_i8(score_tensor + cls_base, class_num, score_thres_i8, max_score);

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_contiguous_i8(box_tensor + base, dfl_len, box_zp, box_scale, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classIds.push_back(max_class_id);

                int mask_base = offset * mask_dim;
                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(deqnt_affine_to_f32(mask_tensor[mask_base + m], mask_zp, mask_scale));
                    }
                }

                validCount++;
            }
        }
    }
    else if (use_score_block)
    {
        const int block = 8;
        std::vector<uint8_t> pass_mask(block);
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; j += block)
            {
                int blk = std::min(block, grid_w - j);
                int base_offset = i * grid_w + j;
                int pass_count = 0;
                if (score_sum_tensor != nullptr)
                {
                    for (int k = 0; k < blk; ++k)
                    {
                        int offset = base_offset + k;
                        uint8_t pass = score_sum_tensor[offset] >= score_sum_thres_i8;
                        pass_mask[k] = pass;
                        pass_count += pass;
                    }
                    if (pass_count == 0)
                    {
                        continue;
                    }
                }

                for (int c = 0; c < class_num; ++c)
                {
                    const int8_t* src = score_tensor + c * grid_len + base_offset;
                    for (int k = 0; k < blk; ++k)
                    {
                        score_block[k * class_num + c] = src[k];
                    }
                }

                for (int k = 0; k < blk; ++k)
                {
                    if (score_sum_tensor != nullptr && pass_mask[k] == 0)
                    {
                        continue;
                    }

                    int offset = base_offset + k;
                    int max_class_id = -1;
                    int8_t max_score = -score_zp;
                    max_class_id = find_max_score_index_i8(score_block.data() + k * class_num, class_num, score_thres_i8, max_score);
                    if (max_class_id < 0)
                    {
                        continue;
                    }

                    float box[4];
                    compute_dfl_strided_i8_offset(box_tensor, grid_len, dfl_len, offset, box_zp, box_scale, box);

                    float x1 = (-box[0] + j + k + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + k + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;
                    float w = x2 - x1;
                    float h = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    scores.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classIds.push_back(max_class_id);

                    if (keep_mask)
                    {
                        for (int m = 0; m < mask_dim; ++m)
                        {
                            maskCoeffs.push_back(deqnt_affine_to_f32(mask_tensor[m * grid_len + offset], mask_zp, mask_scale));
                        }
                    }

                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                int8_t max_score = -score_zp;

                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                    {
                        continue;
                    }
                }

                if (use_score_buf)
                {
                    for (int c = 0; c < class_num; ++c)
                    {
                        score_buf[c] = score_tensor[c * grid_len + offset];
                    }
                    max_class_id = find_max_score_index_i8(score_buf.data(), class_num, score_thres_i8, max_score);
                }
                else
                {
                    max_class_id = find_max_score_index_i8_nchw(class_ptrs.data(), class_num, offset, score_thres_i8, max_score);
                }

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_strided_i8_offset(box_tensor, grid_len, dfl_len, offset, box_zp, box_scale, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(deqnt_affine_to_f32(mask_tensor[m * grid_len + offset], mask_zp, mask_scale));
                    }
                }

                validCount++;
            }
        }
    }

    return validCount;
}

static int process_i8_seg_combined(int8_t* tensor,
                                   int32_t tensor_zp,
                                   float tensor_scale,
                                   int8_t* score_sum_tensor,
                                   int32_t score_sum_zp,
                                   float score_sum_scale,
                                   int grid_h,
                                   int grid_w,
                                   int stride,
                                   int dfl_len,
                                   int class_num,
                                   int mask_dim,
                                   bool nhwc,
                                   bool keep_mask,
                                   std::vector<float>& boxes,
                                   std::vector<float>& scores,
                                   std::vector<int>& classIds,
                                   std::vector<float>& maskCoeffs,
                                   float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int channels = 4 * dfl_len + class_num + mask_dim;
    float score_sum_threshold = threshold < BOX_THRESH ? threshold : BOX_THRESH;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, tensor_zp, tensor_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(score_sum_threshold, score_sum_zp, score_sum_scale);

    static thread_local std::vector<const int8_t*> class_ptrs;
    static thread_local std::vector<int8_t> score_buf;
    static thread_local std::vector<int8_t> score_block;
    bool use_score_buf = !nhwc && class_num <= 128;
    bool use_score_block = !nhwc && !use_score_buf;
    if (!nhwc)
    {
        if (use_score_buf)
        {
            score_buf.resize(class_num);
        }
        else if (use_score_block)
        {
            score_block.resize(static_cast<size_t>(class_num) * 8);
        }
        else
        {
            class_ptrs.resize(class_num);
            for (int c = 0; c < class_num; ++c)
            {
                class_ptrs[c] = tensor + (4 * dfl_len + c) * grid_len;
            }
        }
    }

    if (nhwc)
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                int8_t max_score = -tensor_zp;

                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                    {
                        continue;
                    }
                }

                int base = offset * channels;
                int box_base = base;
                int cls_base = base + 4 * dfl_len;
                int mask_base = cls_base + class_num;
                max_class_id = find_max_score_index_i8(tensor + cls_base, class_num, score_thres_i8, max_score);

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_contiguous_i8(tensor + box_base, dfl_len, tensor_zp, tensor_scale, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(deqnt_affine_to_f32(max_score, tensor_zp, tensor_scale));
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(deqnt_affine_to_f32(tensor[mask_base + m], tensor_zp, tensor_scale));
                    }
                }

                validCount++;
            }
        }
    }
    else if (use_score_block)
    {
        const int block = 8;
        std::vector<uint8_t> pass_mask(block);
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; j += block)
            {
                int blk = std::min(block, grid_w - j);
                int base_offset = i * grid_w + j;
                int pass_count = 0;
                if (score_sum_tensor != nullptr)
                {
                    for (int k = 0; k < blk; ++k)
                    {
                        int offset = base_offset + k;
                        uint8_t pass = score_sum_tensor[offset] >= score_sum_thres_i8;
                        pass_mask[k] = pass;
                        pass_count += pass;
                    }
                    if (pass_count == 0)
                    {
                        continue;
                    }
                }

                for (int c = 0; c < class_num; ++c)
                {
                    const int8_t* src = tensor + (4 * dfl_len + c) * grid_len + base_offset;
                    for (int k = 0; k < blk; ++k)
                    {
                        score_block[k * class_num + c] = src[k];
                    }
                }

                for (int k = 0; k < blk; ++k)
                {
                    if (score_sum_tensor != nullptr && pass_mask[k] == 0)
                    {
                        continue;
                    }

                    int offset = base_offset + k;
                    int max_class_id = -1;
                    int8_t max_score = -tensor_zp;
                    max_class_id = find_max_score_index_i8(score_block.data() + k * class_num, class_num, score_thres_i8, max_score);
                    if (max_class_id < 0)
                    {
                        continue;
                    }

                    float box[4];
                    compute_dfl_strided_i8_offset(tensor, grid_len, dfl_len, offset, tensor_zp, tensor_scale, box);

                    float x1 = (-box[0] + j + k + 0.5f) * stride;
                    float y1 = (-box[1] + i + 0.5f) * stride;
                    float x2 = (box[2] + j + k + 0.5f) * stride;
                    float y2 = (box[3] + i + 0.5f) * stride;
                    float w = x2 - x1;
                    float h = y2 - y1;

                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    scores.push_back(deqnt_affine_to_f32(max_score, tensor_zp, tensor_scale));
                    classIds.push_back(max_class_id);

                    if (keep_mask)
                    {
                        for (int m = 0; m < mask_dim; ++m)
                        {
                            maskCoeffs.push_back(deqnt_affine_to_f32(tensor[(4 * dfl_len + class_num + m) * grid_len + offset], tensor_zp, tensor_scale));
                        }
                    }

                    validCount++;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_h; ++i)
        {
            for (int j = 0; j < grid_w; ++j)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
                int8_t max_score = -tensor_zp;

                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_i8)
                    {
                        continue;
                    }
                }

                if (use_score_buf)
                {
                    for (int c = 0; c < class_num; ++c)
                    {
                        score_buf[c] = tensor[(4 * dfl_len + c) * grid_len + offset];
                    }
                    max_class_id = find_max_score_index_i8(score_buf.data(), class_num, score_thres_i8, max_score);
                }
                else
                {
                    max_class_id = find_max_score_index_i8_nchw(class_ptrs.data(), class_num, offset, score_thres_i8, max_score);
                }

                if (max_class_id < 0)
                {
                    continue;
                }

                float box[4];
                compute_dfl_strided_i8_offset(tensor, grid_len, dfl_len, offset, tensor_zp, tensor_scale, box);

                float x1 = (-box[0] + j + 0.5f) * stride;
                float y1 = (-box[1] + i + 0.5f) * stride;
                float x2 = (box[2] + j + 0.5f) * stride;
                float y2 = (box[3] + i + 0.5f) * stride;
                float w = x2 - x1;
                float h = y2 - y1;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                scores.push_back(deqnt_affine_to_f32(max_score, tensor_zp, tensor_scale));
                classIds.push_back(max_class_id);

                if (keep_mask)
                {
                    for (int m = 0; m < mask_dim; ++m)
                    {
                        maskCoeffs.push_back(deqnt_affine_to_f32(tensor[(4 * dfl_len + class_num + m) * grid_len + offset], tensor_zp, tensor_scale));
                    }
                }

                validCount++;
            }
        }
    }

    return validCount;
}

#if defined(__ARM_NEON)
// 水平归约 float32x4（ARMv8 用 vaddvq_f32，32 位回退到标量求和）
static inline float horizontalSumF32(float32x4_t v)
{
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float buf[4];
    vst1q_f32(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3];
#endif
}

// 连续 int8 -> float32 反量化：按 16 元素向量化（先宽展到 int16 再减 zp，
// 避免 int8 相减溢出），尾部标量兜底。src/dst 均需连续。
static inline void dequantI8ToF32Neon(const int8_t* src, int32_t zp, float scale,
                                      float* dst, int count)
{
    const int16x8_t vzp = vdupq_n_s16(static_cast<int16_t>(zp));
    const float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 16 <= count; i += 16)
    {
        const int8x16_t v = vld1q_s8(src + i);
        const int16x8_t lo = vsubq_s16(vmovl_s8(vget_low_s8(v)), vzp);
        const int16x8_t hi = vsubq_s16(vmovl_s8(vget_high_s8(v)), vzp);
        vst1q_f32(dst + i, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vscale));
        vst1q_f32(dst + i + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vscale));
        vst1q_f32(dst + i + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vscale));
        vst1q_f32(dst + i + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vscale));
    }
    for (; i < count; ++i)
    {
        dst[i] = deqnt_affine_to_f32(src[i], zp, scale);
    }
}
#endif

static bool build_proto_tensor(rknn_app_context_t* app_ctx,
                               rknn_output* outputs,
                               int proto_index,
                               cv::Mat& proto,
                               int& mask_dim,
                               int& mask_h,
                               int& mask_w)
{
    if (proto_index < 0 || !app_ctx || !outputs || !app_ctx->output_attrs)
    {
        return false;
    }

    const auto& attr = app_ctx->output_attrs[proto_index];
    bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
    if (nhwc)
    {
        mask_h = attr.dims[1];
        mask_w = attr.dims[2];
        mask_dim = attr.dims[3];
    }
    else
    {
        mask_dim = attr.dims[1];
        mask_h = attr.dims[2];
        mask_w = attr.dims[3];
    }

    if (mask_dim <= 0 || mask_h <= 0 || mask_w <= 0)
    {
        return false;
    }

    // 转置布局：proto 为 (mask_h*mask_w) x mask_dim，每行是一个像素的掩膜系数。
    // NHWC 反量化写入连续（可直接 NEON 向量化），后续 mask = proto * coeff^T
    // 变成逐像素行点积（缓存友好 + NEON）。
    proto.create(mask_h * mask_w, mask_dim, CV_32F);
    const int plane_size = mask_h * mask_w;

    if (app_ctx->is_quant)
    {
        int8_t* buf = reinterpret_cast<int8_t*>(outputs[proto_index].buf);
        int32_t zp = attr.zp;
        float scale = attr.scale;
        if (nhwc)
        {
            // 每像素 mask_dim 个通道连续：逐像素反量化，写入 proto 一行（连续）
            for (int hw = 0; hw < plane_size; ++hw)
            {
#if defined(__ARM_NEON)
                dequantI8ToF32Neon(buf + hw * mask_dim, zp, scale, proto.ptr<float>(hw), mask_dim);
#else
                float* dst = proto.ptr<float>(hw);
                const int8_t* src = buf + hw * mask_dim;
                for (int c = 0; c < mask_dim; ++c)
                {
                    dst[c] = deqnt_affine_to_f32(src[c], zp, scale);
                }
#endif
            }
        }
        else
        {
            // NCHW：通道平面跨步读取。用 8 通道 x 8 像素块 + 寄存器内 8x8 转置，
            // 读连续、写连续，全程向量化（ARM NEON Programmer's Guide 标准算法）。
            float* proto_data = proto.ptr<float>(0);
#if defined(__ARM_NEON)
            const int16x8_t vzp = vdupq_n_s16(static_cast<int16_t>(zp));
            const float32x4_t vscale = vdupq_n_f32(scale);
            const int ch_blocks = mask_dim / 8;
            const int ch_rem = mask_dim % 8;
            int hw = 0;
            for (; hw + 8 <= plane_size; hw += 8)
            {
                float* dst = proto_data + hw * mask_dim;
                for (int cb = 0; cb < ch_blocks; ++cb)
                {
                    const int c0 = cb * 8;
                    const int8_t* sp = buf + c0 * plane_size + hw;
                    int8x8_t r0 = vld1_s8(sp);
                    int8x8_t r1 = vld1_s8(sp + plane_size);
                    int8x8_t r2 = vld1_s8(sp + 2 * plane_size);
                    int8x8_t r3 = vld1_s8(sp + 3 * plane_size);
                    int8x8_t r4 = vld1_s8(sp + 4 * plane_size);
                    int8x8_t r5 = vld1_s8(sp + 5 * plane_size);
                    int8x8_t r6 = vld1_s8(sp + 6 * plane_size);
                    int8x8_t r7 = vld1_s8(sp + 7 * plane_size);

                    int8x8x2_t a0 = vtrn_s8(r0, r1);
                    int8x8x2_t a1 = vtrn_s8(r2, r3);
                    int8x8x2_t a2 = vtrn_s8(r4, r5);
                    int8x8x2_t a3 = vtrn_s8(r6, r7);

                    int16x4x2_t b0 = vtrn_s16(vreinterpret_s16_s8(a0.val[0]), vreinterpret_s16_s8(a1.val[0]));
                    int16x4x2_t b1 = vtrn_s16(vreinterpret_s16_s8(a0.val[1]), vreinterpret_s16_s8(a1.val[1]));
                    int16x4x2_t b2 = vtrn_s16(vreinterpret_s16_s8(a2.val[0]), vreinterpret_s16_s8(a3.val[0]));
                    int16x4x2_t b3 = vtrn_s16(vreinterpret_s16_s8(a2.val[1]), vreinterpret_s16_s8(a3.val[1]));

                    int32x2x2_t q0 = vtrn_s32(vreinterpret_s32_s16(b0.val[0]), vreinterpret_s32_s16(b2.val[0]));
                    int32x2x2_t q1 = vtrn_s32(vreinterpret_s32_s16(b0.val[1]), vreinterpret_s32_s16(b2.val[1]));
                    int32x2x2_t q2 = vtrn_s32(vreinterpret_s32_s16(b1.val[0]), vreinterpret_s32_s16(b3.val[0]));
                    int32x2x2_t q3 = vtrn_s32(vreinterpret_s32_s16(b1.val[1]), vreinterpret_s32_s16(b3.val[1]));

                    const int8x8_t col[8] = {
                        vreinterpret_s8_s32(q0.val[0]), vreinterpret_s8_s32(q2.val[0]),
                        vreinterpret_s8_s32(q1.val[0]), vreinterpret_s8_s32(q3.val[0]),
                        vreinterpret_s8_s32(q0.val[1]), vreinterpret_s8_s32(q2.val[1]),
                        vreinterpret_s8_s32(q1.val[1]), vreinterpret_s8_s32(q3.val[1]),
                    };
                    for (int k = 0; k < 8; ++k)
                    {
                        const int16x8_t d = vsubq_s16(vmovl_s8(col[k]), vzp);
                        float* row = dst + k * mask_dim + c0;
                        vst1q_f32(row, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d))), vscale));
                        vst1q_f32(row + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d))), vscale));
                    }
                }
                // 通道数非 8 倍数的余数通道，标量兜底
                if (ch_rem > 0)
                {
                    const int c_base = ch_blocks * 8;
                    for (int k = 0; k < 8; ++k)
                    {
                        float* row = dst + k * mask_dim + c_base;
                        for (int c = 0; c < ch_rem; ++c)
                        {
                            row[c] = deqnt_affine_to_f32(buf[(c_base + c) * plane_size + hw + k], zp, scale);
                        }
                    }
                }
            }
            for (; hw < plane_size; ++hw)
            {
                float* dst = proto_data + hw * mask_dim;
                for (int c = 0; c < mask_dim; ++c)
                {
                    dst[c] = deqnt_affine_to_f32(buf[c * plane_size + hw], zp, scale);
                }
            }
#else
            for (int hw = 0; hw < plane_size; ++hw)
            {
                float* dst = proto.ptr<float>(hw);
                for (int c = 0; c < mask_dim; ++c)
                {
                    dst[c] = deqnt_affine_to_f32(buf[c * plane_size + hw], zp, scale);
                }
            }
#endif
        }
    }
    else
    {
        float* buf = reinterpret_cast<float*>(outputs[proto_index].buf);
        if (nhwc)
        {
            // 每像素连续，直接 memcpy
            for (int hw = 0; hw < plane_size; ++hw)
            {
                std::memcpy(proto.ptr<float>(hw), buf + hw * mask_dim, mask_dim * sizeof(float));
            }
        }
        else
        {
            for (int hw = 0; hw < plane_size; ++hw)
            {
                float* dst = proto.ptr<float>(hw);
                for (int c = 0; c < mask_dim; ++c)
                {
                    dst[c] = buf[c * plane_size + hw];
                }
            }
        }
    }

    return true;
}

int post_process_seg(rknn_app_context_t* app_ctx,
                     void* outputs,
                     letterbox_t* letter_box,
                     float conf_threshold,
                     float nms_threshold,
                     seg_detect_result_list* seg_results)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    rknn_output* _outputs = reinterpret_cast<rknn_output*>(outputs);
    if (!app_ctx || !_outputs || !app_ctx->output_attrs || !seg_results)
    {
        return -1;
    }

    seg_results->masks.clear();
    seg_results->boxes.clear();
    seg_results->scores.clear();
    seg_results->class_ids.clear();

    int output_count = app_ctx->io_num.n_output;
    if (output_count < 2)
    {
        return 0;
    }

    bool keep_mask = get_seg_mask_enabled();
    int proto_index = output_count - 1;
    cv::Mat proto;
    int mask_dim = 0;
    int mask_h = 0;
    int mask_w = 0;
    bool has_proto = false;
    if (keep_mask)
    {
        has_proto = build_proto_tensor(app_ctx, _outputs, proto_index, proto, mask_dim, mask_h, mask_w);
    }
    else
    {
        const auto& attr = app_ctx->output_attrs[proto_index];
        bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        if (nhwc)
        {
            mask_h = attr.dims[1];
            mask_w = attr.dims[2];
            mask_dim = attr.dims[3];
        }
        else
        {
            mask_dim = attr.dims[1];
            mask_h = attr.dims[2];
            mask_w = attr.dims[3];
        }
    }

    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> classIds;
    static thread_local std::vector<float> maskCoeffs;

    boxes.clear();
    scores.clear();
    classIds.clear();
    maskCoeffs.clear();

    int validCount = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    int class_count = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    int branch_output_count = output_count - 1;
    bool separate_outputs = branch_output_count % 3 == 0 && branch_output_count / 3 >= 2;
    int output_per_branch = separate_outputs ? branch_output_count / 3 : 0;

    if (separate_outputs)
    {
        for (int i = 0; i < 3; ++i)
        {
            int box_idx = i * output_per_branch;
            int score_idx = box_idx + 1;
            int score_sum_idx = output_per_branch == 4 ? box_idx + 2 : -1;
            int mask_idx = output_per_branch == 4 ? box_idx + 3 : box_idx + 2;

            if (mask_idx < 0 || mask_idx >= proto_index || score_idx >= proto_index || box_idx >= proto_index)
            {
                continue;
            }

            const auto& box_attr = app_ctx->output_attrs[box_idx];
            bool nhwc = box_attr.fmt != RKNN_TENSOR_NCHW;
            int grid_h = nhwc ? box_attr.dims[1] : box_attr.dims[2];
            int grid_w = nhwc ? box_attr.dims[2] : box_attr.dims[3];
            int box_channels = nhwc ? box_attr.dims[3] : box_attr.dims[1];
            int dfl_len = box_channels / 4;
            if (grid_h <= 0 || grid_w <= 0 || dfl_len <= 0)
            {
                continue;
            }

            int stride = model_in_h / grid_h;

            int8_t* score_sum_tensor_i8 = nullptr;
            float* score_sum_tensor_f32 = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0f;
            if (score_sum_idx >= 0 && score_sum_idx < proto_index)
            {
                score_sum_tensor_i8 = reinterpret_cast<int8_t*>(_outputs[score_sum_idx].buf);
                score_sum_tensor_f32 = reinterpret_cast<float*>(_outputs[score_sum_idx].buf);
                score_sum_zp = app_ctx->output_attrs[score_sum_idx].zp;
                score_sum_scale = app_ctx->output_attrs[score_sum_idx].scale;
            }

            if (app_ctx->is_quant)
            {
                validCount += process_i8_seg(reinterpret_cast<int8_t*>(_outputs[box_idx].buf), box_attr.zp, box_attr.scale,
                                             reinterpret_cast<int8_t*>(_outputs[score_idx].buf), app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                             score_sum_tensor_i8, score_sum_zp, score_sum_scale,
                                             reinterpret_cast<int8_t*>(_outputs[mask_idx].buf), app_ctx->output_attrs[mask_idx].zp, app_ctx->output_attrs[mask_idx].scale,
                                             grid_h, grid_w, stride, dfl_len, class_count, mask_dim, nhwc, keep_mask,
                                             boxes, scores, classIds, maskCoeffs, conf_threshold);
            }
            else
            {
                validCount += process_fp32_seg(reinterpret_cast<float*>(_outputs[box_idx].buf),
                                               reinterpret_cast<float*>(_outputs[score_idx].buf),
                                               score_sum_tensor_f32,
                                               reinterpret_cast<float*>(_outputs[mask_idx].buf),
                                               grid_h, grid_w, stride, dfl_len, class_count, mask_dim, nhwc, keep_mask,
                                               boxes, scores, classIds, maskCoeffs, conf_threshold);
            }
        }
    }
    else
    {
        for (int i = 0; i < branch_output_count; ++i)
        {
            const auto& attr = app_ctx->output_attrs[i];
            bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
            int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
            int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
            int channels = nhwc ? attr.dims[3] : attr.dims[1];
            int dfl_len = (channels - class_count - mask_dim) / 4;
            if (grid_h <= 0 || grid_w <= 0 || dfl_len <= 0)
            {
                continue;
            }

            int stride = model_in_h / grid_h;

            if (app_ctx->is_quant)
            {
                validCount += process_i8_seg_combined(reinterpret_cast<int8_t*>(_outputs[i].buf), attr.zp, attr.scale,
                                                      nullptr, 0, 1.0f,
                                                      grid_h, grid_w, stride, dfl_len, class_count, mask_dim, nhwc, keep_mask,
                                                      boxes, scores, classIds, maskCoeffs, conf_threshold);
            }
            else
            {
                validCount += process_fp32_seg_combined(reinterpret_cast<float*>(_outputs[i].buf),
                                                        nullptr,
                                                        grid_h, grid_w, stride, dfl_len, class_count, mask_dim, nhwc, keep_mask,
                                                        boxes, scores, classIds, maskCoeffs, conf_threshold);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_last_seg_post_decode_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (validCount <= 0)
    {
        g_last_seg_post_sort_time_ms = 0.0;
        g_last_seg_post_nms_time_ms = 0.0;
        g_last_seg_post_mask_time_ms = 0.0;
        g_last_seg_post_time_ms = g_last_seg_post_decode_time_ms;
        g_last_seg_candidate_count = 0;
        g_last_seg_class_count = 0;
        return 0;
    }

    std::vector<int> indexArray;
    indexArray.reserve(validCount);
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(scores, 0, validCount - 1, indexArray);
    auto t2 = std::chrono::high_resolution_clock::now();
    g_last_seg_post_sort_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if (validCount > OBJ_NUMB_MAX_SIZE)
    {
        validCount = OBJ_NUMB_MAX_SIZE;
        indexArray.resize(validCount);
    }

    const int seg_top_k = 64;
    if (validCount > seg_top_k)
    {
        validCount = seg_top_k;
        indexArray.resize(validCount);
    }

    std::set<int> class_set(std::begin(classIds), std::end(classIds));
    g_last_seg_candidate_count = validCount;
    g_last_seg_class_count = static_cast<int>(class_set.size());
    for (auto c : class_set)
    {
        nms(validCount, boxes, classIds, indexArray, c, nms_threshold, 4);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    g_last_seg_post_nms_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int crop_left = letter_box ? letter_box->crop_x : 0;
    int crop_top = letter_box ? letter_box->crop_y : 0;
    int crop_w = letter_box ? letter_box->crop_w : model_in_w;
    int crop_h = letter_box ? letter_box->crop_h : model_in_h;
    int crop_right = crop_left + std::max(1, crop_w);
    int crop_bottom = crop_top + std::max(1, crop_h);
    cv::Rect valid_rect_mask(0, 0, 0, 0);
    float mask_scale_w = 1.0f;
    float mask_scale_h = 1.0f;
    if (has_proto && mask_dim > 0)
    {
        int pad_x = letter_box ? letter_box->x_pad : 0;
        int pad_y = letter_box ? letter_box->y_pad : 0;
        int valid_w = std::max(1, model_in_w - pad_x * 2);
        int valid_h = std::max(1, model_in_h - pad_y * 2);
        mask_scale_w = static_cast<float>(mask_w) / static_cast<float>(model_in_w);
        mask_scale_h = static_cast<float>(mask_h) / static_cast<float>(model_in_h);
        int pad_x_mask = std::max(0, std::min(mask_w - 1, static_cast<int>(pad_x * mask_scale_w + 0.5f)));
        int pad_y_mask = std::max(0, std::min(mask_h - 1, static_cast<int>(pad_y * mask_scale_h + 0.5f)));
        int valid_w_mask = std::max(1, std::min(mask_w - pad_x_mask, static_cast<int>(valid_w * mask_scale_w + 0.5f)));
        int valid_h_mask = std::max(1, std::min(mask_h - pad_y_mask, static_cast<int>(valid_h * mask_scale_h + 0.5f)));
        valid_rect_mask = cv::Rect(pad_x_mask, pad_y_mask, valid_w_mask, valid_h_mask);
    }

    // 与 seg_top_k 对齐：密集场景（如车流）目标常超过 32 个，放宽到 64 减少漏检
    const int mask_top_k = 64;
    int mask_kept = 0;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }
        if (mask_kept >= mask_top_k)
        {
            break;
        }
        ++mask_kept;

        int n = indexArray[i];
        float x1 = boxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = boxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + boxes[n * 4 + 2];
        float y2 = y1 + boxes[n * 4 + 3];

        int left = static_cast<int>(clamp(x1, 0, model_in_w) / letter_box->scale) + crop_left;
        int top = static_cast<int>(clamp(y1, 0, model_in_h) / letter_box->scale) + crop_top;
        int right = static_cast<int>(clamp(x2, 0, model_in_w) / letter_box->scale) + crop_left;
        int bottom = static_cast<int>(clamp(y2, 0, model_in_h) / letter_box->scale) + crop_top;

        left = clamp(left, crop_left, crop_right);
        top = clamp(top, crop_top, crop_bottom);
        right = clamp(right, crop_left, crop_right);
        bottom = clamp(bottom, crop_top, crop_bottom);

        if (right <= left || bottom <= top)
        {
            continue;
        }

        seg_results->boxes.emplace_back(cv::Rect(cv::Point(left, top), cv::Point(right, bottom)));
        seg_results->scores.emplace_back(scores[n]);
        seg_results->class_ids.emplace_back(classIds[n]);

        if (!has_proto || mask_dim <= 0)
        {
            seg_results->masks.emplace_back(cv::Mat());
            continue;
        }

        const float* coeff_ptr = maskCoeffs.data() + n * mask_dim;
        const int proto_plane = mask_h * mask_w;
        cv::Mat mask(1, proto_plane, CV_32F);
        {
            // 转置布局下 mask = proto * coeff^T：逐像素行点积，缓存友好 + NEON
            float* mask_data = mask.ptr<float>(0);
            for (int i = 0; i < proto_plane; ++i)
            {
                const float* row = proto.ptr<float>(i);
                float sum = 0.0f;
                int c = 0;
#if defined(__ARM_NEON)
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (; c + 16 <= mask_dim; c += 16)
                {
                    float32x4_t a0 = vld1q_f32(row + c);
                    float32x4_t a1 = vld1q_f32(row + c + 4);
                    float32x4_t a2 = vld1q_f32(row + c + 8);
                    float32x4_t a3 = vld1q_f32(row + c + 12);
                    float32x4_t b0 = vld1q_f32(coeff_ptr + c);
                    float32x4_t b1 = vld1q_f32(coeff_ptr + c + 4);
                    float32x4_t b2 = vld1q_f32(coeff_ptr + c + 8);
                    float32x4_t b3 = vld1q_f32(coeff_ptr + c + 12);
                    vacc = vmlaq_f32(vacc, a0, b0);
                    vacc = vmlaq_f32(vacc, a1, b1);
                    vacc = vmlaq_f32(vacc, a2, b2);
                    vacc = vmlaq_f32(vacc, a3, b3);
                }
                sum = horizontalSumF32(vacc);
#endif
                for (; c < mask_dim; ++c)
                {
                    sum += row[c] * coeff_ptr[c];
                }
                mask_data[i] = sum;
            }
        }
        mask = mask.reshape(1, mask_h);
        cv::Mat mask_nopad = mask(valid_rect_mask);

        cv::Mat mask_bin_small;
        cv::threshold(mask_nopad, mask_bin_small, segMaskLogitThreshold(), 255, cv::THRESH_BINARY);
        mask_bin_small.convertTo(mask_bin_small, CV_8U);

        cv::Rect roi(left - crop_left, top - crop_top, right - left, bottom - top);
        roi.x = std::max(0, std::min(roi.x, crop_w - 1));
        roi.y = std::max(0, std::min(roi.y, crop_h - 1));
        roi.width = std::max(1, std::min(roi.width, crop_w - roi.x));
        roi.height = std::max(1, std::min(roi.height, crop_h - roi.y));

        // 在掩膜小分辨率上取框足迹（浮点 floor/ceil 全覆盖），再 INTER_LINEAR
        // 放大到框大小以平滑边缘；避免整帧放大（候选多时的大头开销）。
        const float f2m_w = letter_box->scale * mask_scale_w;
        const float f2m_h = letter_box->scale * mask_scale_h;
        const float mask_x0 = roi.x * f2m_w;
        const float mask_y0 = roi.y * f2m_h;
        const float mask_x1 = (roi.x + roi.width) * f2m_w;
        const float mask_y1 = (roi.y + roi.height) * f2m_h;
        // 同 yolov26_seg：框可能整体落在掩膜有效区外，钳制出非空 ROI 防越界崩溃
        const int m_x = std::min(std::max(0, static_cast<int>(std::floor(mask_x0))), valid_rect_mask.width - 1);
        const int m_y = std::min(std::max(0, static_cast<int>(std::floor(mask_y0))), valid_rect_mask.height - 1);
        const int m_x2 = std::min(valid_rect_mask.width, std::max(m_x + 1, static_cast<int>(std::ceil(mask_x1))));
        const int m_y2 = std::min(valid_rect_mask.height, std::max(m_y + 1, static_cast<int>(std::ceil(mask_y1))));
        cv::Rect small_roi(m_x, m_y, m_x2 - m_x, m_y2 - m_y);
        cv::Mat mask_box;
        cv::resize(mask_bin_small(small_roi), mask_box, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_LINEAR);
        seg_results->masks.emplace_back(std::move(mask_box));
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    g_last_seg_post_mask_time_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    g_last_seg_post_time_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();

    return 0;
}

double get_last_seg_postprocess_time_ms() { return g_last_seg_post_time_ms; }
double get_last_seg_postprocess_decode_time_ms() { return g_last_seg_post_decode_time_ms; }
double get_last_seg_postprocess_sort_time_ms() { return g_last_seg_post_sort_time_ms; }
double get_last_seg_postprocess_nms_time_ms() { return g_last_seg_post_nms_time_ms; }
double get_last_seg_postprocess_mask_time_ms() { return g_last_seg_post_mask_time_ms; }
int get_last_seg_candidate_count() { return g_last_seg_candidate_count; }
int get_last_seg_class_count() { return g_last_seg_class_count; }
