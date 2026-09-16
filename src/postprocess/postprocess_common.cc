#include "postprocess/postprocess_common.h"

#include <math.h>
#include <set>
#include <stdio.h>
#include <string>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// 读取一个元素为 float（int8 走 zp/scale 反量化，fp16 转 float，fp32 直读）
static inline float pp_common_at(const void* tensor, int ttype, int idx) {
    switch (ttype) {
        case 0:  // int8
            return static_cast<float>(static_cast<const int8_t*>(tensor)[idx]);
        case 1:  // fp16
            return fp16_to_float(static_cast<const uint16_t*>(tensor)[idx]);
        default:  // fp32
            return static_cast<const float*>(tensor)[idx];
    }
}

int y26_scan_cls_sigmoided(rknn_app_context_t* app_ctx, const rknn_output* outputs,
                           int class_count, int cls_start) {
    float gmin = 1e9f, gmax = -1e9f;
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
        const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
        const int channels = nhwc ? attr.dims[3] : attr.dims[1];
        if (grid_h <= 0 || grid_w <= 0) continue;
        // 只扫"含完整 cls 区间"的输出（channels - cls_start >= class_count），排除 seg 的 proto 输出
        const int cls_num = std::min(channels - cls_start, class_count);
        if (cls_num < class_count || cls_num <= 0) continue;
        const int grid_len = grid_h * grid_w;

        int ttype;
        if (!app_ctx->is_quant) {
            ttype = 2;  // fp32（want_float 已转换）
        } else {
            switch (attr.type) {
                case RKNN_TENSOR_INT8: ttype = 0; break;
                case RKNN_TENSOR_FLOAT16: ttype = 1; break;
                default: ttype = 2; break;
            }
        }
        const void* buf = outputs[i].buf;
        const int step = std::max(1, grid_len / 20000);  // 每输出采样 ≤ 2 万网格点
        for (int off = 0; off < grid_len; off += step) {
            for (int c = 0; c < cls_num; ++c) {
                const int idx = nhwc ? off * channels + cls_start + c : (cls_start + c) * grid_len + off;
                const float v = (ttype == 0)
                                    ? (static_cast<const int8_t*>(buf)[idx] - attr.zp) * attr.scale
                                    : pp_common_at(buf, ttype, idx);
                if (v < gmin) gmin = v;
                if (v > gmax) gmax = v;
            }
        }
    }
    return (gmin >= -1.0f && gmax <= 1.02f) ? 1 : -1;
}

int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

Y26TensorType y26_tensor_type_from_attr(bool is_quant, rknn_tensor_type type) {
    if (!is_quant) {
        return Y26TensorType::kFp32;  // runtime want_float 已转换
    }
    switch (type) {
        case RKNN_TENSOR_INT8: return Y26TensorType::kInt8;
        case RKNN_TENSOR_FLOAT16: return Y26TensorType::kFp16;
        default: return Y26TensorType::kFp32;
    }
}

float y26_tensor_at(const void* tensor, Y26TensorType type, int idx) {
    switch (type) {
        case Y26TensorType::kInt8:
            return static_cast<float>(static_cast<const int8_t*>(tensor)[idx]);
        case Y26TensorType::kFp16:
            return fp16_to_float(static_cast<const uint16_t*>(tensor)[idx]);
        case Y26TensorType::kFp32:
            return static_cast<const float*>(tensor)[idx];
    }
    return 0.0f;
}

float y26_conf_to_logit_threshold(bool cls_sigmoided, float conf_threshold) {
    if (cls_sigmoided) {
        return conf_threshold;
    }
    if (conf_threshold <= 0.0f) return -1e9f;
    if (conf_threshold >= 1.0f) return 1e9f;
    return logf(conf_threshold / (1.0f - conf_threshold));
}

void map_box_to_frame(float x1, float y1, float x2, float y2,
                      const letterbox_t* letter_box, int model_in_w, int model_in_h,
                      image_rect_t* box) {
    const int crop_left = letter_box->crop_x;
    const int crop_top = letter_box->crop_y;
    const int crop_right = crop_left + std::max(1, letter_box->crop_w);
    const int crop_bottom = crop_top + std::max(1, letter_box->crop_h);

    const float fx1 = x1 - letter_box->x_pad;
    const float fy1 = y1 - letter_box->y_pad;
    const float fx2 = x2 - letter_box->x_pad;
    const float fy2 = y2 - letter_box->y_pad;

    const int left = static_cast<int>(clamp(fx1, 0, model_in_w) / letter_box->scale) + crop_left;
    const int top = static_cast<int>(clamp(fy1, 0, model_in_h) / letter_box->scale) + crop_top;
    const int right = static_cast<int>(clamp(fx2, 0, model_in_w) / letter_box->scale) + crop_left;
    const int bottom = static_cast<int>(clamp(fy2, 0, model_in_h) / letter_box->scale) + crop_top;

    box->left = clamp(left, crop_left, crop_right);
    box->top = clamp(top, crop_top, crop_bottom);
    box->right = clamp(right, crop_left, crop_right);
    box->bottom = clamp(bottom, crop_top, crop_bottom);
}

void map_point_to_frame(float x, float y, const letterbox_t* letter_box,
                        int model_in_w, int model_in_h, bool content_clamp,
                        float* out_x, float* out_y) {
    const float tx = (x - letter_box->x_pad) / letter_box->scale;
    const float ty = (y - letter_box->y_pad) / letter_box->scale;
    float ux, uy;
    if (content_clamp) {
        // 钳到 letterbox 内容区:关键点不会落到填充区(yolov8-pose 行为)
        const float max_x = (model_in_w - 2.0f * letter_box->x_pad) / letter_box->scale;
        const float max_y = (model_in_h - 2.0f * letter_box->y_pad) / letter_box->scale;
        ux = std::max(0.0f, std::min(tx, max_x));
        uy = std::max(0.0f, std::min(ty, max_y));
    } else {
        // 钳到模型画布(yolov26 系行为)
        ux = std::max(0.0f, std::min(tx, static_cast<float>(model_in_w) / letter_box->scale));
        uy = std::max(0.0f, std::min(ty, static_cast<float>(model_in_h) / letter_box->scale));
    }
    *out_x = ux + letter_box->crop_x;
    *out_y = uy + letter_box->crop_y;
}

float fp16_to_float(uint16_t fp16_val) {
    uint32_t sign = (fp16_val >> 15) & 0x1;
    uint32_t exponent = (fp16_val >> 10) & 0x1F;
    uint32_t mantissa = fp16_val & 0x3FF;

    uint32_t f32_val = 0;
    if (exponent == 0 && mantissa == 0) {
        f32_val = sign << 31;
    } else if (exponent == 0) {
        exponent = 0x10 - 0x0F;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        f32_val = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    } else if (exponent == 0x1F) {
        f32_val = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f32_val = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    union { uint32_t u; float f; } conv;
    conv.u = f32_val;
    return conv.f;
}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                       float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms(int validCount, std::vector<float> &outputLocations, const std::vector<int>& classIds, std::vector<int> &order,
        int filterId, float threshold, int box_stride)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            int base0 = n * box_stride;
            int base1 = m * box_stride;
            float xmin0 = outputLocations[base0 + 0];
            float ymin0 = outputLocations[base0 + 1];
            float xmax0 = outputLocations[base0 + 0] + outputLocations[base0 + 2];
            float ymax0 = outputLocations[base0 + 1] + outputLocations[base0 + 3];

            float xmin1 = outputLocations[base1 + 0];
            float ymin1 = outputLocations[base1 + 1];
            float xmax1 = outputLocations[base1 + 0] + outputLocations[base1 + 2];
            float ymax1 = outputLocations[base1 + 1] + outputLocations[base1 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

#if defined(__ARM_NEON) && defined(__aarch64__)
// 16 个 int8 反量化 + 快速 exp（基 2 拆解 + 4 阶多项式）的 softmax 加权平均，
// 用于 DFL 解码。快速 exp 相对误差约 1e-3，对 int8 模型框回归影响可忽略。
static inline float dflSoftmaxMean16(const int8_t* vals, int32_t zp, float scale)
{
    static const float kWeights[4][4] = {
        {0.0f, 1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f, 7.0f},
        {8.0f, 9.0f, 10.0f, 11.0f},
        {12.0f, 13.0f, 14.0f, 15.0f},
    };
    const int16x8_t vzp = vdupq_n_s16(static_cast<int16_t>(zp));
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t vlog2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
    const float32x4_t c3 = vdupq_n_f32(0.05550410866482158f);
    const float32x4_t c4 = vdupq_n_f32(0.009618129107628477f);
    const int32x4_t v127 = vdupq_n_s32(127);
    const int32x4_t v0 = vdupq_n_s32(0);
    const int32x4_t v255 = vdupq_n_s32(255);

    const int8x16_t v = vld1q_s8(vals);
    const int16x8_t lo = vsubq_s16(vmovl_s8(vget_low_s8(v)), vzp);
    const int16x8_t hi = vsubq_s16(vmovl_s8(vget_high_s8(v)), vzp);
    const float32x4_t x[4] = {
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vscale),
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vscale),
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vscale),
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vscale),
    };

    float32x4_t e[4];
    for (int k = 0; k < 4; ++k) {
        const float32x4_t y = vmulq_f32(x[k], vlog2e);
        const float32x4_t nf = vrndnq_f32(y);
        const float32x4_t frac = vsubq_f32(y, nf);
        // 2^frac 4 阶多项式（frac ∈ [-0.5, 0.5)）
        float32x4_t poly = vmlaq_f32(c1, frac, vmlaq_f32(c2, frac, vmlaq_f32(c3, frac, vmulq_f32(c4, frac))));
        poly = vmlaq_f32(vdupq_n_f32(1.0f), frac, poly);
        // 2^n 通过指数域位操作（n+127)<<23，clamp 防止越界产生 NaN
        int32x4_t ni = vaddq_s32(vcvtq_s32_f32(nf), v127);
        ni = vmaxq_s32(vminq_s32(ni, v255), v0);
        const float32x4_t two_n = vreinterpretq_f32_s32(vshlq_n_s32(ni, 23));
        e[k] = vmulq_f32(poly, two_n);
    }

    const float32x4_t sumv = vaddq_f32(vaddq_f32(e[0], e[1]), vaddq_f32(e[2], e[3]));
    float32x4_t wsum = vmulq_f32(e[0], vld1q_f32(kWeights[0]));
    wsum = vmlaq_f32(wsum, e[1], vld1q_f32(kWeights[1]));
    wsum = vmlaq_f32(wsum, e[2], vld1q_f32(kWeights[2]));
    wsum = vmlaq_f32(wsum, e[3], vld1q_f32(kWeights[3]));

    const float sum = vaddvq_f32(sumv);
    const float weighted = vaddvq_f32(wsum);
    return sum > 0.0f ? weighted / sum : 0.0f;
}
#endif

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

void compute_dfl(float* tensor, int dfl_len, float* box){
    float exp_t[16];
    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] * inv_sum * i;
        }
        box[b] = acc_sum;
    }
}

void compute_dfl_contiguous_f32(const float* tensor, int dfl_len, float* box){
    float exp_t[16];
    if (dfl_len == 16){
        for (int b=0; b<4; b++){
            const float* src = tensor + b * 16;
            float e0 = expf(src[0]);  float e1 = expf(src[1]);  float e2 = expf(src[2]);  float e3 = expf(src[3]);
            float e4 = expf(src[4]);  float e5 = expf(src[5]);  float e6 = expf(src[6]);  float e7 = expf(src[7]);
            float e8 = expf(src[8]);  float e9 = expf(src[9]);  float e10= expf(src[10]); float e11= expf(src[11]);
            float e12= expf(src[12]); float e13= expf(src[13]); float e14= expf(src[14]); float e15= expf(src[15]);
            float sum = e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15;
            float inv = 1.0f / sum;
            float acc =
                e0 * inv * 0  + e1 * inv * 1  + e2 * inv * 2  + e3 * inv * 3  +
                e4 * inv * 4  + e5 * inv * 5  + e6 * inv * 6  + e7 * inv * 7  +
                e8 * inv * 8  + e9 * inv * 9  + e10* inv * 10 + e11* inv * 11 +
                e12* inv * 12 + e13* inv * 13 + e14* inv * 14 + e15* inv * 15;
            box[b] = acc;
        }
        return;
    }
    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        const float* src = tensor + b * dfl_len;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(src[i]);
            exp_sum += exp_t[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] * inv_sum * i;
        }
        box[b] = acc_sum;
    }
}

void compute_dfl_strided_f32_offset(const float* tensor, int grid_len, int dfl_len, int offset, float* box){
    float exp_t[16];
    if (dfl_len == 16){
        for (int b=0; b<4; b++){
            const float* src = tensor + b * 16 * grid_len + offset;
            float e0 = expf(src[0 * grid_len]);  float e1 = expf(src[1 * grid_len]);
            float e2 = expf(src[2 * grid_len]);  float e3 = expf(src[3 * grid_len]);
            float e4 = expf(src[4 * grid_len]);  float e5 = expf(src[5 * grid_len]);
            float e6 = expf(src[6 * grid_len]);  float e7 = expf(src[7 * grid_len]);
            float e8 = expf(src[8 * grid_len]);  float e9 = expf(src[9 * grid_len]);
            float e10= expf(src[10* grid_len]);  float e11= expf(src[11* grid_len]);
            float e12= expf(src[12* grid_len]);  float e13= expf(src[13* grid_len]);
            float e14= expf(src[14* grid_len]);  float e15= expf(src[15* grid_len]);
            float sum = e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15;
            float inv = 1.0f / sum;
            float acc =
                e0 * inv * 0  + e1 * inv * 1  + e2 * inv * 2  + e3 * inv * 3  +
                e4 * inv * 4  + e5 * inv * 5  + e6 * inv * 6  + e7 * inv * 7  +
                e8 * inv * 8  + e9 * inv * 9  + e10* inv * 10 + e11* inv * 11 +
                e12* inv * 12 + e13* inv * 13 + e14* inv * 14 + e15* inv * 15;
            box[b] = acc;
        }
        return;
    }
    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        const float* src = tensor + b * dfl_len * grid_len + offset;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(src[i * grid_len]);
            exp_sum += exp_t[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] * inv_sum * i;
        }
        box[b] = acc_sum;
    }
}

void compute_dfl_contiguous_i8(const int8_t* tensor, int dfl_len, int32_t zp, float scale, float* box){
    float exp_t[16];
    if (dfl_len == 16){
        for (int b=0; b<4; b++){
            const int8_t* src = tensor + b * 16;
#if defined(__ARM_NEON) && defined(__aarch64__)
            box[b] = dflSoftmaxMean16(src, zp, scale);
#else
            float e0 = expf(((float)src[0]  - (float)zp) * scale);  float e1 = expf(((float)src[1]  - (float)zp) * scale);
            float e2 = expf(((float)src[2]  - (float)zp) * scale);  float e3 = expf(((float)src[3]  - (float)zp) * scale);
            float e4 = expf(((float)src[4]  - (float)zp) * scale);  float e5 = expf(((float)src[5]  - (float)zp) * scale);
            float e6 = expf(((float)src[6]  - (float)zp) * scale);  float e7 = expf(((float)src[7]  - (float)zp) * scale);
            float e8 = expf(((float)src[8]  - (float)zp) * scale);  float e9 = expf(((float)src[9]  - (float)zp) * scale);
            float e10= expf(((float)src[10] - (float)zp) * scale);  float e11= expf(((float)src[11] - (float)zp) * scale);
            float e12= expf(((float)src[12] - (float)zp) * scale);  float e13= expf(((float)src[13] - (float)zp) * scale);
            float e14= expf(((float)src[14] - (float)zp) * scale);  float e15= expf(((float)src[15] - (float)zp) * scale);
            float sum = e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15;
            float inv = 1.0f / sum;
            float acc =
                e0 * inv * 0  + e1 * inv * 1  + e2 * inv * 2  + e3 * inv * 3  +
                e4 * inv * 4  + e5 * inv * 5  + e6 * inv * 6  + e7 * inv * 7  +
                e8 * inv * 8  + e9 * inv * 9  + e10* inv * 10 + e11* inv * 11 +
                e12* inv * 12 + e13* inv * 13 + e14* inv * 14 + e15* inv * 15;
            box[b] = acc;
#endif
        }
        return;
    }
    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        const int8_t* src = tensor + b * dfl_len;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(((float)src[i] - (float)zp) * scale);
            exp_sum += exp_t[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] * inv_sum * i;
        }
        box[b] = acc_sum;
    }
}

void compute_dfl_strided_i8_offset(const int8_t* tensor, int grid_len, int dfl_len, int offset, int32_t zp, float scale, float* box){
    float exp_t[16];
    if (dfl_len == 16){
        for (int b=0; b<4; b++){
            const int8_t* src = tensor + b * 16 * grid_len + offset;
#if defined(__ARM_NEON) && defined(__aarch64__)
            // NCHW 跨步布局：先收集 16 个值，再向量化反量化 + 快速 exp
            int8_t vals[16];
            for (int i = 0; i < 16; ++i) {
                vals[i] = src[i * grid_len];
            }
            box[b] = dflSoftmaxMean16(vals, zp, scale);
#else
            float e0 = expf(((float)src[0  * grid_len] - (float)zp) * scale);  float e1 = expf(((float)src[1  * grid_len] - (float)zp) * scale);
            float e2 = expf(((float)src[2  * grid_len] - (float)zp) * scale);  float e3 = expf(((float)src[3  * grid_len] - (float)zp) * scale);
            float e4 = expf(((float)src[4  * grid_len] - (float)zp) * scale);  float e5 = expf(((float)src[5  * grid_len] - (float)zp) * scale);
            float e6 = expf(((float)src[6  * grid_len] - (float)zp) * scale);  float e7 = expf(((float)src[7  * grid_len] - (float)zp) * scale);
            float e8 = expf(((float)src[8  * grid_len] - (float)zp) * scale);  float e9 = expf(((float)src[9  * grid_len] - (float)zp) * scale);
            float e10= expf(((float)src[10 * grid_len] - (float)zp) * scale);  float e11= expf(((float)src[11 * grid_len] - (float)zp) * scale);
            float e12= expf(((float)src[12 * grid_len] - (float)zp) * scale);  float e13= expf(((float)src[13 * grid_len] - (float)zp) * scale);
            float e14= expf(((float)src[14 * grid_len] - (float)zp) * scale);  float e15= expf(((float)src[15 * grid_len] - (float)zp) * scale);
            float sum = e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15;
            float inv = 1.0f / sum;
            float acc =
                e0 * inv * 0  + e1 * inv * 1  + e2 * inv * 2  + e3 * inv * 3  +
                e4 * inv * 4  + e5 * inv * 5  + e6 * inv * 6  + e7 * inv * 7  +
                e8 * inv * 8  + e9 * inv * 9  + e10* inv * 10 + e11* inv * 11 +
                e12* inv * 12 + e13* inv * 13 + e14* inv * 14 + e15* inv * 15;
            box[b] = acc;
#endif
        }
        return;
    }
    for (int b=0; b<4; b++){
        float exp_sum=0;
        float acc_sum=0;
        const int8_t* src = tensor + b * dfl_len * grid_len + offset;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = expf(((float)src[i * grid_len] - (float)zp) * scale);
            exp_sum += exp_t[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] * inv_sum * i;
        }
        box[b] = acc_sum;
    }
}

cv::Point2f mapPointToOriginal(const cv::Point2f& point,
                               const letterbox_t* letter_box,
                               int src_width,
                               int src_height) {
    if (!letter_box) {
        return cv::Point2f(std::clamp(point.x, 0.0f, static_cast<float>(src_width - 1)),
                           std::clamp(point.y, 0.0f, static_cast<float>(src_height - 1)));
    }
    float scale = letter_box->scale > 0.0f ? letter_box->scale : 1.0f;
    float mapped_x = (point.x - static_cast<float>(letter_box->x_pad)) / scale + static_cast<float>(letter_box->crop_x);
    float mapped_y = (point.y - static_cast<float>(letter_box->y_pad)) / scale + static_cast<float>(letter_box->crop_y);
    return cv::Point2f(std::clamp(mapped_x, 0.0f, static_cast<float>(std::max(1, src_width) - 1)),
                       std::clamp(mapped_y, 0.0f, static_cast<float>(std::max(1, src_height) - 1)));
}

