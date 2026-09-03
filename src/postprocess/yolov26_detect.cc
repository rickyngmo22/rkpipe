#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>

// YOLO26 detect 后处理（非 end2end 原始头）
//
// 输出布局（与 YOLOv8 不同）：
//   - 无 DFL：box 通道为直接距离回归（reg_max=1），4 通道 (l,t,r,b)，网格单位
//   - cls 为未 sigmoid 的 logits
//   - one2one 头天然 NMS-free：阈值过滤 + 全尺度 topk 即可，无需 NMS
//   - 每个尺度一个张量 [1, 4+nc, H, W]（NCHW）或 [1, H, W, 4+nc]（NHWC）

namespace {
struct Y26Candidate {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

enum class Y26TensorType { kInt8, kFp16, kFp32 };

// 读一个元素为 float（int8 走 zp/scale 反量化，fp16 转 float，fp32 直读）
static inline float y26_at(const void* tensor, Y26TensorType ttype, int idx) {
    switch (ttype) {
        case Y26TensorType::kInt8:
            return static_cast<float>(static_cast<const int8_t*>(tensor)[idx]);
        case Y26TensorType::kFp16:
            return fp16_to_float(static_cast<const uint16_t*>(tensor)[idx]);
        case Y26TensorType::kFp32:
            return static_cast<const float*>(tensor)[idx];
    }
    return 0.0f;
}

// 单尺度解码。ttype 按该输出张量实际类型指定（混合量化下各输出类型可能不同）。
// cls_sigmoided=true 时 cls 已是概率[0,1]（方案2 图内 sigmoid），阈值/分数都走概率域，不再二次 sigmoid。
static int process_y26_scale(const void* tensor, Y26TensorType ttype, int32_t zp, float scale,
                             bool nhwc, int grid_h, int grid_w, int stride, int class_num,
                             float logit_threshold, bool cls_sigmoided, std::vector<Y26Candidate>& cands)
{
    const int grid_len = grid_h * grid_w;
    const int channels = 4 + class_num;
    int valid = 0;

    auto read_box = [&](int ci, int off) -> float {
        const int idx = nhwc ? off * channels + ci : ci * grid_len + off;
        if (ttype == Y26TensorType::kInt8) {
            return (static_cast<const int8_t*>(tensor)[idx] - zp) * scale;
        }
        return y26_at(tensor, ttype, idx);
    };

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            const int off = i * grid_w + j;
            int bestc = -1;
            float best_logit = -1e9f;

            if (ttype == Y26TensorType::kInt8) {
                // int8 域内找最大（反量化单调），避免逐类 sigmoid
                const int8_t* qptr = static_cast<const int8_t*>(tensor);
                int8_t max_q = -128;
                for (int c = 0; c < class_num; ++c) {
                    const int8_t q = nhwc ? qptr[off * channels + 4 + c] : qptr[(4 + c) * grid_len + off];
                    if (q > max_q) { max_q = q; bestc = c; }
                }
                if (bestc < 0) continue;
                best_logit = (max_q - zp) * scale;
            } else {
                for (int c = 0; c < class_num; ++c) {
                    const float v = y26_at(tensor, ttype, nhwc ? off * channels + 4 + c : (4 + c) * grid_len + off);
                    if (v > best_logit) { best_logit = v; bestc = c; }
                }
                if (bestc < 0) continue;
            }

            if (best_logit < logit_threshold) continue;

            const float l = read_box(0, off);
            const float t = read_box(1, off);
            const float r = read_box(2, off);
            const float b = read_box(3, off);

            Y26Candidate c;
            c.x1 = (j + 0.5f - l) * stride;
            c.y1 = (i + 0.5f - t) * stride;
            c.x2 = (j + 0.5f + r) * stride;
            c.y2 = (i + 0.5f + b) * stride;
            c.score = cls_sigmoided ? best_logit : sigmoidf(best_logit);
            c.cls = bestc;
            cands.push_back(c);
            ++valid;
        }
    }
    return valid;
}
} // namespace

int post_process_yolov26(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                         float conf_threshold, float /*nms_threshold*/,
                         object_detect_result_list* od_results)
{
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    std::vector<Y26Candidate> cands;
    cands.reserve(512);
    memset(od_results, 0, sizeof(object_detect_result_list));

    const int model_in_w = app_ctx->model_width;
    const int model_in_h = app_ctx->model_height;
    const int class_count = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    // 方案2 自适应：首帧扫描 cls 值域判定语义（logits 需 sigmoid / 已 sigmoid 概率 [0,1]）
    if (app_ctx->cls_is_sigmoided == 0) {
        app_ctx->cls_is_sigmoided = y26_scan_cls_sigmoided(app_ctx, _outputs, class_count, 4);
        printf("[yolo26] cls 输出判定: %s\n",
               app_ctx->cls_is_sigmoided == 1 ? "已 sigmoid（概率域，跳过二次 sigmoid）" : "logits（需 sigmoid）");
    }
    const bool cls_sigmoided = (app_ctx->cls_is_sigmoided == 1);

    // 置信度阈值：已 sigmoid → 概率域直接用 conf；logits → 转 logit 域 ln(conf/(1-conf))
    float logit_thr;
    if (cls_sigmoided) {
        logit_thr = conf_threshold;
    } else {
        if (conf_threshold <= 0.0f) logit_thr = -1e9f;
        else if (conf_threshold >= 1.0f) logit_thr = 1e9f;
        else logit_thr = logf(conf_threshold / (1.0f - conf_threshold));
    }

    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
        const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
        const int channels = nhwc ? attr.dims[3] : attr.dims[1];
        if (channels <= 4 || grid_h <= 0 || grid_w <= 0) continue;
        const int cls_num = std::min(channels - 4, class_count);
        const int stride = model_in_h / grid_h;
        // 张量解码类型：
        //  - is_quant=false（全 FP16/FP32 模型）→ want_float=true，runtime 已把全部输出
        //    反量化为 float32，统一按 fp32 读
        //  - is_quant=true（INT8/混合量化）→ want_float=false，各输出保持原生类型，逐张量判断
        Y26TensorType ttype;
        if (!app_ctx->is_quant) {
            ttype = Y26TensorType::kFp32;
        } else {
            switch (attr.type) {
                case RKNN_TENSOR_INT8: ttype = Y26TensorType::kInt8; break;
                case RKNN_TENSOR_FLOAT16: ttype = Y26TensorType::kFp16; break;
                default: ttype = Y26TensorType::kFp32; break;
            }
        }
        process_y26_scale(_outputs[i].buf, ttype, attr.zp, attr.scale, nhwc,
                          grid_h, grid_w, stride, cls_num, logit_thr, cls_sigmoided, cands);
    }

    if (cands.empty()) {
        return 0;
    }

    // one2one 头免 NMS：按置信度降序取 topk
    std::sort(cands.begin(), cands.end(), [](const Y26Candidate& a, const Y26Candidate& b) {
        return a.score > b.score;
    });
    if (cands.size() > OBJ_NUMB_MAX_SIZE) {
        cands.resize(OBJ_NUMB_MAX_SIZE);
    }

    const int crop_left = letter_box->crop_x;
    const int crop_top = letter_box->crop_y;
    const int crop_right = crop_left + std::max(1, letter_box->crop_w);
    const int crop_bottom = crop_top + std::max(1, letter_box->crop_h);

    int count = 0;
    for (const Y26Candidate& d : cands) {
        const float x1 = d.x1 - letter_box->x_pad;
        const float y1 = d.y1 - letter_box->y_pad;
        const float x2 = d.x2 - letter_box->x_pad;
        const float y2 = d.y2 - letter_box->y_pad;

        const int left = static_cast<int>(clamp(x1, 0, model_in_w) / letter_box->scale) + crop_left;
        const int top = static_cast<int>(clamp(y1, 0, model_in_h) / letter_box->scale) + crop_top;
        const int right = static_cast<int>(clamp(x2, 0, model_in_w) / letter_box->scale) + crop_left;
        const int bottom = static_cast<int>(clamp(y2, 0, model_in_h) / letter_box->scale) + crop_top;

        od_results->results[count].box.left = clamp(left, crop_left, crop_right);
        od_results->results[count].box.top = clamp(top, crop_top, crop_bottom);
        od_results->results[count].box.right = clamp(right, crop_left, crop_right);
        od_results->results[count].box.bottom = clamp(bottom, crop_top, crop_bottom);
        od_results->results[count].prop = d.score;
        od_results->results[count].cls_id = d.cls;
        ++count;
    }
    od_results->count = count;
    // 调试：RK_PIPE_DEBUG_Y26=1 打印流水线实际产出的检测（用于排查"不画框"类问题）
    static const bool y26_dbg = []() {
        const char* v = getenv("RK_PIPE_DEBUG_Y26");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (y26_dbg) {
        fprintf(stderr, "[Y26] total_cand=%zu kept=%d:", cands.size(), od_results->count);
        for (int i = 0; i < od_results->count && i < 8; ++i) {
            const object_detect_result& r = od_results->results[i];
            fprintf(stderr, " [%s %.3f (%d,%d,%d,%d)]",
                    coco_cls_to_name(r.cls_id), r.prop, r.box.left, r.box.top, r.box.right, r.box.bottom);
        }
        fprintf(stderr, "\n");
    }
    return 0;
}
