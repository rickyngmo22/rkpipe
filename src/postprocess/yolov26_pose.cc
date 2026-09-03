#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>

// YOLO26 pose 后处理（非 end2end 原始头，one2one 头）
//
// 输出布局（与 yolo26 detect 同构）：3 x [1, 56, H, W]（NCHW）或 [1, H, W, 56]（NHWC），
// 每尺度独立 zp/scale。56 = 4(box 直接距离 l,t,r,b, 网格单位) + 1(cls logit) + 51(17 kpt x3)。
//   - box: 无 DFL，直接距离回归（reg_max=1）→ x1=(j+0.5-l)*stride（与 yolo26 detect 一致）
//   - cls: 未 sigmoid 的 logit
//   - kpt: 通道序 [x, y, conf] 每关键点一组；坐标按 ultralytics Pose26.kpts_decode 解码:
//       kpt_x_pix = (raw_x + j + 0.5) * stride
//       kpt_y_pix = (raw_y + i + 0.5) * stride
//       kpt_conf  = sigmoid(raw_conf)
//     （注意：YOLO26 姿态头是 RealNVP 流式头，解码是 (raw+anchor)*stride，
//      与常规 Pose 的 (raw*2+cell)*stride 不同，不能混用）
//   - one2one 头免 NMS：阈值过滤 + 全尺度 topk

namespace {
struct Y26PoseCandidate {
    float x1, y1, x2, y2;
    float score;
    pose_keypoint kpts[KEYPOINT_NUM];
};

enum class Y26TensorType { kInt8, kFp16, kFp32 };

// 读一个元素为 float（int8 走 zp/scale 反量化，fp16 转 float，fp32 直读）
static inline float y26p_at(const void* tensor, Y26TensorType ttype, int idx) {
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
static int process_y26_pose_scale(const void* tensor, Y26TensorType ttype, int32_t zp, float scale,
                                  bool nhwc, int grid_h, int grid_w, int stride,
                                  int class_num, int kpt_num, float logit_threshold, bool cls_sigmoided,
                                  std::vector<Y26PoseCandidate>& cands)
{
    const int grid_len = grid_h * grid_w;
    const int channels = 4 + class_num + kpt_num * 3;
    int valid = 0;

    auto read_ch = [&](int ci, int off) -> float {
        const int idx = nhwc ? off * channels + ci : ci * grid_len + off;
        if (ttype == Y26TensorType::kInt8) {
            return (static_cast<const int8_t*>(tensor)[idx] - zp) * scale;
        }
        return y26p_at(tensor, ttype, idx);
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
                    const float v = y26p_at(tensor, ttype, nhwc ? off * channels + 4 + c : (4 + c) * grid_len + off);
                    if (v > best_logit) { best_logit = v; bestc = c; }
                }
                if (bestc < 0) continue;
            }

            if (best_logit < logit_threshold) continue;

            const float l = read_ch(0, off);
            const float t = read_ch(1, off);
            const float r = read_ch(2, off);
            const float b = read_ch(3, off);

            Y26PoseCandidate c;
            c.x1 = (j + 0.5f - l) * stride;
            c.y1 = (i + 0.5f - t) * stride;
            c.x2 = (j + 0.5f + r) * stride;
            c.y2 = (i + 0.5f + b) * stride;
            c.score = cls_sigmoided ? best_logit : sigmoidf(best_logit);

            for (int k = 0; k < kpt_num && k < KEYPOINT_NUM; ++k) {
                const int cx = 4 + class_num + k * 3;
                const float rx = read_ch(cx, off);
                const float ry = read_ch(cx + 1, off);
                const float rc = read_ch(cx + 2, off);
                // YOLO26 用 Pose26.kpts_decode（RealNVP 流式头，与常规 Pose 不同）：
                //   kpt_pix = (raw + anchor) * stride，anchor = cell + 0.5
                // 注意不是常规 Pose 的 (raw*2 + cell) * stride！
                c.kpts[k].x = (rx + j + 0.5f) * stride;
                c.kpts[k].y = (ry + i + 0.5f) * stride;
                c.kpts[k].conf = sigmoidf(rc);
            }
            // 模型关键点数超过代码上限时，多余关键点置 0（不参与绘制）
            for (int k = kpt_num; k < KEYPOINT_NUM; ++k) {
                c.kpts[k].x = c.kpts[k].y = c.kpts[k].conf = 0.0f;
            }

            cands.push_back(c);
            ++valid;
        }
    }
    return valid;
}
} // namespace

int post_process_yolov26_pose(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                              float conf_threshold, float /*nms_threshold*/,
                              pose_detect_result_list* pd_results)
{
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    std::vector<Y26PoseCandidate> cands;
    cands.reserve(256);
    memset(pd_results, 0, sizeof(pose_detect_result_list));

    const int model_in_w = app_ctx->model_width;
    const int model_in_h = app_ctx->model_height;

    // 姿态模型只关心 person 类
    const int class_count = 1;

    // 方案2 自适应：首帧扫描 cls 值域判定语义（logits 需 sigmoid / 已 sigmoid 概率 [0,1]）
    if (app_ctx->cls_is_sigmoided == 0) {
        app_ctx->cls_is_sigmoided = y26_scan_cls_sigmoided(app_ctx, _outputs, class_count, 4);
        printf("[yolo26-pose] cls 输出判定: %s\n",
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
        if (channels <= 5 || grid_h <= 0 || grid_w <= 0) continue;
        const int kpt_num = (channels - 5) / 3;
        if (kpt_num <= 0 || kpt_num * 3 + 5 != channels) continue;
        const int stride = model_in_h / grid_h;
        // 张量解码类型（与 yolo26 detect 一致）：
        //  - is_quant=false（全 FP16/FP32 模型）→ want_float=true，runtime 已反量化为 float32
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
        process_y26_pose_scale(_outputs[i].buf, ttype, attr.zp, attr.scale, nhwc,
                               grid_h, grid_w, stride, class_count, kpt_num, logit_thr, cls_sigmoided, cands);
    }

    if (cands.empty()) {
        return 0;
    }

    // one2one 头免 NMS：按置信度降序取 topk
    std::sort(cands.begin(), cands.end(), [](const Y26PoseCandidate& a, const Y26PoseCandidate& b) {
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
    for (const Y26PoseCandidate& d : cands) {
        pose_detect_result* result = &pd_results->results[count];

        const float x1 = d.x1 - letter_box->x_pad;
        const float y1 = d.y1 - letter_box->y_pad;
        const float x2 = d.x2 - letter_box->x_pad;
        const float y2 = d.y2 - letter_box->y_pad;

        const int left = static_cast<int>(clamp(x1, 0, model_in_w) / letter_box->scale) + crop_left;
        const int top = static_cast<int>(clamp(y1, 0, model_in_h) / letter_box->scale) + crop_top;
        const int right = static_cast<int>(clamp(x2, 0, model_in_w) / letter_box->scale) + crop_left;
        const int bottom = static_cast<int>(clamp(y2, 0, model_in_h) / letter_box->scale) + crop_top;

        result->box.left = clamp(left, crop_left, crop_right);
        result->box.top = clamp(top, crop_top, crop_bottom);
        result->box.right = clamp(right, crop_left, crop_right);
        result->box.bottom = clamp(bottom, crop_top, crop_bottom);
        result->box_conf = d.score;
        result->cls_id = 0;

        for (int k = 0; k < KEYPOINT_NUM; ++k) {
            const float kpx = d.kpts[k].x - letter_box->x_pad;
            const float kpy = d.kpts[k].y - letter_box->y_pad;
            result->keypoints[k].x = clamp(kpx, 0, model_in_w) / letter_box->scale + crop_left;
            result->keypoints[k].y = clamp(kpy, 0, model_in_h) / letter_box->scale + crop_top;
            result->keypoints[k].conf = d.kpts[k].conf;
        }
        ++count;
    }
    pd_results->count = count;

    // 调试：RK_PIPE_DEBUG_Y26=1 打印流水线实际产出的姿态（用于排查"不画骨架"类问题）
    static const bool y26p_dbg = []() {
        const char* v = getenv("RK_PIPE_DEBUG_Y26");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (y26p_dbg) {
        fprintf(stderr, "[Y26P] cand=%zu kept=%d:", cands.size(), pd_results->count);
        for (int i = 0; i < pd_results->count && i < 8; ++i) {
            const pose_detect_result& r = pd_results->results[i];
            fprintf(stderr, " [%.3f (%d,%d,%d,%d) kp0=(%.0f,%.0f)]",
                    r.box_conf, r.box.left, r.box.top, r.box.right, r.box.bottom,
                    r.keypoints[0].x, r.keypoints[0].y);
        }
        fprintf(stderr, "\n");
    }
    return 0;
}
