#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>

// YOLO26 detect 后处理（非 end2end 原始头）
//
// 支持两种输出布局（按输出张量 channels 自动判定，逐张量独立 zp/scale）：
//
// 1) 融合布局（旧）：每尺度一个张量
//      [1, 4+nc, H, W]（NCHW）或 [1, H, W, 4+nc]（NHWC）
//    4 通道直接距离回归 (l,t,r,b) + nc 通道 cls logits
//
// 2) 拆分布局（split6，新，量化友好）：每尺度两个张量成对出现
//      box [1, 4,  H, W]   直接距离回归
//      cls [1, nc, H, W]   分类 logits（或已 sigmoid 概率，首帧自动判定）
//    box/cls 各自独立量化 scale，避免融合张量共享 scale 被 cls 极端 logits 撑爆。
//
// 共同点：无 DFL（reg_max=1）；one2one 头 NMS-free，阈值过滤 + 全尺度 topk。
// ⚠ 融合布局下 4 通道 box 张量会被误当"框+类"读（cls 从 ch4 起）→ 必须按 channels 判布局。

namespace {
struct Y26Candidate {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

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
        return y26_tensor_at(tensor, ttype, idx);
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
                    const float v = y26_tensor_at(tensor, ttype, nhwc ? off * channels + 4 + c : (4 + c) * grid_len + off);
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

// ---- 拆分布局（split6）：单尺度解码（box 与 cls 是两个独立张量，各带 zp/scale）----
static int process_y26_scale_split(const void* box_tensor, Y26TensorType box_t,
                                   int32_t box_zp, float box_scale, bool box_nhwc,
                                   const void* cls_tensor, Y26TensorType cls_t,
                                   int32_t cls_zp, float cls_scale, bool cls_nhwc,
                                   int grid_h, int grid_w, int stride, int class_num,
                                   float logit_threshold, bool cls_sigmoided,
                                   std::vector<Y26Candidate>& cands)
{
    const int grid_len = grid_h * grid_w;
    int valid = 0;

    auto read_box = [&](int ci, int off) -> float {
        const int idx = box_nhwc ? off * 4 + ci : ci * grid_len + off;
        if (box_t == Y26TensorType::kInt8) {
            return (static_cast<const int8_t*>(box_tensor)[idx] - box_zp) * box_scale;
        }
        return y26_tensor_at(box_tensor, box_t, idx);
    };

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            const int off = i * grid_w + j;
            int bestc = -1;
            float best_logit = -1e9f;

            if (cls_t == Y26TensorType::kInt8) {
                // int8 域内找最大（反量化单调），避免逐类 sigmoid
                const int8_t* qptr = static_cast<const int8_t*>(cls_tensor);
                int8_t max_q = -128;
                for (int c = 0; c < class_num; ++c) {
                    const int8_t q = cls_nhwc ? qptr[off * class_num + c] : qptr[c * grid_len + off];
                    if (q > max_q) { max_q = q; bestc = c; }
                }
                if (bestc < 0) continue;
                best_logit = (max_q - cls_zp) * cls_scale;
            } else {
                for (int c = 0; c < class_num; ++c) {
                    const float v = y26_tensor_at(cls_tensor, cls_t, cls_nhwc ? off * class_num + c : c * grid_len + off);
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

    // ---- 布局判定：逐输出扫描 channels ----
    //   box   张量: channels == 4
    //   cls   张量: channels == class_count（拆分布局，nc != 4 前提下无歧义）
    //   融合  张量: channels == 4 + class_count（或其它 > 4 的旧布局）
    struct Y26ScalePair { int box_idx = -1; int cls_idx = -1; int grid_h = 0; int grid_w = 0; };
    std::vector<Y26ScalePair> pairs;
    std::vector<int> fused_idx;
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
        const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
        const int channels = nhwc ? attr.dims[3] : attr.dims[1];
        if (grid_h <= 0 || grid_w <= 0) continue;

        if (channels == 4 && class_count != 4) {
            Y26ScalePair* p = nullptr;
            for (auto& q : pairs) {
                if (q.grid_h == grid_h && q.grid_w == grid_w && q.box_idx < 0) { p = &q; break; }
            }
            if (!p) { pairs.push_back(Y26ScalePair()); p = &pairs.back(); p->grid_h = grid_h; p->grid_w = grid_w; }
            p->box_idx = (int)i;
        } else if (channels == class_count) {
            Y26ScalePair* p = nullptr;
            for (auto& q : pairs) {
                if (q.grid_h == grid_h && q.grid_w == grid_w && q.cls_idx < 0) { p = &q; break; }
            }
            if (!p) { pairs.push_back(Y26ScalePair()); p = &pairs.back(); p->grid_h = grid_h; p->grid_w = grid_w; }
            p->cls_idx = (int)i;
        } else if (channels > 4) {
            fused_idx.push_back((int)i);
        }
    }

    bool split_layout = false;
    for (const auto& q : pairs) {
        if (q.box_idx >= 0 && q.cls_idx >= 0) { split_layout = true; break; }
    }

    // 方案2 自适应：首帧扫描 cls 值域判定语义（logits 需 sigmoid / 已 sigmoid 概率 [0,1]）
    // 拆分布局 cls_start=0：box 张量(channels==4) 因 cls_num<class_count 被扫描函数自动跳过
    if (app_ctx->cls_is_sigmoided == 0) {
        const int cls_start = split_layout ? 0 : 4;
        app_ctx->cls_is_sigmoided = y26_scan_cls_sigmoided(app_ctx, _outputs, class_count, cls_start);
        printf("[yolo26] 布局: %s | cls 输出判定: %s\n",
               split_layout ? "拆分 box/cls" : "融合 box+cls",
               app_ctx->cls_is_sigmoided == 1 ? "已 sigmoid（概率域，跳过二次 sigmoid）" : "logits（需 sigmoid）");
    }
    const bool cls_sigmoided = (app_ctx->cls_is_sigmoided == 1);

    // 置信度阈值：已 sigmoid → 概率域直接用 conf；logits → 转 logit 域 ln(conf/(1-conf))
    const float logit_thr = y26_conf_to_logit_threshold(cls_sigmoided, conf_threshold);

    if (split_layout) {
        for (const auto& q : pairs) {
            if (q.box_idx < 0 || q.cls_idx < 0) continue;
            const rknn_tensor_attr& battr = app_ctx->output_attrs[q.box_idx];
            const rknn_tensor_attr& cattr = app_ctx->output_attrs[q.cls_idx];
            const int stride = model_in_h / q.grid_h;
            const int cls_num = class_count;  // 拆分布局 cls 张量 channels == class_count
            process_y26_scale_split(
                _outputs[q.box_idx].buf, y26_tensor_type_from_attr(app_ctx->is_quant, battr.type),
                battr.zp, battr.scale, battr.fmt != RKNN_TENSOR_NCHW,
                _outputs[q.cls_idx].buf, y26_tensor_type_from_attr(app_ctx->is_quant, cattr.type),
                cattr.zp, cattr.scale, cattr.fmt != RKNN_TENSOR_NCHW,
                q.grid_h, q.grid_w, stride, cls_num, logit_thr, cls_sigmoided, cands);
        }
    } else {
        for (int i : fused_idx) {
            const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
            const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
            const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
            const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
            const int channels = nhwc ? attr.dims[3] : attr.dims[1];
            const int cls_num = std::min(channels - 4, class_count);
            const int stride = model_in_h / grid_h;
            const Y26TensorType ttype = y26_tensor_type_from_attr(app_ctx->is_quant, attr.type);
            process_y26_scale(_outputs[i].buf, ttype, attr.zp, attr.scale, nhwc,
                              grid_h, grid_w, stride, cls_num, logit_thr, cls_sigmoided, cands);
        }
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

    int count = 0;
    for (const Y26Candidate& d : cands) {
        map_box_to_frame(d.x1, d.y1, d.x2, d.y2, letter_box, model_in_w, model_in_h,
                         &od_results->results[count].box);
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
