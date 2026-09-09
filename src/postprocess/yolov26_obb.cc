#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <opencv2/imgproc.hpp>

// YOLO26 OBB 后处理（非 end2end 原始头，one2one 头）
//
// 输出布局：3 x [1, 4+1+nc, H, W]（NCHW）或 [1, H, W, 4+1+nc]（NHWC），每尺度独立 zp/scale。
// **通道序（板端探针实测）：4 box + 1 angle + nc cls**
//   0-3: box 直接距离 l,t,r,b（reg_max=1，网格单位），解码按 ultralytics dist2rbox:
//       dx=(r-l)/2, dy=(b-t)/2；cx = dx*cos(angle) - dy*sin(angle) + (j+0.5)
//       cy = dx*sin(angle) + dy*cos(angle) + (i+0.5)；w=l+r, h=t+b；再 ×stride
//   4:   angle —— **原始弧度**（探针实测值域约 [-0.9, 0.9]；OBB26 头无 sigmoid，
//       训练损失直接与弧度目标比较并按 π 周期包裹）
//   5..4+nc: cls logit（未 sigmoid）
//   ⚠ 布局坑：不是 [box, cls, angle]！若按 [box, cls, angle] 读会把 angle 当 cls
//     （score 恒≈0.7、类全错）且把负 cls logit 当 angle（旋转乱飞）。
// one2one 头本应免 NMS，但分数塌缩让相邻 cell 同时命中 → 加旋转 IoU NMS 抑制重叠

namespace {
struct Y26OBBCandidate {
    float cx, cy, w, h;      // 模型输入像素坐标（中心点 + 全宽高）
    float angle;             // 弧度
    float score;
    int cls;
};

static int process_y26_obb_scale(const void* tensor, Y26TensorType ttype, int32_t zp, float scale,
                                 bool nhwc, int grid_h, int grid_w, int stride,
                                 int class_num, float logit_threshold, bool cls_sigmoided,
                                 std::vector<Y26OBBCandidate>& cands)
{
    const int grid_len = grid_h * grid_w;
    const int channels = 4 + 1 + class_num;  // box(4) + angle(1) + cls(nc)
    int valid = 0;

    auto read_ch = [&](int ci, int off) -> float {
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
                    const int ci = 5 + c;  // cls 从 ch5 开始（ch4 是 angle）
                    const int8_t q = nhwc ? qptr[off * channels + ci] : qptr[ci * grid_len + off];
                    if (q > max_q) { max_q = q; bestc = c; }
                }
                if (bestc < 0) continue;
                best_logit = (max_q - zp) * scale;
            } else {
                for (int c = 0; c < class_num; ++c) {
                    const int ci = 5 + c;
                    const float v = y26_tensor_at(tensor, ttype, nhwc ? off * channels + ci : ci * grid_len + off);
                    if (v > best_logit) { best_logit = v; bestc = c; }
                }
                if (bestc < 0) continue;
            }

            if (best_logit < logit_threshold) continue;

            const float l = read_ch(0, off);
            const float t = read_ch(1, off);
            const float r = read_ch(2, off);
            const float b = read_ch(3, off);
            // OBB26：angle 在 ch4，是原始弧度（无 sigmoid；探针实测值域约 [-0.9, 0.9]）
            const float angle = read_ch(4, off);

            const float dx = (r - l) * 0.5f;
            const float dy = (b - t) * 0.5f;
            const float cos_a = cosf(angle);
            const float sin_a = sinf(angle);

            Y26OBBCandidate c;
            c.cx = (dx * cos_a - dy * sin_a + j + 0.5f) * stride;
            c.cy = (dx * sin_a + dy * cos_a + i + 0.5f) * stride;
            c.w = (l + r) * stride;
            c.h = (t + b) * stride;
            c.angle = angle;
            c.score = cls_sigmoided ? best_logit : sigmoidf(best_logit);
            c.cls = bestc;
            cands.push_back(c);
            ++valid;
        }
    }
    return valid;
}
// 旋转框 NMS（OpenCV rotatedRectangleIntersection 算旋转 IoU）。
// one2one 头在 INT8 分数塌缩下会让目标周围多个相邻 cell 同时命中（score≈0.5），
// 不抑制会产生大量重叠旋转框（量化角度各异 → 视觉上"乱飞"）。
static void nms_y26_obb(std::vector<Y26OBBCandidate>& cands, float nms_threshold) {
    const size_t n = cands.size();
    if (n <= 1) return;
    std::vector<cv::RotatedRect> rects;
    rects.reserve(n);
    for (const auto& c : cands) {
        // 与绘制端 drawOBBResultsBGR 相同的 RotatedRect 构造（角度转度）
        rects.emplace_back(cv::Point2f(c.cx, c.cy), cv::Size2f(c.w, c.h), c.angle * 57.2957795f);
    }
    std::vector<bool> keep(n, true);
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < n; ++j) {
            if (!keep[j]) continue;
            const cv::RotatedRect& a = rects[i];
            const cv::RotatedRect& b = rects[j];
            // AABB 预筛：外接框不相交则跳过精确 IoU
            const float dx = std::abs(a.center.x - b.center.x);
            const float dy = std::abs(a.center.y - b.center.y);
            if (dx > 0.5f * (a.size.width + b.size.width) || dy > 0.5f * (a.size.height + b.size.height)) {
                continue;
            }
            std::vector<cv::Point2f> inter;
            const int isect = cv::rotatedRectangleIntersection(a, b, inter);
            if (isect == cv::INTERSECT_NONE || inter.size() < 3) continue;
            const float inter_area = static_cast<float>(cv::contourArea(inter));
            const float union_area = a.size.width * a.size.height + b.size.width * b.size.height - inter_area;
            if (union_area <= 0.0f) continue;
            if (inter_area / union_area > nms_threshold) {
                keep[j] = false;
            }
        }
    }
    size_t w = 0;
    for (size_t i = 0; i < n; ++i) {
        if (keep[i]) cands[w++] = cands[i];
    }
    cands.resize(w);
}
} // namespace

int post_process_yolov26_obb(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                             float conf_threshold, float nms_threshold,
                             obb_detect_result_list* od_results)
{
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    std::vector<Y26OBBCandidate> cands;
    cands.reserve(256);
    memset(od_results, 0, sizeof(obb_detect_result_list));

    const int model_in_h = app_ctx->model_height;
    const int class_count = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    // 方案2 自适应：首帧扫描 cls 值域判定语义（OBB cls 在 ch5 起，ch4 是 angle）
    if (app_ctx->cls_is_sigmoided == 0) {
        app_ctx->cls_is_sigmoided = y26_scan_cls_sigmoided(app_ctx, _outputs, class_count, 5);
        printf("[yolo26-obb] cls 输出判定: %s\n",
               app_ctx->cls_is_sigmoided == 1 ? "已 sigmoid（概率域，跳过二次 sigmoid）" : "logits（需 sigmoid）");
    }
    const bool cls_sigmoided = (app_ctx->cls_is_sigmoided == 1);

    // 置信度阈值：已 sigmoid → 概率域直接用 conf；logits → 转 logit 域 ln(conf/(1-conf))
    const float logit_thr = y26_conf_to_logit_threshold(cls_sigmoided, conf_threshold);

    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
        const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
        const int channels = nhwc ? attr.dims[3] : attr.dims[1];
        if (channels <= 5 || grid_h <= 0 || grid_w <= 0) continue;
        const int cls_num = std::min(channels - 5, class_count);
        if (cls_num <= 0) continue;
        const int stride = model_in_h / grid_h;
        const Y26TensorType ttype = y26_tensor_type_from_attr(app_ctx->is_quant, attr.type);
        process_y26_obb_scale(_outputs[i].buf, ttype, attr.zp, attr.scale, nhwc,
                              grid_h, grid_w, stride, cls_num, logit_thr, cls_sigmoided, cands);
    }

    if (cands.empty()) {
        return 0;
    }

    // 先按分数排序并截断到上限，再 NMS：FP16 真实分数下候选可达数百，NMS O(n²)
    // 的几何 IoU 在未截断时会算爆（曾实测掉到 5 FPS）。
    std::sort(cands.begin(), cands.end(), [](const Y26OBBCandidate& a, const Y26OBBCandidate& b) {
        return a.score > b.score;
    });
    if (cands.size() > OBJ_NUMB_MAX_SIZE) {
        cands.resize(OBJ_NUMB_MAX_SIZE);
    }
    // one2one 头 + 分数塌缩会产生重叠候选：按旋转 IoU 抑制（保留最高分）
    nms_y26_obb(cands, nms_threshold);

    const int crop_left = letter_box->crop_x;
    const int crop_top = letter_box->crop_y;
    const int crop_right = crop_left + std::max(1, letter_box->crop_w);
    const int crop_bottom = crop_top + std::max(1, letter_box->crop_h);

    int count = 0;
    for (const Y26OBBCandidate& d : cands) {
        obb_detect_result* result = &od_results->results[count];

        // 中心点/宽高映射回原图（模型坐标 - pad）/scale + crop
        const float cx_orig = (d.cx - letter_box->x_pad) / letter_box->scale + crop_left;
        const float cy_orig = (d.cy - letter_box->y_pad) / letter_box->scale + crop_top;
        const float w_orig = d.w / letter_box->scale;
        const float h_orig = d.h / letter_box->scale;

        // 与 yolov8_obb 结果约定一致：x,y = 未旋转 AABB 左上角，w/h = 全宽高，angle 弧度
        result->box.x = clamp(cx_orig - w_orig * 0.5f, crop_left, crop_right);
        result->box.y = clamp(cy_orig - h_orig * 0.5f, crop_top, crop_bottom);
        result->box.w = clamp(w_orig, 0, crop_right - crop_left);
        result->box.h = clamp(h_orig, 0, crop_bottom - crop_top);
        result->box.angle = d.angle;
        result->prop = d.score;
        result->cls_id = d.cls;
        ++count;
    }
    od_results->count = count;

    // 调试：RK_PIPE_DEBUG_Y26=1 打印流水线实际产出的 obb 结果
    static const bool y26o_dbg = []() {
        const char* v = getenv("RK_PIPE_DEBUG_Y26");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (y26o_dbg) {
        fprintf(stderr, "[Y26O] cand=%zu kept=%d:", cands.size(), od_results->count);
        for (int i = 0; i < od_results->count && i < 8; ++i) {
            const obb_detect_result& r = od_results->results[i];
            fprintf(stderr, " [%s %.3f (%.0f,%.0f %dx%d a=%.2f)]",
                    coco_cls_to_name(r.cls_id), r.prop,
                    r.box.x + r.box.w * 0.5f, r.box.y + r.box.h * 0.5f,
                    r.box.w, r.box.h, r.box.angle);
        }
        fprintf(stderr, "\n");
    }
    return 0;
}
