#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// YOLO26 seg 后处理（非 end2end 原始头，one2one 头）
//
// 输出（板端探针实测布局）：3 x [1, 116, H, W] + proto [1, 32, 160, 160]，NCHW（int8/fp16/fp32）
//   116 = 4(box 直接距离, 网格单位) + 80(cls logit) + 32(mask 系数)
//   box: 无 DFL，直接距离回归（reg_max=1）→ x1=(j+0.5-l)*stride（与 yolo26 detect 一致）
//   cls: logit → sigmoid，one2one 头 + 分数塌缩 → 排序 topk + 框 NMS 抑制重叠
//   mask: mask = sigmoid(proto @ coeff) → 阈值 → resize 到检测框

namespace {

// 每尺度解码：ch0-3 box 直接距离，ch4..4+class_num-1 cls logit，ch4+class_num.. mask 系数
static int process_y26_seg_scale(const void* tensor, Y26TensorType ttype, int32_t zp, float scale,
                                 bool nhwc, int grid_h, int grid_w, int stride,
                                 int class_num, int mask_dim, float logit_threshold, bool cls_sigmoided,
                                 std::vector<float>& boxes, std::vector<float>& scores,
                                 std::vector<int>& classIds, std::vector<float>& maskCoeffs)
{
    const int grid_len = grid_h * grid_w;
    const int channels = 4 + class_num + mask_dim;
    const int cls_start = 4;
    const int mask_start = 4 + class_num;
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
                const int8_t* qptr = static_cast<const int8_t*>(tensor);
                int8_t max_q = -128;
                for (int c = 0; c < class_num; ++c) {
                    const int8_t q = nhwc ? qptr[off * channels + cls_start + c]
                                          : qptr[(cls_start + c) * grid_len + off];
                    if (q > max_q) { max_q = q; bestc = c; }
                }
                if (bestc < 0) continue;
                best_logit = (max_q - zp) * scale;
            } else {
                for (int c = 0; c < class_num; ++c) {
                    const float v = y26_tensor_at(tensor, ttype, nhwc ? off * channels + cls_start + c
                                                                : (cls_start + c) * grid_len + off);
                    if (v > best_logit) { best_logit = v; bestc = c; }
                }
                if (bestc < 0) continue;
            }

            if (best_logit < logit_threshold) continue;

            const float l = read_ch(0, off);
            const float t = read_ch(1, off);
            const float r = read_ch(2, off);
            const float b = read_ch(3, off);

            const float x1 = (j + 0.5f - l) * stride;
            const float y1 = (i + 0.5f - t) * stride;
            const float x2 = (j + 0.5f + r) * stride;
            const float y2 = (i + 0.5f + b) * stride;

            boxes.push_back(x1);
            boxes.push_back(y1);
            boxes.push_back(x2 - x1);
            boxes.push_back(y2 - y1);
            scores.push_back(cls_sigmoided ? best_logit : sigmoidf(best_logit));
            classIds.push_back(bestc);

            for (int m = 0; m < mask_dim; ++m) {
                maskCoeffs.push_back(read_ch(mask_start + m, off));
            }
            ++valid;
        }
    }
    return valid;
}

// 提取 proto 到 (mask_h*mask_w) x mask_dim 的 float Mat（每行一个像素的系数）
static bool build_y26_proto(rknn_app_context_t* app_ctx, rknn_output* outputs, int proto_index,
                            cv::Mat& proto, int& mask_dim, int& mask_h, int& mask_w)
{
    if (proto_index < 0 || !app_ctx || !outputs || !app_ctx->output_attrs) return false;
    const auto& attr = app_ctx->output_attrs[proto_index];
    const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
    mask_dim = nhwc ? attr.dims[3] : attr.dims[1];
    mask_h = nhwc ? attr.dims[1] : attr.dims[2];
    mask_w = nhwc ? attr.dims[2] : attr.dims[3];
    if (mask_dim <= 0 || mask_h <= 0 || mask_w <= 0) return false;

    const int plane_size = mask_h * mask_w;
    proto.create(plane_size, mask_dim, CV_32F);
    float* dst = proto.ptr<float>(0);
    const bool is_quant = app_ctx->is_quant;

    if (is_quant && attr.type == RKNN_TENSOR_INT8) {
        const int8_t* buf = static_cast<const int8_t*>(outputs[proto_index].buf);
        const int32_t zp = attr.zp;
        const float sc = attr.scale;
        if (nhwc) {
            for (int hw = 0; hw < plane_size; ++hw) {
                const int8_t* src = buf + hw * mask_dim;
                float* row = dst + hw * mask_dim;
                for (int c = 0; c < mask_dim; ++c) row[c] = (src[c] - zp) * sc;
            }
        } else {
            for (int hw = 0; hw < plane_size; ++hw) {
                float* row = dst + hw * mask_dim;
                for (int c = 0; c < mask_dim; ++c) row[c] = (buf[c * plane_size + hw] - zp) * sc;
            }
        }
    } else {
        // fp16/fp32：want_float=true 路径下 runtime 已输出 float32
        const float* buf = static_cast<const float*>(outputs[proto_index].buf);
        if (nhwc) {
            for (int hw = 0; hw < plane_size; ++hw)
                std::memcpy(dst + hw * mask_dim, buf + hw * mask_dim, mask_dim * sizeof(float));
        } else {
            for (int hw = 0; hw < plane_size; ++hw) {
                float* row = dst + hw * mask_dim;
                for (int c = 0; c < mask_dim; ++c) row[c] = buf[c * plane_size + hw];
            }
        }
    }
    return true;
}

}  // namespace

int post_process_yolov26_seg(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                             float conf_threshold, float nms_threshold,
                             seg_detect_result_list* seg_results)
{
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    if (!app_ctx || !_outputs || !app_ctx->output_attrs || !seg_results) return -1;

    seg_results->masks.clear();
    seg_results->boxes.clear();
    seg_results->scores.clear();
    seg_results->class_ids.clear();

    const int output_count = static_cast<int>(app_ctx->io_num.n_output);
    if (output_count < 2) return 0;
    const int proto_index = output_count - 1;

    const bool keep_mask = get_seg_mask_enabled();
    cv::Mat proto;
    int mask_dim = 0, mask_h = 0, mask_w = 0;
    bool has_proto = false;
    if (keep_mask) {
        has_proto = build_y26_proto(app_ctx, _outputs, proto_index, proto, mask_dim, mask_h, mask_w);
    } else if (output_count >= 2) {
        // 不生成掩膜时仍需 mask_dim 以正确解码通道
        const auto& attr = app_ctx->output_attrs[proto_index];
        mask_dim = (attr.fmt != RKNN_TENSOR_NCHW) ? attr.dims[3] : attr.dims[1];
    }

    std::vector<float> boxes, scores, maskCoeffs;
    std::vector<int> classIds;
    boxes.reserve(512);
    scores.reserve(256);
    classIds.reserve(256);
    maskCoeffs.reserve(256 * (mask_dim > 0 ? mask_dim : 1));

    const int model_in_h = app_ctx->model_height;
    const int model_in_w = app_ctx->model_width;
    const int class_count = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    // 方案2 自适应：首帧扫描 cls 值域判定语义（proto 输出已由共享函数排除）
    if (app_ctx->cls_is_sigmoided == 0) {
        app_ctx->cls_is_sigmoided = y26_scan_cls_sigmoided(app_ctx, _outputs, class_count, 4);
        printf("[yolo26-seg] cls 输出判定: %s\n",
               app_ctx->cls_is_sigmoided == 1 ? "已 sigmoid（概率域，跳过二次 sigmoid）" : "logits（需 sigmoid）");
    }
    const bool cls_sigmoided = (app_ctx->cls_is_sigmoided == 1);

    // 置信度阈值：已 sigmoid → 概率域直接用 conf；logits → 转 logit 域 ln(conf/(1-conf))
    const float logit_thr = y26_conf_to_logit_threshold(cls_sigmoided, conf_threshold);

    for (int i = 0; i < proto_index; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
        const int grid_h = nhwc ? attr.dims[1] : attr.dims[2];
        const int grid_w = nhwc ? attr.dims[2] : attr.dims[3];
        const int channels = nhwc ? attr.dims[3] : attr.dims[1];
        if (channels <= 4 + class_count || grid_h <= 0 || grid_w <= 0) continue;
        const int cur_mask_dim = std::min(channels - 4 - class_count, mask_dim);
        if (cur_mask_dim <= 0) continue;
        const int stride = model_in_h / grid_h;
        const Y26TensorType ttype = y26_tensor_type_from_attr(app_ctx->is_quant, attr.type);
        process_y26_seg_scale(_outputs[i].buf, ttype, attr.zp, attr.scale, nhwc,
                              grid_h, grid_w, stride, class_count, cur_mask_dim, logit_thr, cls_sigmoided,
                              boxes, scores, classIds, maskCoeffs);
    }

    const int validCount = static_cast<int>(scores.size());
    if (validCount <= 0) return 0;

    // 按分数排序（nms() 内部按 order 顺序比较），截断到上限防 O(n²) 爆炸，再类内框 NMS
    std::vector<int> indexArray(validCount);
    for (int i = 0; i < validCount; ++i) indexArray[i] = i;
    std::sort(indexArray.begin(), indexArray.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    int n_valid = validCount;
    if (n_valid > OBJ_NUMB_MAX_SIZE) n_valid = OBJ_NUMB_MAX_SIZE;

    // nms() 会把被抑制的 order 置 -1，直接作用于 indexArray 的排序副本
    std::vector<int> nms_order(indexArray.begin(), indexArray.begin() + n_valid);
    for (int c = 0; c < class_count; ++c) {
        nms(n_valid, boxes, classIds, nms_order, c, nms_threshold, 4);
    }

    const int crop_left = letter_box ? letter_box->crop_x : 0;
    const int crop_top = letter_box ? letter_box->crop_y : 0;

    // 掩膜有效区（去掉 letterbox 黑边）
    cv::Rect valid_rect_mask(0, 0, 0, 0);
    float mask_scale_w = 1.0f, mask_scale_h = 1.0f;
    if (has_proto && mask_dim > 0) {
        const int pad_x = letter_box ? letter_box->x_pad : 0;
        const int pad_y = letter_box ? letter_box->y_pad : 0;
        const int valid_w = std::max(1, model_in_w - pad_x * 2);
        const int valid_h = std::max(1, model_in_h - pad_y * 2);
        mask_scale_w = static_cast<float>(mask_w) / static_cast<float>(model_in_w);
        mask_scale_h = static_cast<float>(mask_h) / static_cast<float>(model_in_h);
        const int pad_x_mask = std::max(0, std::min(mask_w - 1, static_cast<int>(pad_x * mask_scale_w + 0.5f)));
        const int pad_y_mask = std::max(0, std::min(mask_h - 1, static_cast<int>(pad_y * mask_scale_h + 0.5f)));
        const int valid_w_mask = std::max(1, std::min(mask_w - pad_x_mask, static_cast<int>(valid_w * mask_scale_w + 0.5f)));
        const int valid_h_mask = std::max(1, std::min(mask_h - pad_y_mask, static_cast<int>(valid_h * mask_scale_h + 0.5f)));
        valid_rect_mask = cv::Rect(pad_x_mask, pad_y_mask, valid_w_mask, valid_h_mask);
    }

    const int mask_top_k = 64;
    int mask_kept = 0;
    for (int i = 0; i < n_valid; ++i) {
        if (nms_order[i] == -1) continue;
        if (mask_kept >= mask_top_k) break;
        ++mask_kept;

        const int n = nms_order[i];
        const float x1 = boxes[n * 4 + 0];
        const float y1 = boxes[n * 4 + 1];
        const float x2 = x1 + boxes[n * 4 + 2];
        const float y2 = y1 + boxes[n * 4 + 3];

        image_rect_t mapped{};
        map_box_to_frame(x1, y1, x2, y2, letter_box, model_in_w, model_in_h, &mapped);
        const int box_left = mapped.left;
        const int box_top = mapped.top;
        const int box_right = mapped.right;
        const int box_bottom = mapped.bottom;
        if (box_right <= box_left || box_bottom <= box_top) continue;

        seg_results->boxes.emplace_back(cv::Rect(cv::Point(box_left, box_top), cv::Point(box_right, box_bottom)));
        seg_results->scores.emplace_back(scores[n]);
        seg_results->class_ids.emplace_back(classIds[n]);

        if (!has_proto || mask_dim <= 0) {
            seg_results->masks.emplace_back(cv::Mat());
            continue;
        }

        const float* coeff_ptr = maskCoeffs.data() + static_cast<size_t>(n) * mask_dim;
        const int proto_plane = mask_h * mask_w;
        cv::Mat mask(1, proto_plane, CV_32F);
        float* mask_data = mask.ptr<float>(0);
        for (int p = 0; p < proto_plane; ++p) {
            const float* row = proto.ptr<float>(p);
            float sum = 0.0f;
            for (int c = 0; c < mask_dim; ++c) sum += row[c] * coeff_ptr[c];
            mask_data[p] = sum;
        }
        mask = mask.reshape(1, mask_h);
        cv::Mat mask_nopad = mask(valid_rect_mask);

        // logit 阈值二值化（-0.5 兼容弱/量化掩膜，RK_PIPE_SEG_MASK_THRESH 可覆盖）
        static const float mask_thresh = []() {
            const char* v = std::getenv("RK_PIPE_SEG_MASK_THRESH");
            return (v && *v) ? static_cast<float>(std::atof(v)) : -0.5f;
        }();
        cv::Mat mask_bin_small;
        cv::threshold(mask_nopad, mask_bin_small, mask_thresh, 255, cv::THRESH_BINARY);
        mask_bin_small.convertTo(mask_bin_small, CV_8U);

        const int roi_w = box_right - box_left;
        const int roi_h = box_bottom - box_top;
        const float f2m_w = letter_box->scale * mask_scale_w;
        const float f2m_h = letter_box->scale * mask_scale_h;
        const float m_x0 = static_cast<float>(box_left - crop_left) * f2m_w;
        const float m_y0 = static_cast<float>(box_top - crop_top) * f2m_h;
        const float m_x1 = static_cast<float>(box_right - crop_left) * f2m_w;
        const float m_y1 = static_cast<float>(box_bottom - crop_top) * f2m_h;
        // 框投影到掩膜有效区，必须钳制出非空 ROI：极端情况下框整体落在有效区外
        // （如 letterbox 取整边界），旧逻辑 max(1, m_x2-m_x) 会生成越界 ROI 直接崩溃
        const int m_x = std::min(std::max(0, static_cast<int>(std::floor(m_x0))), valid_rect_mask.width - 1);
        const int m_y = std::min(std::max(0, static_cast<int>(std::floor(m_y0))), valid_rect_mask.height - 1);
        const int m_x2 = std::min(valid_rect_mask.width, std::max(m_x + 1, static_cast<int>(std::ceil(m_x1))));
        const int m_y2 = std::min(valid_rect_mask.height, std::max(m_y + 1, static_cast<int>(std::ceil(m_y1))));
        cv::Rect small_roi(m_x, m_y, m_x2 - m_x, m_y2 - m_y);
        cv::Mat mask_box;
        cv::resize(mask_bin_small(small_roi), mask_box, cv::Size(roi_w, roi_h), 0, 0, cv::INTER_LINEAR);
        seg_results->masks.emplace_back(std::move(mask_box));
    }

    // 调试：RK_PIPE_DEBUG_Y26=1 打印流水线实际产出的 seg 结果（与其他 yolo26 任务保持一致）
    static const bool y26s_dbg = []() {
        const char* v = std::getenv("RK_PIPE_DEBUG_Y26");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    if (y26s_dbg) {
        fprintf(stderr, "[Y26S] cand=%d kept=%zu:", validCount, seg_results->boxes.size());
        const size_t n_show = std::min<size_t>(seg_results->boxes.size(), 8);
        for (size_t i = 0; i < n_show; ++i) {
            const cv::Rect& r = seg_results->boxes[i];
            fprintf(stderr, " [%s %.3f (%d,%d %dx%d) mask=%dpx]",
                    coco_cls_to_name(seg_results->class_ids[i]), seg_results->scores[i],
                    r.x, r.y, r.width, r.height,
                    seg_results->masks[i].empty() ? 0 : cv::countNonZero(seg_results->masks[i]));
        }
        fprintf(stderr, "\n");
    }
    return 0;
}
