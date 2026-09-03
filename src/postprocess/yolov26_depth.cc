#include "postprocess/postprocess.h"
#include "postprocess/postprocess_common.h"

#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>

// YOLO26 depth 后处理
//
// 模型输出：单张量 [1, 1, H, W]（NCHW int8/fp16/fp32），H=W=模型输入尺寸(640)。
// 值域：depth = exp(clamp(head,-4,5))，正值米制（本模型量化范围实测约 [0.5, 27.7]m）。
//
// 处理：反量化 → 按 letterbox 裁掉黑边并放大回原帧尺寸 → 归一化到 CV_8UC1（近=亮）。
// 归一化上限可用 RK_PIPE_DEPTH_MAX 覆盖（米，0=按当前帧 max 自适应，默认 20）。
// 归一化下限/上限可用 RK_PIPE_DEPTH_HISTOGRAM=1 切换为直方图百分位（1%/99%）自适应，
// 避免近处噪点/远处离群值把对比度拉平。

static float depthMaxMeters() {
    const char* v = std::getenv("RK_PIPE_DEPTH_MAX");
    if (!v || !*v) return 20.0f;
    const float out = std::atof(v);
    return out <= 0.0f ? 0.0f : out;
}

static bool depthHistogramEnabled() {
    const char* v = std::getenv("RK_PIPE_DEPTH_HISTOGRAM");
    return v && *v && std::string(v) != "0";
}

// 直方图百分位范围：把 [mn, mx] 分成 bins 个桶，返回累计占比达 p_lo%/p_hi% 的边界值，
// 用于剔除极小/极大离群值，避免它们主导 min-max 归一化。
static void histogramPercentileRange(const cv::Mat& m, double& lo, double& hi,
                                     double p_lo = 1.0, double p_hi = 99.0) {
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(m, &mn, &mx);
    if (mx - mn < 1e-6) {
        lo = mn;
        hi = mx;
        return;
    }
    const int bins = 256;
    std::vector<int> hist(static_cast<size_t>(bins), 0);
    const float inv = static_cast<float>(bins) / static_cast<float>(mx - mn);
    const float base = static_cast<float>(mn);
    const float* p = m.ptr<float>(0);
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>((p[i] - base) * inv);
        if (b < 0) b = 0;
        else if (b >= bins) b = bins - 1;
        ++hist[static_cast<size_t>(b)];
    }
    const int total = std::max(1, n);
    const int lo_target = static_cast<int>(total * p_lo / 100.0);
    const int hi_target = static_cast<int>(total * p_hi / 100.0);
    int cum = 0;
    lo = mn;
    hi = mx;
    for (int b = 0; b < bins; ++b) {
        cum += hist[static_cast<size_t>(b)];
        if (lo == mn && cum >= lo_target) {
            lo = mn + static_cast<double>(b) * (mx - mn) / bins;
        }
        if (cum >= hi_target) {
            hi = mn + static_cast<double>(b) * (mx - mn) / bins;
            break;
        }
    }
}

int post_process_yolov26_depth(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                               float /*conf_threshold*/, float /*nms_threshold*/, cv::Mat* depth_out,
                               float* depth_lo, float* depth_hi)
{
    if (!app_ctx || !outputs || !depth_out) return -1;
    rknn_output* _outputs = static_cast<rknn_output*>(outputs);
    if (app_ctx->io_num.n_output < 1) return -1;

    const rknn_tensor_attr& attr = app_ctx->output_attrs[0];
    const bool nhwc = attr.fmt != RKNN_TENSOR_NCHW;
    const int H = nhwc ? attr.dims[1] : attr.dims[2];
    const int W = nhwc ? attr.dims[2] : attr.dims[3];
    if (H <= 0 || W <= 0) return -1;

    // 1) 反量化到 float32
    cv::Mat depth_f(H, W, CV_32F);
    if (app_ctx->is_quant && attr.type == RKNN_TENSOR_INT8) {
        // int8：dst = src*scale + (-zp*scale)，convertTo 向量化
        cv::Mat qview(H, W, CV_8SC1, _outputs[0].buf);
        qview.convertTo(depth_f, CV_32F, attr.scale, -static_cast<float>(attr.zp) * attr.scale);
    } else if (app_ctx->is_quant && attr.type == RKNN_TENSOR_FLOAT16) {
        const uint16_t* buf = static_cast<const uint16_t*>(_outputs[0].buf);
        float* dst = depth_f.ptr<float>(0);
        const int n = H * W;
        for (int i = 0; i < n; ++i) dst[i] = fp16_to_float(buf[i]);
    } else {
        // want_float=true：runtime 已输出 float32
        std::memcpy(depth_f.data, _outputs[0].buf, static_cast<size_t>(H) * W * sizeof(float));
    }

    // 2) 裁掉 letterbox 黑边：直接输出"有效区子图"（不放大到原帧）。
    //    原帧目标位置 roi 由调用方（DepthDetector::runInference）根据 letterbox 计算；
    //    绘制时再放大一次，省掉每个 worker 每帧的 1080p resize/minMax/convertTo 后处理。
    const int pad_x = letter_box ? letter_box->x_pad : 0;
    const int pad_y = letter_box ? letter_box->y_pad : 0;
    const int crop_w = std::max(1, letter_box ? letter_box->crop_w : W);
    const int crop_h = std::max(1, letter_box ? letter_box->crop_h : H);
    const float scale = (letter_box && letter_box->scale > 0.0f) ? letter_box->scale : 1.0f;
    int vx = std::max(0, std::min(pad_x, W - 1));
    int vy = std::max(0, std::min(pad_y, H - 1));
    int vw = std::max(0, std::min(static_cast<int>(crop_w * scale + 0.5f), W - vx));
    int vh = std::max(0, std::min(static_cast<int>(crop_h * scale + 0.5f), H - vy));
    if (vw <= 0 || vh <= 0) { vx = 0; vy = 0; vw = W; vh = H; }

    cv::Mat depth_crop = depth_f(cv::Rect(vx, vy, vw, vh));

    // 3) 归一化到 8bit（近=亮）：[lo, hi] -> [0,255] 后反相（仅在有效区上统计）
    //    lo/hi 默认取帧内 min/max；RK_PIPE_DEPTH_HISTOGRAM=1 时用直方图百分位（1%/99%）
    //    剔除近处噪点与远处离群值，防止极端值把对比度拉平
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(depth_crop, &mn, &mx);
    const float cap = depthMaxMeters();
    double lo = mn, hi = mx;
    if (depthHistogramEnabled()) {
        histogramPercentileRange(depth_crop, lo, hi, 1.0, 99.0);
    }
    if (cap > 0.0f) {
        hi = std::min(hi, static_cast<double>(cap));
    }
    if (hi <= lo) {
        hi = lo + 1.0;
    }
    const float range = static_cast<float>(hi - lo);
    if (depth_lo) *depth_lo = static_cast<float>(lo);
    if (depth_hi) *depth_hi = static_cast<float>(hi);

    cv::Mat u8;
    depth_crop.convertTo(u8, CV_8UC1, 255.0 / range, -static_cast<float>(lo) * 255.0 / range);
    cv::bitwise_not(u8, u8);
    *depth_out = u8;
    return 0;
}
