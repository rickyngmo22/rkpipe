// yolo26_obb_probe.c - 探针：yolo26n_obb.rknn（非 end2end）输出布局 + dist2rbox 解码自检
// 预期 3 x [1, 4+nc+1, H, W] NCHW（nc=15 → 20 通道），angle 原始弧度
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

static const int MODEL_SIZE = 640;

static void dump_attr(const char* tag, rknn_tensor_attr* a) {
    printf("%s[%d] name=%s n_dims=%d dims=[%d,%d,%d,%d] fmt=%s type=%s qnt=%s zp=%d scale=%f\n",
           tag, a->index, a->name, a->n_dims, a->dims[0], a->dims[1], a->dims[2], a->dims[3],
           a->fmt == RKNN_TENSOR_NCHW ? "NCHW" : (a->fmt == RKNN_TENSOR_NHWC ? "NHWC" : "other"),
           a->type == RKNN_TENSOR_INT8 ? "int8" : (a->type == RKNN_TENSOR_FLOAT32 ? "fp32" : (a->type == RKNN_TENSOR_FLOAT16 ? "fp16" : "?")),
           a->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ? "asym" : (a->qnt_type == RKNN_TENSOR_QNT_DFP ? "dfp" : "none"),
           a->zp, a->scale);
}

static unsigned char* read_file(const char* p, int* sz) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = (int)n; return b;
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void letterbox(const cv::Mat& src, cv::Mat& out_rgb, float& scale, float& pad_x, float& pad_y) {
    float sc = std::min((float)MODEL_SIZE / src.cols, (float)MODEL_SIZE / src.rows);
    int rw = std::max(1, (int)std::lround(src.cols * sc));
    int rh = std::max(1, (int)std::lround(src.rows * sc));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(rw, rh));
    cv::Mat canvas(MODEL_SIZE, MODEL_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    int px = (MODEL_SIZE - rw) / 2, py = (MODEL_SIZE - rh) / 2;
    resized.copyTo(canvas(cv::Rect(px, py, rw, rh)));
    cv::cvtColor(canvas, out_rgb, cv::COLOR_BGR2RGB);
    scale = sc; pad_x = (float)px; pad_y = (float)py;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/userdata/rk_pipe/model/yolo26n_obb.rknn";
    const char* img_path   = argc > 2 ? argv[2] : "/userdata/rk_pipe/test_images/000000000139.jpg";
    float conf = argc > 3 ? atof(argv[3]) : 0.1f;
    printf("== yolo26_obb_probe model=%s img=%s conf=%.2f\n", model_path, img_path, conf);

    int msz = 0; unsigned char* mdata = read_file(model_path, &msz);
    if (!mdata) { printf("read model fail\n"); return -1; }
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, mdata, msz, 0, NULL); free(mdata);
    if (ret < 0) { printf("rknn_init fail ret=%d\n", ret); return -1; }

    rknn_input_output_num io;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) { printf("query io fail\n"); return -1; }
    printf("in=%d out=%d\n", io.n_input, io.n_output);

    rknn_tensor_attr in_attr[8], out_attr[16];
    for (int i = 0; i < io.n_input; i++) { memset(&in_attr[i],0,sizeof(in_attr[i])); in_attr[i].index=i; rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&in_attr[i],sizeof(in_attr[i])); dump_attr("IN ",&in_attr[i]); }
    for (int i = 0; i < io.n_output; i++) { memset(&out_attr[i],0,sizeof(out_attr[i])); out_attr[i].index=i; rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&out_attr[i],sizeof(out_attr[i])); dump_attr("OUT",&out_attr[i]); }

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) { printf("imread fail\n"); return -1; }
    cv::Mat rgb; float scale, px, py;
    letterbox(img, rgb, scale, px, py);

    rknn_input in; memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    in.size = MODEL_SIZE*MODEL_SIZE*3; in.buf = rgb.data;
    if (rknn_inputs_set(ctx, 1, &in) != 0) { printf("inputs_set fail\n"); return -1; }
    ret = rknn_run(ctx, NULL);
    if (ret < 0) { printf("rknn_run fail ret=%d\n", ret); return -1; }
    printf("rknn_run OK\n");

    rknn_output out[16];
    for (int i = 0; i < io.n_output; i++) { memset(&out[i],0,sizeof(out[i])); out[i].index=i; out[i].want_float=1; }
    ret = rknn_outputs_get(ctx, io.n_output, out, NULL);
    if (ret < 0) { printf("outputs_get fail ret=%d\n", ret); return -1; }

    for (int oi = 0; oi < io.n_output; oi++) {
        rknn_tensor_attr& a = out_attr[oi];
        int C = a.fmt == RKNN_TENSOR_NCHW ? a.dims[1] : a.dims[3];
        int H = a.fmt == RKNN_TENSOR_NCHW ? a.dims[2] : a.dims[1];
        int W = a.fmt == RKNN_TENSOR_NCHW ? a.dims[3] : a.dims[2];
        int GL = H * W;
        float* p = (float*)out[oi].buf;
        int nc = C - 5;  // 4 box + 1 angle + nc cls
        printf("== OUT[%d] C=%d H=%d W=%d nc=%d\n", oi, C, H, W, nc);
        if (nc < 1) continue;
        int stride = MODEL_SIZE / H;
        printf("  per-channel range (ch0-3 box, ch4 angle, ch5..%d cls):\n", 4 + nc);
        for (int c = 0; c < C; c++) {
            float mn = 1e9f, mx = -1e9f, mean = 0;
            for (int g = 0; g < GL; g++) {
                float v = p[c*GL + g];
                if (v > mx) mx = v; if (v < mn) mn = v; mean += v;
            }
            if (c < 4 || c == 4 || c == 3) {
                printf("    ch%02d: min=%9.3f max=%9.3f mean=%8.3f%s\n", c, mn, mx, GL ? mean/GL : 0,
                       c == 4 ? " <== angle" : "");
            }
        }
        // top cells：按最佳 cls logit（ch5 起）排序
        struct Cell { int off; float s; int cls; };
        std::vector<Cell> cells;
        for (int g = 0; g < GL; g++) {
            float best = -1e9f; int bc = -1;
            for (int c = 0; c < nc; c++) {
                float v = p[(5+c)*GL + g];
                if (v > best) { best = v; bc = c; }
            }
            if (bc >= 0) cells.push_back({g, sigmoid(best), bc});
        }
        std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b){ return a.s > b.s; });
        printf("  top cells:\n");
        for (int k = 0; k < 8 && k < (int)cells.size(); k++) {
            int g = cells[k].off;
            int j = g % W, i = g / W;
            float l = p[0*GL+g], t = p[1*GL+g], r = p[2*GL+g], b = p[3*GL+g];
            float angle = p[4*GL+g];  // ch4 = angle（弧度）
            // dist2rbox 解码（模型像素坐标）
            float dx = (r - l) * 0.5f, dy = (b - t) * 0.5f;
            float ca = cosf(angle), sa = sinf(angle);
            float cx = (dx*ca - dy*sa + j + 0.5f) * stride;
            float cy = (dx*sa + dy*ca + i + 0.5f) * stride;
            float w = (l + r) * stride, h = (t + b) * stride;
            printf("    cell(%d,%d) cls=%d score=%.3f raw=[%.2f %.2f %.2f %.2f a=%.3f] "
                   "rbox_pix=(cx=%.1f cy=%.1f w=%.1f h=%.1f a=%.1fdeg)\n",
                   i, j, cells[k].cls, cells[k].s, l, t, r, b, angle,
                   cx, cy, w, h, angle * 57.2958f);
        }
    }

    rknn_outputs_release(ctx, io.n_output, out);
    rknn_destroy(ctx);
    printf("== done\n");
    return 0;
}
