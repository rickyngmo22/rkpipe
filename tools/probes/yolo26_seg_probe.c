// yolo26_seg_probe.c - 探针：yolo26n_seg.rknn 输出布局判定
// 预期 3 x [1,116,H,W] + proto[1,32,160,160]
// 116 = 4 box + 80 cls + 32 mask。用每通道均值分布判定布局：
//   box 通道均值>0（距离），cls 通道均值很负（logit），mask 系数均值≈0
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
    const char* model_path = argc > 1 ? argv[1] : "/userdata/rk_pipe/model/yolo26n_seg.rknn";
    const char* img_path   = argc > 2 ? argv[2] : "/userdata/rk_pipe/frame_obb_test.jpg";
    float conf = argc > 3 ? atof(argv[3]) : 0.1f;
    printf("== yolo26_seg_probe model=%s img=%s conf=%.2f\n", model_path, img_path, conf);

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

    // 对前 3 个尺度输出打印每通道 mean（布局判定依据：box>0, cls 很负, mask≈0）
    for (int oi = 0; oi < io.n_output - 1; oi++) {
        rknn_tensor_attr& a = out_attr[oi];
        int C = a.fmt == RKNN_TENSOR_NCHW ? a.dims[1] : a.dims[3];
        int H = a.fmt == RKNN_TENSOR_NCHW ? a.dims[2] : a.dims[1];
        int W = a.fmt == RKNN_TENSOR_NCHW ? a.dims[3] : a.dims[2];
        int GL = H * W;
        float* p = (float*)out[oi].buf;
        printf("== OUT[%d] C=%d H=%d W=%d channel means:\n", oi, C, H, W);
        for (int c = 0; c < C; c++) {
            double sum = 0;
            for (int g = 0; g < GL; g++) sum += p[c*GL + g];
            printf("  %s%02d: %8.3f", (c % 8 == 0) ? "\n    " : "", c, (float)(sum / GL));
        }
        printf("\n");
    }

    // proto 输出统计
    {
        rknn_tensor_attr& a = out_attr[io.n_output - 1];
        int C = a.fmt == RKNN_TENSOR_NCHW ? a.dims[1] : a.dims[3];
        int H = a.fmt == RKNN_TENSOR_NCHW ? a.dims[2] : a.dims[1];
        int W = a.fmt == RKNN_TENSOR_NCHW ? a.dims[3] : a.dims[2];
        int GL = H * W;
        float* p = (float*)out[io.n_output - 1].buf;
        printf("== PROTO OUT[%d] C=%d H=%d W=%d per-channel range:\n", io.n_output - 1, C, H, W);
        for (int c = 0; c < C; c++) {
            float mn = 1e9f, mx = -1e9f;
            for (int g = 0; g < GL; g++) {
                float v = p[c*GL + g];
                if (v > mx) mx = v; if (v < mn) mn = v;
            }
            printf("  ch%02d: min=%7.3f max=%7.3f\n", c, mn, mx);
        }
    }

    rknn_outputs_release(ctx, io.n_output, out);
    rknn_destroy(ctx);
    printf("== done\n");
    return 0;
}
