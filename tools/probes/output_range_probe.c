// output_range_probe.c - 通用探针：打印任意 detect 模型各输出张量的逐通道 min/max/mean
// 用途：对比 yolo26 vs yolov8/v11 输出 logits 值域，解释 INT8 分数塌缩差异
// 用法: ./output_range_probe <model.rknn> [image.jpg]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

static void dump_attr(rknn_tensor_attr* a) {
    printf("  [%d] name=%-16s dims=[%d,%d,%d,%d] fmt=%s type=%s zp=%d scale=%f\n",
           a->index, a->name, a->dims[0], a->dims[1], a->dims[2], a->dims[3],
           a->fmt == RKNN_TENSOR_NCHW ? "NCHW" : (a->fmt == RKNN_TENSOR_NHWC ? "NHWC" : "other"),
           a->type == RKNN_TENSOR_INT8 ? "int8" : (a->type == RKNN_TENSOR_FLOAT32 ? "fp32" : (a->type == RKNN_TENSOR_FLOAT16 ? "fp16" : "?")),
           a->zp, a->scale);
}

static unsigned char* read_file(const char* p, int* sz) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = (int)n; return b;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/userdata/rk_pipe/model/yolo26n.rknn";
    const char* img_path   = argc > 2 ? argv[2] : "/userdata/rk_pipe/test_images/000000000139.jpg";
    printf("== output_range_probe model=%s img=%s\n", model_path, img_path);

    int msz = 0; unsigned char* mdata = read_file(model_path, &msz);
    if (!mdata) { printf("read model fail\n"); return -1; }
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, mdata, msz, 0, NULL); free(mdata);
    if (ret < 0) { printf("rknn_init fail ret=%d\n", ret); return -1; }

    rknn_sdk_version ver;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
        printf("runtime api=%s drv=%s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num io;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) { printf("query io fail\n"); return -1; }
    printf("in=%d out=%d\n", io.n_input, io.n_output);

    rknn_tensor_attr in_attr[8], out_attr[16];
    for (int i = 0; i < io.n_input; i++) { memset(&in_attr[i], 0, sizeof(in_attr[i])); in_attr[i].index = i; rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(in_attr[i])); dump_attr(&in_attr[i]); }
    for (int i = 0; i < io.n_output; i++) { memset(&out_attr[i], 0, sizeof(out_attr[i])); out_attr[i].index = i; rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(out_attr[i])); dump_attr(&out_attr[i]); }

    // 模型输入尺寸（NHWC: dims1=H, dims2=W; NCHW: dims2=H, dims3=W）
    const bool in_nhwc = in_attr[0].fmt == RKNN_TENSOR_NHWC;
    const int model_h = in_nhwc ? in_attr[0].dims[1] : in_attr[0].dims[2];
    const int model_w = in_nhwc ? in_attr[0].dims[2] : in_attr[0].dims[3];

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) { printf("imread fail\n"); return -1; }
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(model_w, model_h));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rknn_input in; memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    in.size = model_w * model_h * 3; in.buf = rgb.data;
    if (rknn_inputs_set(ctx, 1, &in) != 0) { printf("inputs_set fail\n"); return -1; }
    ret = rknn_run(ctx, NULL);
    if (ret < 0) { printf("rknn_run fail ret=%d\n", ret); return -1; }
    printf("rknn_run OK\n");

    rknn_output out[16];
    for (int i = 0; i < io.n_output; i++) { memset(&out[i], 0, sizeof(out[i])); out[i].index = i; out[i].want_float = 1; }
    ret = rknn_outputs_get(ctx, io.n_output, out, NULL);
    if (ret < 0) { printf("outputs_get fail ret=%d\n", ret); return -1; }

    for (int oi = 0; oi < io.n_output; oi++) {
        const rknn_tensor_attr& a = out_attr[oi];
        const bool nhwc = a.fmt == RKNN_TENSOR_NHWC;
        // 通道数：NCHW->dims[1]，NHWC->dims[3]；平面 HW
        const int C = nhwc ? a.dims[3] : a.dims[1];
        const int H = nhwc ? a.dims[1] : a.dims[2];
        const int W = nhwc ? a.dims[2] : a.dims[3];
        const int plane = H * W;
        if (C <= 0 || plane <= 0) { printf("OUT[%d] skip (C=%d H=%d W=%d)\n", oi, C, H, W); continue; }
        const float* p = static_cast<const float*>(out[oi].buf);

        float gmin = 1e9f, gmax = -1e9f;
        // 只打印部分通道（head 通常 C<=144；若超大则抽样打印 + 全量全局 min/max）
        const int show = (C <= 160) ? C : 160;
        printf("OUT[%d] C=%d H=%d W=%d (show first %d channels):\n", oi, C, H, W, show);
        for (int c = 0; c < C; c++) {
            float mn = 1e9f, mx = -1e9f, sum = 0;
            for (int k = 0; k < plane; k++) {
                const float v = nhwc ? p[k * C + c] : p[c * plane + k];
                if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
                if (v < gmin) gmin = v; if (v > gmax) gmax = v;
            }
            if (c < show)
                printf("  ch%3d: min=%9.2f max=%9.2f mean=%8.2f\n", c, mn, mx, sum / plane);
        }
        printf("  [GLOBAL min=%9.2f max=%9.2f]\n", gmin, gmax);
    }

    rknn_outputs_release(ctx, io.n_output, out);
    rknn_destroy(ctx);
    printf("== done\n");
    return 0;
}
