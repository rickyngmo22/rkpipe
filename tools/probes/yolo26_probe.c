// yolo26_probe.c - 统一探针：验证 yolo26 detect 模型（非 end2end 3x[1,84,H,W] 或 end2end [1,300,6]）
// 非 end2end: box 直接距离(reg_max=1) + cls logits(sigmoid) + topk 免 NMS
// end2end:    [1,300,6] 已含 TopK 结果，打印列统计 + 原始行便于判读
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

struct Det { float x1,y1,x2,y2,score; int cls; };

// ---- 非 end2end：3 x [1, 4+nc, H, W] ----
static void decode_three_scales(rknn_context ctx, rknn_output* out, rknn_tensor_attr* attrs,
                                int n_out, float conf, const cv::Mat& img,
                                float scale, float pad_x, float pad_y,
                                const std::vector<std::string>& labels) {
    std::vector<Det> dets;
    for (int oi = 0; oi < n_out; oi++) {
        int C = attrs[oi].dims[1], H = attrs[oi].dims[2], W = attrs[oi].dims[3];
        if (C < 5 || H <= 0 || W <= 0) continue;
        int stride = MODEL_SIZE / H;
        int grid_len = H * W;
        float* p = (float*)out[oi].buf;
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
            int off = i*W + j;
            float best=-1e9f; int bestc=-1;
            for (int c = 4; c < C; c++) { float s = sigmoid(p[c*grid_len+off]); if (s>best){best=s;bestc=c-4;} }
            if (best < conf) continue;
            float l=p[0*grid_len+off], t=p[1*grid_len+off], r=p[2*grid_len+off], b=p[3*grid_len+off];
            Det d;
            d.x1=(j+0.5f-l)*stride; d.y1=(i+0.5f-t)*stride; d.x2=(j+0.5f+r)*stride; d.y2=(i+0.5f+b)*stride;
            d.score=best; d.cls=bestc;
            dets.push_back(d);
        }
    }
    std::sort(dets.begin(), dets.end(), [](const Det& a,const Det& b){return a.score>b.score;});
    if (dets.size()>100) dets.resize(100);
    printf("== detections (>=%.2f): %zu\n", conf, dets.size());
    int shown=0;
    for (auto& d : dets) {
        float x1=(d.x1-pad_x)/scale, y1=(d.y1-pad_y)/scale, x2=(d.x2-pad_x)/scale, y2=(d.y2-pad_y)/scale;
        if (x1<0)x1=0; if(y1<0)y1=0; if(x2>img.cols)x2=img.cols; if(y2>img.rows)y2=img.rows;
        if (shown<20) {
            const char* name=(d.cls>=0&&d.cls<(int)labels.size())?labels[d.cls].c_str():"?";
            printf("  %-16s score=%.3f box=(%.0f,%.0f,%.0f,%.0f) wh640=(%.0f,%.0f)\n",
                   name, d.score, x1,y1,x2,y2, d.x2-d.x1, d.y2-d.y1);
        }
        shown++;
    }
}

// ---- end2end：单输出 [1, 300, 6]，打印列统计 + top rows（列序以实测为准）----
static void dump_end2end(rknn_output* out, rknn_tensor_attr& attr, const std::vector<std::string>& labels) {
    float* p = (float*)out->buf;
    int n = attr.n_elems;
    int last_dim = (attr.n_dims == 3) ? attr.dims[2] : attr.dims[3];
    int rows = n / last_dim;
    printf("== end2end OUT shape=(%d,%d,%d) rows=%d last_dim=%d\n",
           attr.dims[0], attr.dims[1], attr.dims[2], rows, last_dim);
    for (int c = 0; c < last_dim; c++) {
        float mn=1e9f,mx=-1e9f,sum=0; int cnt=0;
        for (int r=0;r<rows;r++){float v=p[r*last_dim+c]; if(v>mx)mx=v; if(v<mn)mn=v; sum+=v;cnt++;}
        printf("  col%d: min=%9.3f max=%9.3f mean=%8.3f\n", c, mn, mx, cnt?sum/cnt:0);
    }
    // 假设 [x1,y1,x2,y2,cls,score]，按 col5 降序打印
    struct Row { float v[8]; float s; } best[16];
    int nb=0;
    for (int r=0;r<rows;r++) {
        float s=p[r*last_dim+5]; int slot=-1;
        for (int k=0;k<nb;k++) if (s>best[k].s){slot=k;break;}
        if (nb<16) {
            if (slot<0) slot=nb;
            for (int k=nb;k>slot;k--) best[k]=best[k-1];
            best[slot].s=s; for(int c=0;c<last_dim&&c<8;c++) best[slot].v[c]=p[r*last_dim+c];
            nb++;
        }
    }
    for (int k=0;k<nb;k++) {
        int cls=(int)best[k].v[4];
        const char* name=(cls>=0&&cls<(int)labels.size())?labels[cls].c_str():"?";
        printf("  [%s cls=%.0f score=%.3f xyxy=(%.0f,%.0f,%.0f,%.0f)]\n", name, best[k].v[4],
               best[k].s, best[k].v[0], best[k].v[1], best[k].v[2], best[k].v[3]);
    }
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/userdata/rk_pipe/model/yolo26n.rknn";
    const char* img_path   = argc > 2 ? argv[2] : "/userdata/rk_pipe/test_images/000000000139.jpg";
    float conf = argc > 3 ? atof(argv[3]) : 0.25f;
    printf("== yolo26_probe model=%s img=%s conf=%.2f\n", model_path, img_path, conf);

    std::vector<std::string> labels;
    { FILE* f=fopen("/userdata/rk_pipe/model/coco_80_labels_list.txt","r"); char line[256];
      while (f && fgets(line,sizeof(line),f)) { line[strcspn(line,"\r\n")]=0; labels.push_back(line); }
      if (f) fclose(f); }

    int msz=0; unsigned char* mdata=read_file(model_path,&msz);
    if (!mdata) { printf("read model fail\n"); return -1; }
    rknn_context ctx=0;
    int ret=rknn_init(&ctx,mdata,msz,0,NULL); free(mdata);
    if (ret<0) { printf("rknn_init fail ret=%d\n",ret); return -1; }

    rknn_sdk_version ver;
    if (rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&ver,sizeof(ver))==0)
        printf("runtime api=%s drv=%s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num io;
    if (rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io))!=0) { printf("query io fail\n"); return -1; }
    printf("in=%d out=%d\n", io.n_input, io.n_output);

    rknn_tensor_attr in_attr[8], out_attr[16];
    for (int i=0;i<io.n_input;i++){memset(&in_attr[i],0,sizeof(in_attr[i]));in_attr[i].index=i;rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&in_attr[i],sizeof(in_attr[i]));dump_attr("IN ",&in_attr[i]);}
    for (int i=0;i<io.n_output;i++){memset(&out_attr[i],0,sizeof(out_attr[i]));out_attr[i].index=i;rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&out_attr[i],sizeof(out_attr[i]));dump_attr("OUT",&out_attr[i]);}

    cv::Mat img=cv::imread(img_path,cv::IMREAD_COLOR);
    if (img.empty()) { printf("imread fail\n"); return -1; }
    cv::Mat rgb; float scale, px, py;
    letterbox(img, rgb, scale, px, py);

    rknn_input in; memset(&in,0,sizeof(in));
    in.index=0; in.type=RKNN_TENSOR_UINT8; in.fmt=RKNN_TENSOR_NHWC;
    in.size=MODEL_SIZE*MODEL_SIZE*3; in.buf=rgb.data;
    if (rknn_inputs_set(ctx,1,&in)!=0){printf("inputs_set fail\n");return -1;}
    ret=rknn_run(ctx,NULL);
    if (ret<0){printf("rknn_run fail ret=%d\n",ret);return -1;}
    printf("rknn_run OK\n");

    rknn_output out[16];
    for (int i=0;i<io.n_output;i++){memset(&out[i],0,sizeof(out[i]));out[i].index=i;out[i].want_float=1;}
    ret=rknn_outputs_get(ctx,io.n_output,out,NULL);
    if (ret<0){printf("outputs_get fail ret=%d\n",ret);return -1;}

    // 判断格式：end2end 单输出 [1,300,6] vs 非 end2end 多输出
    bool is_end2end = (io.n_output == 1 && out_attr[0].n_dims == 3 && out_attr[0].dims[2] == 6);
    if (is_end2end) {
        dump_end2end(&out[0], out_attr[0], labels);
    } else {
        decode_three_scales(ctx, out, out_attr, io.n_output, conf, img, scale, px, py, labels);
    }

    rknn_outputs_release(ctx, io.n_output, out);
    rknn_destroy(ctx);
    printf("== done\n");
    return 0;
}
