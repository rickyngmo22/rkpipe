#pragma once

// RetinaFace（M8 人脸检测,RKNN Model Zoo RetinaFace_mobile320 对接）解码纯函数:
//   PriorBox 生成（320×320 为 4200 anchor:steps 8/16/32 → 网格 40/20/10 × 每位置
//   2 个 min_size [16,32]/[64,128]/[256,512],枚举顺序与 zoo 一致:层主序、行外列内、
//   min_size 最内,已逐项对照 zoo examples/RetinaFace/cpp/rknn_box_priors.h）→
//   center-form 解码（box variances [0.1,0.2]）→ landmark 解码（variance 0.1,
//   5 点独立 center-form）→ 分数降序截断 → IoU NMS。
//
// 输出布局（模型交付说明）:box(1,4200,4) / cls(1,4200,2,图内已 Softmax) /
// landm(1,4200,10),均为归一化模型坐标系;解码输出乘回模型边长成像素坐标,
// letterbox 逆映射回原图由调用方负责。
//
// 纯逻辑、无硬件依赖,进 CI 纯逻辑源集单测可覆盖（anchor 数量/首尾锚点值/解码闭合/NMS）。

#include <vector>

struct RetinaFaceItem {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;  // 模型坐标系像素
    float score = 0.0f;
    float landmarks[10] = {0.0f};  // 5 点 (x,y) 交错,模型坐标系像素
};

// 生成先验框,平铺 [N][4]（cx, cy, w, h,归一化）。320×320 → N=4200,640×640 → N=16800。
// 枚举顺序必须与模型输出一致（zoo PriorBox: product(range(h), range(w)) × min_sizes）
std::vector<float> retinafacePriorBoxes(int model_size = 320);

struct RetinaFaceRawHeads {
    const float* box = nullptr;    // [num_anchors*4] 原始回归值
    const float* cls = nullptr;    // [num_anchors*2] 图内已 Softmax（背景列 0,人脸列 1）
    const float* landm = nullptr;  // [num_anchors*10] 原始回归值
    int num_anchors = 0;
};

// 解码:conf 阈值过滤（取 cls 第 1 列）→ 分数降序截断 max_before_nms → center-form
// 解码 → IoU NMS。输出坐标为模型坐标系像素（乘 model_size）。
// 指针缺失 / num_anchors<=0 / 先验框数量对不上 → 返回空。
std::vector<RetinaFaceItem> retinafaceDecode(const RetinaFaceRawHeads& heads,
                                             float conf_threshold,
                                             float nms_threshold,
                                             int model_size = 320,
                                             int max_before_nms = 256,
                                             int max_results = 128);
