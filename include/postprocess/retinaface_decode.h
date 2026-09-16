#pragma once

// RetinaFace（M8，RKNN Model Zoo RetinaFace_mobile320 对接）解码纯函数：
//   PriorBox 生成 4200 anchor（steps 8/16/32，每位置 2 个 min_size）→
//   center-form 解码（variances [0.1,0.2]）→ 分数截断 → IoU NMS。
//   landmark 每点独立 center-form（variance 0.1）。
//
// 输出布局（交付说明）：box(1,4200,4) / cls(1,4200,2) 图内已 Softmax / landm(1,4200,10)。
// 坐标均为 320×320 模型坐标系归一化值；调用方负责 letterbox 逆映射回原图。
//
// 纯逻辑，无硬件依赖，CI 单测可覆盖（anchor 数量/首尾锚点值/解码闭合）。

#include <vector>

struct RetinaFaceItem {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;  // 模型坐标系像素
    float score = 0.0f;
    float landmarks[10] = {0.0f};  // 5 点 (x,y)，模型坐标系像素
};

// 生成先验框（320×320 mobile 配置：[40,20,10] 网格 × [16,32]/[64,128]/[256,512]）。
// 返回平铺 [4200][4]（cx, cy, w, h，归一化）。枚举顺序必须与模型输出一致：
// 外层行 i、内层列 j（zoo PriorBox: product(range(h), range(w))）
std::vector<float> retinafacePriorBoxes(int model_size = 320);

struct RetinaFaceRawHeads {
    const float* box = nullptr;     // [4200*4]
    const float* cls = nullptr;     // [4200*2]，已 Softmax（取列 1）
    const float* landm = nullptr;   // [4200*10]
    int num_anchors = 0;
};

// 解码：conf 阈值过滤 → 分数降序截断 max_before_nms → IoU NMS。
// 输出坐标为模型坐标系像素（乘 model_size）
std::vector<RetinaFaceItem> retinafaceDecode(const RetinaFaceRawHeads& heads,
                                             float conf_threshold,
                                             float nms_threshold,
                                             int model_size = 320,
                                             int max_before_nms = 256,
                                             int max_results = 128);
