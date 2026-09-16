#pragma once

// YOLO26-Detect3D（Ultralytics feat/detect3d，MonoCon 风格单目 3D 头）输出解码纯函数。
// 支持两种布局（strip_detect3d_topk.py 适配细节见脚本头注释）：
//   原版 [1,300,14]：[x1,y1,x2,y2, conf, cls, cx3d,cy3d,depth,sin,cos,h3,w3,l3]（已解码）
//   板端 [1,N,38]：原始量，解码全部在本头文件完成（图内只留真机验证过的算子）：
//     [0:4) 框 [4:7) 类别分数 [7:9) 中心偏移 [9] log深度 [10:13) 尺寸残差 [13] q3d
//     [14:26) 朝向 bin 分数 [26:38) bin 残差
// 纯函数无硬件依赖，CI 单测覆盖。

#include <algorithm>
#include <cmath>

#include "utils/common.h"

struct Detect3DItem {
    image_rect_t box = {};   // 2D 框（映射回原帧后）
    float conf = 0.0f;
    int cls_id = -1;
    float center_u = 0.0f;   // 3D 投影中心（原帧像素）
    float center_v = 0.0f;
    float depth_m = 0.0f;
    float sin_alpha = 0.0f;
    float cos_alpha = 0.0f;
    float h3 = 0.0f;         // 3D 尺寸（米）：高/宽/长
    float w3 = 0.0f;
    float l3 = 0.0f;
};

// KITTI 尺寸先验 (h,w,l)，与训练 set_dim_priors 的 known 表一致：
// 顺序按数据集 names：0=Car, 1=Pedestrian, 2=Cyclist。
// 板端导出（strip_detect3d_topk.py）的尺寸列为 exp(residual)（先验无关），
// 解码时按 argmax 类别补乘（规避 NPU 小尺寸 GEMM 列损坏）。
static const float kDetect3DDimPriors[3][3] = {
    {1.529757f, 1.618807f, 3.892148f},  // Car
    {1.760000f, 0.660000f, 0.840000f},  // Pedestrian
    {1.730000f, 0.600000f, 1.760000f},  // Cyclist
};

// 3D 线框投影：由检测项 + P2（3x4 行主序，12 值）计算包围盒 8 角点的图像坐标。
// 约定：相机系 x 右 / y 下 / z 前；center_u/v 为 3D 包围盒**几何中心（半高）**的投影像素
// （真机截图实测：按底面中心画会整体抬高约 h/2，见 2026-08-29 线框标定记录）；
// 朝向约定（2026-09-02 真机修正）：模型输出的 sin/cos 即最终航向 ry（车长方向相对
// 相机 x 轴），直接 atan2 还原，不叠加视角修正 theta——原 ry = alpha + atan2(Z,X) 会把
// 正前方车辆转横约 90°（车长投影到侧向，线框横向摊开且越偏离画面中心越歪）。
// 备选式（若模型预测的是 KITTI 观测角 α）：ry = alpha + atan2(X, Z)；画面中心附近与
// 直读几乎等价，若修后两侧远处车辆线框仍不贴合再切换验证。
// 角点序：0-3 底面（沿 +x 起），4-7 顶面；12 条边 = 底 4 + 顶 4 + 立柱 4。
// 返回 false：P2 无效 / 深度非正 / 尺寸非正 / 角点落到相机后方。
inline bool computeDetect3DCorners2D(const Detect3DItem& it, const float* p2, float out[8][2]) {
    if (p2 == nullptr || it.depth_m <= 0.0f) {
        return false;
    }
    const float fx = p2[0], fy = p2[5], cx = p2[2], cy = p2[6];
    if (!(fx > 0.0f && fy > 0.0f)) {
        return false;
    }
    const float Z = it.depth_m;
    const float X = (it.center_u - cx) * Z / fx;
    const float Y = (it.center_v - cy) * Z / fy;
    const float ry = std::atan2(it.sin_alpha, it.cos_alpha);
    const float l = it.l3, h = it.h3, w = it.w3;
    if (!(l > 0.0f && h > 0.0f && w > 0.0f)) {
        return false;
    }
    const float kXc[8] = {l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2};
    const float kYc[8] = {h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2};
    const float kZc[8] = {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2};
    const float c = std::cos(ry), s = std::sin(ry);
    for (int i = 0; i < 8; ++i) {
        const float X3 = c * kXc[i] + s * kZc[i] + X;
        const float Y3 = kYc[i] + Y;
        const float Z3 = -s * kXc[i] + c * kZc[i] + Z;
        const float pw = p2[8] * X3 + p2[9] * Y3 + p2[10] * Z3 + p2[11];
        if (!(pw > 1e-3f)) {
            return false;  // 角点在相机后方，线框不绘制
        }
        out[i][0] = (p2[0] * X3 + p2[1] * Y3 + p2[2] * Z3 + p2[3]) / pw;
        out[i][1] = (p2[4] * X3 + p2[5] * Y3 + p2[6] * Z3 + p2[7]) / pw;
    }
    return true;
}

// 38 列板端布局解码（strip_detect3d_topk.py 深裁剪导出，解码全部在 C++ 完成，
// 图内只留 Conv/Sigmoid/Mul/Concat/Transpose/Slice 等真机验证过的算子）：
//   [0:4)   2D 框 x1,y1,x2,y2（模型输入像素）
//   [4:7)   类别分数（sigmoid，已含 quality^0.5 融合）→ conf=最大值，cls=argmax
//   [7:9)   3D 中心偏移（相对 2D 框中心，×框宽高还原，框宽高 clamp_min 1.0）
//   [9]     log 深度 → exp(clamp(ln0.1, ln200))
//   [10:13) 尺寸残差 → KITTI 先验 × exp(clamp(±4))
//   [13]    q3d logit（quality 已折进类别分数，忽略）
//   [14:26) 朝向 bin 分数（12）/[26:38) bin 残差（12）
//       multibin：bin=argmax(bins)，alpha = bin×(2π/12) + tanh(res[bin])×(π/12)
inline Detect3DItem detect3dDecodeRowCols(const float* row, int cols) {
    Detect3DItem it;
    if (row == nullptr) {
        return it;
    }
    it.box.left = row[0];
    it.box.top = row[1];
    it.box.right = row[2];
    it.box.bottom = row[3];
    if (cols == 14) {
        // 原版导出（end2end topk 后）：[box4, conf, cls, 8×d3]，尺寸为最终米值
        it.conf = row[4];
        it.cls_id = static_cast<int>(row[5] + 0.5f);
        it.center_u = row[6];
        it.center_v = row[7];
        it.depth_m = row[8];
        it.sin_alpha = row[9];
        it.cos_alpha = row[10];
        it.h3 = row[11];
        it.w3 = row[12];
        it.l3 = row[13];
        return it;
    }
    if (cols != 38) {
        return it;
    }
    // 类别分数（已含 quality 融合）
    int best = 0;
    for (int c = 1; c < 3; ++c) {
        if (row[4 + c] > row[4 + best]) {
            best = c;
        }
    }
    it.conf = row[4 + best];
    it.cls_id = best;
    // 3D 中心 = 框中心 + 偏移 × 框宽高（clamp_min 1.0，同导出侧 _inference）
    const float bcx = (row[0] + row[2]) * 0.5f;
    const float bcy = (row[1] + row[3]) * 0.5f;
    const float bw = std::max(row[2] - row[0], 1.0f);
    const float bh = std::max(row[3] - row[1], 1.0f);
    it.center_u = bcx + row[7] * bw;
    it.center_v = bcy + row[8] * bh;
    // 深度：exp(clamp(log 深度, ln0.1, ln200))
    it.depth_m = std::exp(std::clamp(row[9], std::log(0.1f), std::log(200.0f)));
    // 尺寸：KITTI 先验 × exp(clamp(残差, ±4))
    const int pr = (it.cls_id >= 0 && it.cls_id < 3) ? it.cls_id : 0;
    it.h3 = kDetect3DDimPriors[pr][0] * std::exp(std::clamp(row[10], -4.0f, 4.0f));
    it.w3 = kDetect3DDimPriors[pr][1] * std::exp(std::clamp(row[11], -4.0f, 4.0f));
    it.l3 = kDetect3DDimPriors[pr][2] * std::exp(std::clamp(row[12], -4.0f, 4.0f));
    // multibin 朝向（sin/cos 对角度 wrap 不敏感，无需归一化）
    int bin = 0;
    for (int b = 1; b < 12; ++b) {
        if (row[14 + b] > row[14 + bin]) {
            bin = b;
        }
    }
    constexpr float kBinSize = 2.0f * static_cast<float>(M_PI) / 12.0f;
    const float residual = std::tanh(row[26 + bin]) * (kBinSize * 0.5f);
    const float alpha = static_cast<float>(bin) * kBinSize + residual;
    it.sin_alpha = std::sin(alpha);
    it.cos_alpha = std::cos(alpha);
    return it;
}

// 一行 14 列 float → 结构化结果（不做 letterbox 逆映射）
inline Detect3DItem detect3dDecodeRow(const float* row) {
    return detect3dDecodeRowCols(row, 14);
}

// letterbox 逆映射：模型输入坐标 → 原帧坐标（与 detect 后处理同约定：
// orig = (model - pad) / scale + crop），并裁剪到原帧范围
inline void detect3dMapToFrame(Detect3DItem& it, const letterbox_t* lb,
                               int model_w, int model_h, int frame_w, int frame_h) {
    if (lb == nullptr || lb->scale <= 0.0f) {
        // 无 letterbox：模型坐标即原帧坐标
        it.box.left = std::max(0, std::min(frame_w - 1, static_cast<int>(it.box.left)));
        it.box.top = std::max(0, std::min(frame_h - 1, static_cast<int>(it.box.top)));
        it.box.right = std::max(0, std::min(frame_w - 1, static_cast<int>(it.box.right)));
        it.box.bottom = std::max(0, std::min(frame_h - 1, static_cast<int>(it.box.bottom)));
        it.center_u = std::max(0.0f, std::min(static_cast<float>(frame_w - 1), it.center_u));
        it.center_v = std::max(0.0f, std::min(static_cast<float>(frame_h - 1), it.center_v));
        return;
    }
    const int crop_left = lb->crop_x;
    const int crop_top = lb->crop_y;
    // 约定与 detect 后处理一致：orig = clamp(model - pad) / scale + crop
    auto map_x = [&](float v) {
        const float c = std::clamp(v - lb->x_pad, 0.0f, static_cast<float>(model_w));
        return static_cast<int>(c / lb->scale) + crop_left;
    };
    auto map_y = [&](float v) {
        const float c = std::clamp(v - lb->y_pad, 0.0f, static_cast<float>(model_h));
        return static_cast<int>(c / lb->scale) + crop_top;
    };
    it.box.left = std::max(0, std::min(frame_w - 1, map_x(it.box.left)));
    it.box.top = std::max(0, std::min(frame_h - 1, map_y(it.box.top)));
    it.box.right = std::max(0, std::min(frame_w - 1, map_x(it.box.right)));
    it.box.bottom = std::max(0, std::min(frame_h - 1, map_y(it.box.bottom)));
    it.center_u = std::max(0.0f, std::min(static_cast<float>(frame_w - 1),
                                          static_cast<float>(map_x(it.center_u))));
    it.center_v = std::max(0.0f, std::min(static_cast<float>(frame_h - 1),
                                          static_cast<float>(map_y(it.center_v))));
}
