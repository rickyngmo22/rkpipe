#pragma once

// D3 3D 线框（路线 B）：由"2D 检测框 + depth 模型接地深度 + P2"几何反推 3D 包围盒，
// 复用 Detect3DItem 与 computeDetect3DCorners2D 投影（线框绘制零新增数学）。
//
// 与 YOLO26-Detect3D（KITTI 3D 头）的区别：深度不来自 3D 头回归（跨场景泛化差，
// 见 docs/demo_detect3d.md），而来自 aux depth 模型在框底接地带的采样（米制 dense depth）；
// 尺寸由几何反推而非模型回归。纯函数无硬件依赖，CI 单测可直接覆盖。
//
// 几何约定（侧视先验，alpha_deg=0）：
//   - 接地深度 Z：路侧车辆侧视时框底 ≈ 车轮接地，深度最可靠（车窗/背景污染最小）；
//   - h = ph·Z/fy（像素高 × 深度 / 焦距）；
//   - l = pw·Z/fx（侧视时像素宽 ≈ 车长）；
//   - w = l × kD3WidthRatio（宽长比先验，KITTI Car 1.618/3.892 ≈ 0.416）；
//   - 朝向：alpha = alpha_deg（先验，路侧车沿路停放/行驶 → 相对朝向 ≈ 0）。
//     computeDetect3DCorners2D 内部 ry = alpha + theta（theta=atan2(Z,X)），
//     alpha=0 时车身垂直于视线 = 侧视，与上述 l/w 反推自洽。
//   - 对向/同向车（正视车尾/车头）用 alpha_deg=90：此时像素宽 ≈ 车宽，
//     l 的反推不成立（退化为先验比），距离与高度仍然正确。
//   - 3D 中心取 2D 框中心（computeDetect3DCorners2D 的 center 为 3D 包围盒半高几何中心）。
// 注意：Z 为接地深度，中心深度略有差异（俯仰角余弦因子），小俯仰下 <2%，忽略。

#include <cmath>

#include "postprocess/detect3d_decode.h"

// 宽长比先验（w = l × ratio）：KITTI Car 3D 尺寸先验 w/l = 1.618807/3.892148
static const float kD3WidthRatio = 0.4158f;

// 由 2D 框 + 接地深度构造 Detect3DItem（仅含线框投影所需字段；conf/cls_id 由调用方按需填）。
// p2 用于取 fx/fy/cx/cy；ground_z 必须为正。返回 false = 参数无效。
inline bool buildDetect3DItemFromDepth(const image_rect_t& box, float ground_z,
                                       const float* p2, float alpha_deg, Detect3DItem* out) {
    if (out == nullptr || p2 == nullptr || !(ground_z > 0.0f)) {
        return false;
    }
    const float fx = p2[0], fy = p2[5];
    if (!(fx > 0.0f && fy > 0.0f)) {
        return false;
    }
    const float pw = static_cast<float>(box.right - box.left);
    const float ph = static_cast<float>(box.bottom - box.top);
    if (!(pw > 1.0f && ph > 1.0f)) {
        return false;
    }
    out->box = box;
    out->depth_m = ground_z;
    out->center_u = static_cast<float>(box.left + box.right) * 0.5f;
    out->center_v = static_cast<float>(box.top + box.bottom) * 0.5f;
    out->h3 = ph * ground_z / fy;
    out->l3 = pw * ground_z / fx;
    out->w3 = out->l3 * kD3WidthRatio;
    const float alpha = alpha_deg * static_cast<float>(M_PI) / 180.0f;
    out->sin_alpha = std::sin(alpha);
    out->cos_alpha = std::cos(alpha);
    return true;
}
