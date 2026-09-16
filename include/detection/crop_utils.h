#pragma once

// 两阶级联（M0 CompositeDetector 底座）共享的裁剪几何：
//   stage1 检测框 → 按二级模型宽高比扩成正形（含外扩系数）→ 正/逆仿射系数
//   → 正向系数裁剪出二级模型输入，逆向系数把二级结果映射回原图。
//
// 从 RtmposeDetector 的 buildRtmposeAffine 抽取（MMPose TopdownAffine 的轴对齐版本），
// 模板化以同时服务 rtmpose_affine_t 与未来的 composite 级联结果类型。
// 纯数学，无硬件依赖，CI 单测可覆盖（crop_utils 往返精度）。

#include <cmath>

#include "utils/common.h"  // image_rect_t

namespace crop_utils {

// 按模型宽高比把框扩成正形（bbox_xyxy2cs 的轴对齐版本），输出正向/逆向 2x3 仿射系数。
// pad 为框外扩系数（1.0=训练分布；YOLO 框偏紧可调 1.1~1.25）。
// Aff 类型须含 float m[6]（正向）与 float im[6]（逆向）。
//   正向：crop_x = (orig_x - cx) * sx + model_w/2
//   逆向：orig_x = (crop_x - model_w/2) / sx + cx
template <typename Aff>
void buildSquareAffine(const image_rect_t& box, int model_w, int model_h, float pad, Aff* aff) {
    float w = static_cast<float>(box.right - box.left) * pad;
    float h = static_cast<float>(box.bottom - box.top) * pad;
    const float cx = (box.left + box.right) * 0.5f;
    const float cy = (box.top + box.bottom) * 0.5f;

    const float aspect = static_cast<float>(model_w) / static_cast<float>(model_h);
    if (w > h * aspect) {
        h = w / aspect;
    } else {
        w = h * aspect;
    }
    w = w > 1.0f ? w : 1.0f;
    h = h > 1.0f ? h : 1.0f;

    const float sx = static_cast<float>(model_w) / w;
    const float sy = static_cast<float>(model_h) / h;

    aff->m[0] = sx;
    aff->m[1] = 0.0f;
    aff->m[2] = model_w * 0.5f - sx * cx;
    aff->m[3] = 0.0f;
    aff->m[4] = sy;
    aff->m[5] = model_h * 0.5f - sy * cy;

    aff->im[0] = 1.0f / sx;
    aff->im[1] = 0.0f;
    aff->im[2] = cx - model_w * 0.5f / sx;
    aff->im[3] = 0.0f;
    aff->im[4] = 1.0f / sy;
    aff->im[5] = cy - model_h * 0.5f / sy;
}

// 二级模型坐标系点 → 原图坐标（关键点/文本行角点等回映射）
template <typename Aff>
void mapPointBack(const Aff& aff, float crop_x, float crop_y, float* orig_x, float* orig_y) {
    *orig_x = aff.im[0] * crop_x + aff.im[1] * crop_y + aff.im[2];
    *orig_y = aff.im[3] * crop_x + aff.im[4] * crop_y + aff.im[5];
}

}  // namespace crop_utils
