#include "postprocess/retinaface_decode.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kVarCenter = 0.1f;  // box/landm 中心项方差
constexpr float kVarSize = 0.2f;    // box 尺寸项方差
constexpr int kSteps[3] = {8, 16, 32};
constexpr int kMinSizes[3][2] = {{16, 32}, {64, 128}, {256, 512}};

// IoU（axis-aligned）;a/b 为 x1,y1,x2,y2
float iouXYXY(const RetinaFaceItem& a, const RetinaFaceItem& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

}  // namespace

std::vector<float> retinafacePriorBoxes(int model_size) {
    std::vector<float> priors;
    if (model_size <= 0) {
        return priors;
    }
    const float fs = static_cast<float>(model_size);
    // 预估容量（320 → 4200,640 → 16800）
    size_t total = 0;
    for (const int step : kSteps) {
        const int grid = (model_size + step - 1) / step;  // ceil,与 zoo 一致
        total += static_cast<size_t>(grid) * grid * 2;
    }
    priors.reserve(total * 4);
    for (int layer = 0; layer < 3; ++layer) {
        const int grid = (model_size + kSteps[layer] - 1) / kSteps[layer];
        for (int i = 0; i < grid; ++i) {          // 行（外层,与 zoo product 一致）
            for (int j = 0; j < grid; ++j) {      // 列
                for (int m = 0; m < 2; ++m) {     // 每位置 2 个 min_size（最内层）
                    const float cx = (j + 0.5f) * kSteps[layer] / fs;
                    const float cy = (i + 0.5f) * kSteps[layer] / fs;
                    const float pw = static_cast<float>(kMinSizes[layer][m]) / fs;
                    priors.push_back(cx);
                    priors.push_back(cy);
                    priors.push_back(pw);
                    priors.push_back(pw);
                }
            }
        }
    }
    return priors;
}

std::vector<RetinaFaceItem> retinafaceDecode(const RetinaFaceRawHeads& heads,
                                             float conf_threshold,
                                             float nms_threshold,
                                             int model_size,
                                             int max_before_nms,
                                             int max_results) {
    std::vector<RetinaFaceItem> out;
    if (!heads.box || !heads.cls || !heads.landm || heads.num_anchors <= 0) {
        return out;
    }
    const int n = heads.num_anchors;
    const float fs = static_cast<float>(model_size);

    const std::vector<float> priors = retinafacePriorBoxes(model_size);
    if (static_cast<int>(priors.size() / 4) < n) {
        return out;  // 先验框数量与模型头对不上,拒绝解码（防越界读）
    }

    // 阈值过滤 + 截断（OBB 教训:先截断再 NMS 防 O(n²) 算爆）
    std::vector<int> idx;
    idx.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (heads.cls[i * 2 + 1] >= conf_threshold) {
            idx.push_back(i);
        }
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return heads.cls[a * 2 + 1] > heads.cls[b * 2 + 1];
    });
    if (static_cast<int>(idx.size()) > max_before_nms) {
        idx.resize(max_before_nms);
    }

    out.reserve(idx.size());
    for (const int i : idx) {
        const float* pb = &priors[static_cast<size_t>(i) * 4];
        const float* loc = &heads.box[static_cast<size_t>(i) * 4];
        // center-form → corner-form（归一化）→ 模型坐标像素
        const float cx = pb[0] + kVarCenter * pb[2] * loc[0];
        const float cy = pb[1] + kVarCenter * pb[3] * loc[1];
        const float w = pb[2] * std::exp(kVarSize * loc[2]);
        const float h = pb[3] * std::exp(kVarSize * loc[3]);
        RetinaFaceItem item;
        item.x1 = (cx - w * 0.5f) * fs;
        item.y1 = (cy - h * 0.5f) * fs;
        item.x2 = (cx + w * 0.5f) * fs;
        item.y2 = (cy + h * 0.5f) * fs;
        item.score = heads.cls[i * 2 + 1];
        if (!std::isfinite(item.x1) || !std::isfinite(item.y1) ||
            !std::isfinite(item.x2) || !std::isfinite(item.y2)) {
            continue;  // 回归值病态（exp 溢出等）,丢弃防污染 NMS
        }
        for (int k = 0; k < 5; ++k) {
            item.landmarks[k * 2] =
                (pb[0] + kVarCenter * pb[2] * heads.landm[static_cast<size_t>(i) * 10 + k * 2]) * fs;
            item.landmarks[k * 2 + 1] =
                (pb[1] + kVarCenter * pb[3] * heads.landm[static_cast<size_t>(i) * 10 + k * 2 + 1]) * fs;
        }
        out.push_back(item);
    }

    // IoU NMS（分数已降序）
    std::vector<RetinaFaceItem> kept;
    kept.reserve(out.size());
    for (const auto& item : out) {
        bool suppressed = false;
        for (const auto& k : kept) {
            if (iouXYXY(k, item) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(item);
            if (static_cast<int>(kept.size()) >= max_results) {
                break;
            }
        }
    }
    return kept;
}
