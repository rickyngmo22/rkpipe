#include "postprocess/cls_top1.h"

#include <cmath>

ClsTop1 clsTop1(const float* values, int num_classes, bool input_is_prob) {
    ClsTop1 out;
    if (!values || num_classes <= 0) {
        return out;
    }
    int best = 0;
    float best_v = values[0];
    for (int c = 1; c < num_classes; ++c) {
        if (values[c] > best_v) {
            best_v = values[c];
            best = c;
        }
    }
    out.cls_id = best;
    if (input_is_prob) {
        out.score = best_v;
        return out;
    }
    // 数值稳定 softmax（只在输出分数时算，argmax 与单调变换无关）
    double sum = 0.0;
    for (int c = 0; c < num_classes; ++c) {
        sum += std::exp(static_cast<double>(values[c] - best_v));
    }
    out.score = static_cast<float>(1.0 / sum);
    return out;
}
