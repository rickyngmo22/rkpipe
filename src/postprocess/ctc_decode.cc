#include "postprocess/ctc_decode.h"

#include <cmath>
#include <cstdio>
#include <fstream>

std::vector<std::string> loadCtcDict(const std::string& path) {
    std::vector<std::string> dict;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::fprintf(stderr, "[rk_pipe][ctc] cannot open dict: %s\n", path.c_str());
        return dict;
    }
    std::string line;
    while (std::getline(in, line)) {
        // 去掉行尾 \r（Windows 字典）与行内空白字符后面内容？——字典行就是单字符，
        // 只裁 \r，保留字符本身（可能是全角空格等）
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        dict.push_back(line);
    }
    return dict;
}

float ctcSoftmaxProb(const float* logits, int num_classes, int row, int col) {
    const float* row_ptr = logits + static_cast<size_t>(row) * num_classes;
    float max_v = row_ptr[0];
    for (int c = 1; c < num_classes; ++c) {
        if (row_ptr[c] > max_v) {
            max_v = row_ptr[c];
        }
    }
    double sum = 0.0;
    for (int c = 0; c < num_classes; ++c) {
        sum += std::exp(static_cast<double>(row_ptr[c] - max_v));
    }
    return static_cast<float>(std::exp(static_cast<double>(row_ptr[col] - max_v)) / sum);
}

CtcLine ctcGreedyDecode(const float* logits, int time_steps, int num_classes, int blank_index,
                        const std::vector<std::string>& dict, bool logits_are_probs) {
    CtcLine out;
    if (!logits || time_steps <= 0 || num_classes <= 1) {
        return out;
    }
    if (blank_index < 0) {
        blank_index = 0;
    }
    if (blank_index >= num_classes) {
        return out;
    }

    int prev = -1;
    double score_sum = 0.0;
    int score_count = 0;
    for (int t = 0; t < time_steps; ++t) {
        const float* row_ptr = logits + static_cast<size_t>(t) * num_classes;
        int best = 0;
        float best_v = row_ptr[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row_ptr[c] > best_v) {
                best_v = row_ptr[c];
                best = c;
            }
        }
        if (best == blank_index || best == prev) {
            prev = best;
            continue;
        }
        prev = best;
        // 已是概率 → 原值即分数；raw logits → softmax 归一
        score_sum += logits_are_probs ? best_v : ctcSoftmaxProb(logits, num_classes, t, best);
        ++score_count;
        // 字典映射：1..dict.size() → dict[idx-1]；越界（use_space_char 追加的空格等）→ 空格
        if (best >= 1 && best <= static_cast<int>(dict.size())) {
            out.text += dict[best - 1];
        } else {
            out.text += " ";
        }
    }
    out.score = score_count > 0 ? static_cast<float>(score_sum / score_count) : 0.0f;
    return out;
}

RecLayout resolveRecLayout(int dim0, int dim1, int dict_size) {
    RecLayout layout;
    const bool dict_valid = dict_size > 0;
    auto in_class_range = [dict_size](int v) {
        return v >= dict_size && v <= dict_size + 2;
    };
    if (dict_valid && !in_class_range(dim1) && in_class_range(dim0)) {
        // 末维装不下类数、次末维装得下 → [B,C,T]（如 LPRNet [1,68,18]）
        layout.time_steps = dim1;
        layout.num_classes = dim0;
        layout.channel_major = true;
    } else {
        // [B,T,C]（如 PP-OCRv4 Rec [1,40,6625]）；无字典信息时默认此布局
        layout.time_steps = dim0;
        layout.num_classes = dim1;
        layout.channel_major = false;
    }
    return layout;
}
