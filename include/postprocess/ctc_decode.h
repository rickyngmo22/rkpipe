#pragma once

// CTC（Connectionist Temporal Classification）贪心解码（M6 OCR Rec / M7 LPRNet 共用）：
//   逐时间步取最大概率类别 → 跳过 blank → 折叠连续重复 → 按字典映射为文本。
//
// 类别索引约定（PaddleOCR CTCLabelDecode / LPRNet 一致）：
//   0 = blank，1..dict.size() = 字典第 i 行，C-1（或 > dict.size()）= 空格（use_space_char）。
// 分数 = 被采纳字符步的概率均值；输出已是概率时取原值，raw logits 时先 softmax（auto）。
//
// 纯逻辑，无硬件依赖，CI 单测可覆盖。

#include <string>
#include <vector>

// 字典：每行一个字符（UTF-8），加载后 dict[i] 对应类别索引 i+1（0 留给 blank）
std::vector<std::string> loadCtcDict(const std::string& path);

struct CtcLine {
    std::string text;
    float score = 0.0f;
};

// logits: [T, C] 行主序；blank_index < 0 时取 0；
// logits_are_probs=true 时输入已是概率（如 PP-OCRv4 Rec 图内带 Softmax），分数直接取原值
CtcLine ctcGreedyDecode(const float* logits, int time_steps, int num_classes, int blank_index,
                        const std::vector<std::string>& dict, bool logits_are_probs = false);

// softmax 第 row 行第 col 列的概率（数值稳定版）；ctcGreedyDecode 内部用
float ctcSoftmaxProb(const float* logits, int num_classes, int row, int col);

// rec 输出布局识别（交付说明：rec=[1,40,6625] 为 [B,T,C]，LPRNet=[1,68,18] 为 [B,C,T]）。
// 用字典尺寸区分：CTC 类数 C 恒在 [dict_size, dict_size+2] 区间
//   （dict 含 blank：C=dict_size；blank+空格在字典外：C=dict_size+2）
struct RecLayout {
    int time_steps = 0;
    int num_classes = 0;
    bool channel_major = false;  // true = [B,C,T]（读出需转置）
};
RecLayout resolveRecLayout(int dim0, int dim1, int dict_size);
