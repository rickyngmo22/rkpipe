#pragma once

// 二级分类（M0 CompositeDetector）解码纯函数：argmax top-1 + softmax 分数。
// 纯逻辑，无硬件依赖，CI 单测可覆盖。

// input_is_prob=true 时输入已是概率（分数取原值）；false = raw logits（softmax 归一）
struct ClsTop1 {
    int cls_id = -1;
    float score = 0.0f;
};

ClsTop1 clsTop1(const float* values, int num_classes, bool input_is_prob);
