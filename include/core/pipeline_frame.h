#pragma once

// 流水线帧数据契约：输入帧（Mat 或零拷贝 NV12 dma-buf）+ 该帧的任务结果。
// 这是输入/检测/跟踪/输出各模块共享的数据面；线程模型与调度策略不在本头文件。
//
// 零拷贝约定：isZeroCopy=true 时帧数据在 bufferFrame（NV12 dma-buf，fd/virt_addr），
// matFrame 为空；非零拷贝路径帧数据在 matFrame（BGR）。输出侧用 empty() 判定本帧是否有效。

#include <string>

#include <opencv2/opencv.hpp>

#include "utils/common.h"
#include "core/task_result.h"

struct PipelineFrame {
    int index = 0;
    cv::Mat matFrame;
    image_buffer_t bufferFrame = {};
    bool isZeroCopy = false;
    bool hasResult = false;
    TaskResult result;
    // depth 辅助任务结果（CV_8UC1 有效区低分辨率），由输出线程放大 + 伪彩，
    // 避免每个 worker 每帧做 1080p JET + addWeighted 全帧混合
    cv::Mat auxDepth;
    cv::Rect auxDepthRoi = cv::Rect(0, 0, 0, 0);
    float auxDepthLo = 0.0f;  // 深度 8bit 反相图归一化范围（米），距离文字还原用
    float auxDepthHi = 0.0f;
    // 本帧源名（图片目录输入 = 文件路径），检测结果导出（--dump-detections）映射 image_id 用
    std::string sourceName;

    bool empty() const {
        if (isZeroCopy) return bufferFrame.virt_addr == nullptr && bufferFrame.fd <= 0;
        return matFrame.empty();
    }
};
