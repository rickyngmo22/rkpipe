#pragma once

#include <atomic>

#include <opencv2/opencv.hpp>

#include "../core/pipeline_frame.h"
#include "input_source.h"

bool readNextPipelineFrame(InputSource* input_source,
                           PipelineFrame& frame,
                           int frame_index,
                           std::atomic<int>* read_count = nullptr);

bool readNextFrame(InputSource* input_source,
                   cv::Mat& frame_mat,
                   image_buffer_t& frame_buffer,
                   bool& is_zero_copy);

// 释放流水线帧持有的缓冲（NV12 dma-buf + matFrame），供自适应跳帧丢弃帧时使用
void releasePipelineFrame(PipelineFrame& frame);

bool ensureBgrFrame(PipelineFrame& frame);

bool ensureBgrFrame(cv::Mat& frame_mat,
                    const image_buffer_t& frame_buffer,
                    bool is_zero_copy);
