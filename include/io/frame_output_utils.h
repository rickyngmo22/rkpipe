#pragma once

#include <functional>

#include <opencv2/opencv.hpp>

#include "core/pipeline_frame.h"
#include "io/output_router.h"

void clearReleasedFrame(image_buffer_t& frame);

void finalizePipelineFrameOutput(OutputRouter& output_router,
                                 int frame_index,
                                 bool output_enabled,
                                 bool overlay_on_output,
                                 PipelineFrame& frame);

void finalizeSequentialFrameOutput(OutputRouter& output_router,
                                   int frame_index,
                                   bool output_enabled,
                                   cv::Mat& frame_mat,
                                   image_buffer_t& frame_buffer,
                                   bool& use_zero_copy,
                                   const std::function<bool()>& ensure_bgr_fn);

void releaseFrameBufferIfNeeded(image_buffer_t& frame_buffer);
