#include "io/frame_output_utils.h"

#include <utility>

#include "io/video_reader.h"
#include "utils/draw_utils.h"

void clearReleasedFrame(image_buffer_t& frame) {
    frame.priv_data = nullptr;
    frame.virt_addr = nullptr;
    frame.fd = 0;
    frame.size = 0;
    frame.width = 0;
    frame.height = 0;
    frame.width_stride = 0;
    frame.height_stride = 0;
}

void finalizePipelineFrameOutput(OutputRouter& output_router,
                                 int frame_index,
                                 bool output_enabled,
                                 bool overlay_on_output,
                                 PipelineFrame& frame) {
    // 延迟叠加路径：seg 掩膜由输出线程补画（预览降采样时），坐标按缩放比例换算
    const SegTaskResult* seg_result = frame.hasResult ? std::get_if<SegTaskResult>(&frame.result) : nullptr;
    const seg_detect_result_list* seg_overlay = (seg_result && overlay_on_output) ? &seg_result->data : nullptr;

    bool zero_copy_written = false;
    if (output_enabled && frame.isZeroCopy) {
        zero_copy_written = output_router.write(frame_index, frame.bufferFrame, seg_overlay);
    }

    if (zero_copy_written) {
        // 零拷贝视频写入时由 video writer 持有并释放 buffer；
        // 纯 web 预览场景没有接管者，需要在这里释放（AVFrame/MppFrame）。
        if (!output_router.supportsZeroCopyVideo()) {
            VideoReader::releaseFrame(frame.bufferFrame);
        }
        clearReleasedFrame(frame.bufferFrame);
        return;
    }

    cv::Mat output_frame = frame.matFrame;
    // BGR(CPU) 路径：无缩放，直接在 matFrame 上补画（scale=1.0）
    if (seg_overlay && !output_frame.empty()) {
        drawSegResultsBGR(output_frame, *seg_overlay, 1.0);
    }
    VideoReader::releaseFrame(frame.bufferFrame);
    if (output_enabled && !output_frame.empty()) {
        output_router.write(frame_index, std::move(output_frame));
    }
}

void finalizeSequentialFrameOutput(OutputRouter& output_router,
                                   int frame_index,
                                   bool output_enabled,
                                   cv::Mat& frame_mat,
                                   image_buffer_t& frame_buffer,
                                   bool& use_zero_copy,
                                   const std::function<bool()>& ensure_bgr_fn) {
    if (!output_enabled) {
        return;
    }

    bool zero_copy_written = false;
    if (use_zero_copy) {
        zero_copy_written = output_router.write(frame_index, frame_buffer);
        if (zero_copy_written) {
            // 纯 web 预览没有视频写入器接管 buffer，必须把 dma-buf 归还 reader 池，
            // 否则槽位泄漏，sequential 模式约一个池深（16）帧后读取阻塞。
            if (!output_router.supportsZeroCopyVideo()) {
                VideoReader::releaseFrame(frame_buffer);
            }
            clearReleasedFrame(frame_buffer);
        } else {
            ensure_bgr_fn();
            use_zero_copy = false;
        }
    }

    if (!zero_copy_written) {
        cv::Mat output_frame = std::move(frame_mat);
        if (!output_frame.empty()) {
            output_router.write(frame_index, std::move(output_frame));
        }
    }
}

void releaseFrameBufferIfNeeded(image_buffer_t& frame_buffer) {
    if (frame_buffer.virt_addr || frame_buffer.fd > 0 || frame_buffer.priv_data) {
        VideoReader::releaseFrame(frame_buffer);
    }
}
