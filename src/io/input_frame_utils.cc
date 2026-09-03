#include "../../include/io/input_frame_utils.h"

#include <cstddef>

#include "../../include/io/video_reader.h"

namespace {

bool debugYuvEnabled() {
    const char* value = std::getenv("RK_PIPE_DEBUG_YUV");
    return value && *value && std::string(value) != "0";
}

bool cloneBgrBuffer(const image_buffer_t& frame_buffer, cv::Mat& frame_mat) {
    int stride = frame_buffer.width_stride > 0 ? frame_buffer.width_stride : frame_buffer.width * 3;
    cv::Mat bgr(frame_buffer.height, frame_buffer.width, CV_8UC3, frame_buffer.virt_addr, stride);
    frame_mat = bgr.clone();
    return !frame_mat.empty();
}

}  // namespace

bool readNextPipelineFrame(InputSource* input_source,
                           PipelineFrame& frame,
                           int frame_index,
                           std::atomic<int>* read_count) {
    if (!input_source) {
        return false;
    }

    frame.matFrame.release();
    frame.bufferFrame = {};
    frame.isZeroCopy = false;

    if (input_source->read(frame.bufferFrame)) {
        if (frame.bufferFrame.format == IMAGE_FORMAT_BGR888 || frame.bufferFrame.fd <= 0) {
            cloneBgrBuffer(frame.bufferFrame, frame.matFrame);
            VideoReader::releaseFrame(frame.bufferFrame);
            frame.isZeroCopy = false;
        } else {
            frame.isZeroCopy = true;
        }
        frame.index = frame_index;
        frame.sourceName = input_source->currentSourceName();
        if (read_count) {
            read_count->fetch_add(1);
        }
        return true;
    }

    if (input_source->read(frame.matFrame)) {
        frame.isZeroCopy = false;
        frame.index = frame_index;
        frame.sourceName = input_source->currentSourceName();
        if (read_count) {
            read_count->fetch_add(1);
        }
        return true;
    }

    return false;
}

bool readNextFrame(InputSource* input_source,
                   cv::Mat& frame_mat,
                   image_buffer_t& frame_buffer,
                   bool& is_zero_copy) {
    if (!input_source) {
        return false;
    }

    frame_mat.release();
    frame_buffer = {};
    is_zero_copy = false;

    if (input_source->read(frame_buffer)) {
        if (frame_buffer.format == IMAGE_FORMAT_BGR888 || frame_buffer.fd <= 0) {
            cloneBgrBuffer(frame_buffer, frame_mat);
            VideoReader::releaseFrame(frame_buffer);
            is_zero_copy = false;
        } else {
            is_zero_copy = true;
        }
        return true;
    }

    if (input_source->read(frame_mat)) {
        is_zero_copy = false;
        return true;
    }

    return false;
}

bool ensureBgrFrame(PipelineFrame& frame) {
    if (!frame.matFrame.empty() || !frame.isZeroCopy) {
        return !frame.matFrame.empty() || !frame.empty();
    }

    // #region debug-point A:zero-copy-bgr-convert
    size_t step = frame.bufferFrame.width_stride > 0 ? frame.bufferFrame.width_stride : frame.bufferFrame.width;
    int hs = frame.bufferFrame.height_stride > 0 ? frame.bufferFrame.height_stride : frame.bufferFrame.height;
    cv::Mat nv12(hs * 3 / 2, frame.bufferFrame.width, CV_8UC1, frame.bufferFrame.virt_addr, step);
    cv::Mat bgr_full;
    int cvt_code = frame.bufferFrame.format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
    if (debugYuvEnabled()) {
        std::fprintf(stderr,
                     "[DEBUG][YUV][A] ensureBgrFrame pipeline idx=%d fmt=%d w=%d h=%d ws=%d hs=%d step=%zu cvt=%s\n",
                     frame.index,
                     static_cast<int>(frame.bufferFrame.format),
                     frame.bufferFrame.width,
                     frame.bufferFrame.height,
                     frame.bufferFrame.width_stride,
                     frame.bufferFrame.height_stride,
                     step,
                     cvt_code == cv::COLOR_YUV2BGR_NV21 ? "NV21" : "NV12");
    }
    cv::cvtColor(nv12, bgr_full, cvt_code);

    if (hs > frame.bufferFrame.height) {
        frame.matFrame = bgr_full(cv::Rect(0, 0, frame.bufferFrame.width, frame.bufferFrame.height)).clone();
    } else {
        frame.matFrame = bgr_full;
    }
    // #endregion
    return !frame.matFrame.empty();
}

void releasePipelineFrame(PipelineFrame& frame) {
    if (frame.bufferFrame.fd > 0) {
        VideoReader::releaseFrame(frame.bufferFrame);
    }
    frame.bufferFrame = {};
    frame.matFrame.release();
    frame.isZeroCopy = false;
    frame.hasResult = false;
    frame.result = TaskResult{};
}

bool ensureBgrFrame(cv::Mat& frame_mat,
                    const image_buffer_t& frame_buffer,
                    bool is_zero_copy) {
    if (!frame_mat.empty() || !is_zero_copy) {
        return !frame_mat.empty() || !is_zero_copy;
    }

    // #region debug-point B:sequential-bgr-convert
    size_t step = frame_buffer.width_stride > 0 ? frame_buffer.width_stride : frame_buffer.width;
    int hs = frame_buffer.height_stride > 0 ? frame_buffer.height_stride : frame_buffer.height;
    cv::Mat nv12(hs * 3 / 2, frame_buffer.width, CV_8UC1, frame_buffer.virt_addr, step);
    cv::Mat bgr_full;
    int cvt_code = frame_buffer.format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
    if (debugYuvEnabled()) {
        std::fprintf(stderr,
                     "[DEBUG][YUV][B] ensureBgrFrame sequential fmt=%d w=%d h=%d ws=%d hs=%d step=%zu cvt=%s\n",
                     static_cast<int>(frame_buffer.format),
                     frame_buffer.width,
                     frame_buffer.height,
                     frame_buffer.width_stride,
                     frame_buffer.height_stride,
                     step,
                     cvt_code == cv::COLOR_YUV2BGR_NV21 ? "NV21" : "NV12");
    }
    cv::cvtColor(nv12, bgr_full, cvt_code);

    if (hs > frame_buffer.height) {
        frame_mat = bgr_full(cv::Rect(0, 0, frame_buffer.width, frame_buffer.height)).clone();
    } else {
        frame_mat = bgr_full;
    }
    // #endregion
    return !frame_mat.empty();
}
