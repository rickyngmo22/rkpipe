#pragma once

#include <functional>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "../utils/common.h"
#include "io_defs.h"
#include "postprocess/postprocess.h"
#include "video_output.h"
#include "web_preview_server.h"

class AppConfig;

class OutputRouter {
public:
    bool init(const FrameInfo& info, const AppConfig& options);

    void write(int frame_index, cv::Mat frame);
    bool write(int frame_index, const image_buffer_t& frame);
    // seg_overlay 非空时：预览降采样（defer_seg_overlay）路径下，在转换后的 BGR 帧上补画叠加
    bool write(int frame_index, const image_buffer_t& frame, const seg_detect_result_list* seg_overlay);

    bool supportsZeroCopyDisplay() const;
    bool supportsZeroCopyVideo() const;
    double displayFPS() const;
    double writeFPS() const;
    void stop();
    bool enabled() const;

    // 输入侧统计源（丢帧数 / 重连次数），转发到 WebPreviewServer /status.json
    using InputStatSource = WebPreviewServer::InputStatSource;
    void setInputStatSource(InputStatSource source);

    // 输出量化指标（转发到 WebPreviewServer / VideoOutput）
    std::uint64_t webPublishedFrames() const;
    std::uint64_t webDroppedFrames() const;
    double webEncodeMs() const;
    int webClientCount() const;
    long long videoWrittenFrames() const;
    long long videoDroppedFrames() const;

private:
    std::unique_ptr<VideoOutput> video_;
      std::unique_ptr<WebPreviewServer> web_preview_;
    bool zero_copy_video_enabled_ = false;
};
