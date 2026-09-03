#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "io/io_defs.h"
#include "io/video_reader.h"

class InputSource {
public:
    virtual ~InputSource() = default;
    virtual bool open(const std::string& uri) = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool read(image_buffer_t& frame) { return false; }
    virtual FrameInfo getInfo() const = 0;
    virtual bool isOpened() const = 0;
    virtual void release() = 0;

    // 输入侧统计（丢帧数 / 重连次数），供 /status.json 暴露
    virtual void getInputStats(std::uint64_t* dropped_frames, std::uint64_t* reconnects) const {
        if (dropped_frames) *dropped_frames = 0;
        if (reconnects) *reconnects = 0;
    }

    // 最近一次成功 read 的源名（图片目录 = 文件路径；视频 = uri）。无历史返回空。
    virtual std::string currentSourceName() const { return {}; }

    static std::unique_ptr<InputSource> create(const std::string& uri);
};

class VideoInputSource : public InputSource {
public:
    bool open(const std::string& uri) override;
    bool read(cv::Mat& frame) override;
    bool read(image_buffer_t& frame) override;
    FrameInfo getInfo() const override;
    bool isOpened() const override;
    void release() override;
    void getInputStats(std::uint64_t* dropped_frames, std::uint64_t* reconnects) const override;

private:
    std::unique_ptr<VideoReader> reader_;
    FrameInfo info_;
    std::string uri_;
    int read_failures_ = 0;
    bool using_rkmpp_ = false;
    bool is_stream_source_ = false;   // 实时流（rtsp/rtmp/udp/http/...）断流时退避重连
    VideoReaderFactory::ReaderType active_reader_type_ = VideoReaderFactory::ReaderType::OPENCV;
    std::uint64_t reconnect_count_ = 0;  // 断流重连成功次数

    // 实时流断流后阻塞重连（RK_PIPE_STREAM_RECONNECT_MS 控制间隔，默认 3000ms）
    bool reconnectStreamBlocking();
};

class ImageInputSource : public InputSource {
public:
    bool open(const std::string& uri) override;
    bool read(cv::Mat& frame) override;
    FrameInfo getInfo() const override;
    bool isOpened() const override;
    void release() override;
    std::string currentSourceName() const override;

    static std::unique_ptr<ImageInputSource> create(const std::string& folderPath);

private:
    std::vector<std::string> files_;
    size_t index_ = 0;
    int width_ = 640;
    int height_ = 480;
};
