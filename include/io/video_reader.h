#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "common.h"

class VideoReader {
public:
    virtual ~VideoReader() = default;

    virtual bool open(const std::string& source) = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool read(image_buffer_t& image) = 0;
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual double getFPS() const = 0;
    virtual int getTotalFrames() const = 0;
    virtual void release() = 0;
    virtual bool isOpened() const = 0;

    // 输入侧丢弃帧数（解码器 discard/errinfo），用于 /status.json 暴露，判断是否 NPU-bound
    virtual std::uint64_t droppedFrameCount() const { return 0; }

    static void releaseFrame(image_buffer_t& image);
};

class OpenCVVideoReader : public VideoReader {
public:
    OpenCVVideoReader();
    ~OpenCVVideoReader() override;

    bool open(const std::string& source) override;
    bool read(cv::Mat& frame) override;
    bool read(image_buffer_t& image) override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int getTotalFrames() const override;
    void release() override;
    bool isOpened() const override;

private:
    cv::VideoCapture cap;
    int width;
    int height;
    double fps;
    int total_frames;
    // 摄像头画面翻转（cv::flip flipCode）：1=水平镜像，0=垂直，-1=180°。0=不翻转。
    int flip_code_;
};

class RKMPPVideoReader : public VideoReader {
public:
    RKMPPVideoReader();
    ~RKMPPVideoReader() override;

    bool open(const std::string& source) override;
    bool read(cv::Mat& frame) override;
    bool read(image_buffer_t& image) override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int getTotalFrames() const override;
    void release() override;
    bool isOpened() const override;
    bool isEofReached() const;
    std::uint64_t droppedFrameCount() const override;

private:
    void* pipe_;
    std::string source_;
    int width;
    int height;
    double fps;
    int total_frames;
    size_t frame_bytes;
    void* internal_state_;
};

// 用 libavcodec 加载 Rockchip MPP 硬解（h264_rkmpp / hevc_rkmpp）。
// 复用 ffmpeg 验证过的解码路径（阻塞取帧 + 外部缓冲组 + 帧生命周期管理），
// 避免手搓 MPP 循环时解码器积压丢帧。输出统一转为 BGR888 CPU 帧。
class FFmpegRKMPPVideoReader : public VideoReader {
public:
    FFmpegRKMPPVideoReader();
    ~FFmpegRKMPPVideoReader() override;

    bool open(const std::string& source) override;
    bool read(cv::Mat& frame) override;
    bool read(image_buffer_t& image) override;
    int getWidth() const override;
    int getHeight() const override;
    double getFPS() const override;
    int getTotalFrames() const override;
    void release() override;
    bool isOpened() const override;

private:
    void* impl_;  // FFmpegRKMPPState
    int width;
    int height;
    double fps;
    int total_frames;
};

class VideoReaderFactory {
public:
    enum class ReaderType {
        OPENCV,
        FFMPEG,
        GSTREAMER,
        RKMPP,
        FFMPEG_RKMPP
    };

    static std::unique_ptr<VideoReader> createReader(ReaderType type = ReaderType::OPENCV);
};
