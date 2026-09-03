#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <opencv2/opencv.hpp>
#include "common.h"

class VideoOutput {
public:
    struct Config {
        std::string windowName = "Output";
        int windowWidth = 640;
        int windowHeight = 480;
        double targetFPS = 30.0;
        double displayFPS = 30.0;
        double writeFPS = 30.0;
        bool enableDisplay = true;
        bool enableWrite = false;
        std::string outputPath;
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double quality = 0.7;
        std::string qualityMode = "balanced";
        double bitrateMbps = 0.0;
        int crf = -1;
        std::string preset;
        bool lowLatency = false;
        int maxQueueSize = 20;
        int maxMemoryMB = 100;
        bool useRkMpp = false;
        int dropFramesOnOverflow = -1;  // -1=自动(网络输出才丢帧)；0=队列满阻塞(背压)；1=队满丢帧
    };

    explicit VideoOutput(Config config);
    virtual ~VideoOutput();

    static int calculateOptimalQueueSize(int width, int height, int maxMemoryMB = 100);
    static size_t estimateFrameMemory(int width, int height);

    bool init();
    void pushFrame(cv::Mat frame);
    void pushFrame(int frameIndex, cv::Mat frame);
    bool pushFrame(int frameIndex, const image_buffer_t& frame);
    void setDisplayEnabled(bool enabled);
    void setWriteEnabled(bool enabled);
    void setTargetFPS(double fps);
    double getCurrentFPS() const;
    double getDisplayFPS() const;
    double getWriteFPS() const;
    bool isRunning() const;
    void stop();
    int getQueuedFrameCount() const;
    size_t getCurrentMemoryUsage() const;
    int getMaxQueueSize() const { return config_.maxQueueSize; }

    // 输出量化指标
    long long getWrittenFrames() const { return written_frames_.load(); }
    long long getDroppedFrames() const { return dropped_frames_.load(); }

private:
    struct RKMPPWriter;

    struct QueuedFrame {
        int index = -1;
        cv::Mat frame;
    };

    struct QueuedFrameYuv {
        int index = -1;
        image_buffer_t frame = {};
    };

    void displayLoop();
    void writeLoop();

    Config config_;
    bool initialized_ = false;

    std::atomic<bool> running_{false};
    std::thread displayThread_;
    std::thread writeThread_;

    std::queue<QueuedFrame> displayQueue_;
    std::queue<QueuedFrame> writeQueue_;
    std::queue<QueuedFrameYuv> writeQueueYuv_;
    mutable std::mutex displayMutex_;
    mutable std::mutex writeMutex_;
    std::condition_variable cvDisplay_;
    std::condition_variable cvWrite_;

    std::atomic<double> targetDisplayFPS_{30.0};
    std::atomic<double> targetWriteFPS_{30.0};
    std::atomic<double> currentDisplayFPS_{0.0};
    std::atomic<double> currentWriteFPS_{0.0};

    cv::VideoWriter videoWriter_;
    std::unique_ptr<RKMPPWriter> rkmppWriter_;
    bool writeEnabled_ = false;
    bool displayEnabled_ = true;
    bool dropFramesOnOverflow_ = true;

    int displayFrameCount_ = 0;
    std::chrono::steady_clock::time_point displayFpsCalcStart_;
    std::atomic<size_t> currentMemoryUsage_{0};
    std::atomic<int> lastDisplayedIndex_{-1};
    std::atomic<int> lastWrittenIndex_{-1};

    // 输出量化指标
    std::atomic<long long> written_frames_{0};
    std::atomic<long long> dropped_frames_{0};
};

class VideoOutputFactory {
public:
    enum class OutputType {
        OPENCV,
        FFMPEG,
        GSTREAMER,
        RKMPP
    };

    static std::unique_ptr<VideoOutput> create(OutputType type = OutputType::OPENCV, VideoOutput::Config config = VideoOutput::Config());
};
