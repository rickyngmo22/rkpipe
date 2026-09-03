#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <opencv2/opencv.hpp>

class VideoDisplay {
public:
    virtual ~VideoDisplay() = default;

    virtual bool init(const std::string& windowName, int width, int height) = 0;

    virtual void pushFrame(const cv::Mat& frame) = 0;

    virtual void setTargetFPS(double fps) = 0;

    virtual double getTargetFPS() const = 0;

    virtual double getCurrentFPS() const = 0;

    virtual bool isRunning() const = 0;

    virtual void stop() = 0;

    virtual int getQueuedFrameCount() const = 0;

    virtual void clearQueue() = 0;
};

class OpenCVDisplay : public VideoDisplay {
public:
    OpenCVDisplay();
    ~OpenCVDisplay() override;

    bool init(const std::string& windowName, int width, int height) override;

    void pushFrame(const cv::Mat& frame) override;

    void setTargetFPS(double fps) override;

    double getTargetFPS() const override;

    double getCurrentFPS() const override;

    bool isRunning() const override;

    void stop() override;

    int getQueuedFrameCount() const override;

    void clearQueue() override;

private:
    void displayLoop();

    std::string windowName_;
    int width_;
    int height_;

    std::atomic<bool> running_{false};
    std::thread displayThread_;

    cv::Mat frontBuffer_;
    cv::Mat backBuffer_;
    mutable std::mutex bufferMutex_;

    std::queue<cv::Mat> frameQueue_;
    mutable std::mutex queueMutex_;
    std::condition_variable cvFrame_;

    std::atomic<double> targetFPS_{30.0};
    std::atomic<double> currentFPS_{0.0};

    std::chrono::steady_clock::time_point lastFrameTime_;
    int frameCount_{0};
    std::chrono::steady_clock::time_point fpsCalcStart_;
};

class VideoDisplayFactory {
public:
    enum class DisplayType {
        OPENCV,
        FFMPEG,
        GSTREAMER,
        RKMPP
    };

    static std::unique_ptr<VideoDisplay> create(DisplayType type = DisplayType::OPENCV);
};
