#include "io/video_display.h"
#include <iostream>
#include <chrono>
#include <algorithm>

OpenCVDisplay::OpenCVDisplay()
    : width_(0), height_(0)
{
}

OpenCVDisplay::~OpenCVDisplay()
{
    stop();
}

bool OpenCVDisplay::init(const std::string& windowName, int width, int height)
{
    windowName_ = windowName;
    width_ = width;
    height_ = height;

    cv::namedWindow(windowName_, cv::WINDOW_NORMAL);
    if (width > 0 && height > 0) {
        cv::resizeWindow(windowName_, width, height);
    }

    running_ = true;
    displayThread_ = std::thread(&OpenCVDisplay::displayLoop, this);
    fpsCalcStart_ = std::chrono::steady_clock::now();

    return true;
}

void OpenCVDisplay::pushFrame(const cv::Mat& frame)
{
    if (!running_) return;

    cv::Mat copy;
    if (frame.empty()) return;

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (frameQueue_.size() < 30) {
            frameQueue_.push(frame.clone());
        }
    }
    cvFrame_.notify_one();
}

void OpenCVDisplay::setTargetFPS(double fps)
{
    targetFPS_ = std::max(1.0, std::min(fps, 120.0));
}

double OpenCVDisplay::getTargetFPS() const
{
    return targetFPS_.load();
}

double OpenCVDisplay::getCurrentFPS() const
{
    return currentFPS_.load();
}

bool OpenCVDisplay::isRunning() const
{
    return running_.load();
}

void OpenCVDisplay::stop()
{
    if (!running_.exchange(false)) {
        return;
    }

    cvFrame_.notify_all();
    if (displayThread_.joinable()) {
        displayThread_.join();
    }

    cv::destroyWindow(windowName_);
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (!frameQueue_.empty()) {
            frameQueue_.pop();
        }
    }
}

int OpenCVDisplay::getQueuedFrameCount() const
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    return static_cast<int>(frameQueue_.size());
}

void OpenCVDisplay::clearQueue()
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    while (!frameQueue_.empty()) {
        frameQueue_.pop();
    }
}

void OpenCVDisplay::displayLoop()
{
    auto nextFrameTime = std::chrono::steady_clock::now();
    lastFrameTime_ = nextFrameTime;

    while (running_) {
        cv::Mat frame;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            cvFrame_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !frameQueue_.empty() || !running_; });

            if (!running_) break;

            if (!frameQueue_.empty()) {
                frame = std::move(frameQueue_.front());
                frameQueue_.pop();
            }
        }

        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            backBuffer_ = frame.clone();
        }

        cv::imshow(windowName_, backBuffer_);
        cv::waitKey(1);

        double fps = targetFPS_.load();
        if (fps > 0) {
            auto frameDuration = std::chrono::milliseconds(static_cast<int>(1000.0 / fps));
            nextFrameTime = std::chrono::steady_clock::now() + frameDuration;

            if (nextFrameTime > std::chrono::steady_clock::now()) {
                std::this_thread::sleep_until(nextFrameTime);
            }
        }

        frameCount_++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsCalcStart_).count();
        if (elapsed >= 1000) {
            currentFPS_ = frameCount_ * 1000.0 / elapsed;
            frameCount_ = 0;
            fpsCalcStart_ = now;
        }
    }
}

std::unique_ptr<VideoDisplay> VideoDisplayFactory::create(DisplayType type)
{
    switch (type) {
        case DisplayType::OPENCV:
            return std::make_unique<OpenCVDisplay>();
        case DisplayType::FFMPEG:
        case DisplayType::GSTREAMER:
        case DisplayType::RKMPP:
            std::cerr << "Display type not implemented yet, falling back to OpenCV" << std::endl;
            return std::make_unique<OpenCVDisplay>();
        default:
            return std::make_unique<OpenCVDisplay>();
    }
}
