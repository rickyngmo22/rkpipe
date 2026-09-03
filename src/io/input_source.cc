#include "io/input_source.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

namespace {
// 返回 true 表示 YOLOV8_VIDEO_READER 被显式设置，out 为解析结果。
// 未设置或值无法识别时返回 false，交由调用方按输入类型自动分流。
bool getReaderFromEnv(VideoReaderFactory::ReaderType& out) {
    const char* env = std::getenv("YOLOV8_VIDEO_READER");
    if (!env || env[0] == '\0') {
        return false;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "opencv") {
        out = VideoReaderFactory::ReaderType::OPENCV;
        return true;
    }
    if (value == "rkmpp") {
        out = VideoReaderFactory::ReaderType::RKMPP;
        return true;
    }
    if (value == "ffmpeg_rkmpp" || value == "rkmpp_ffmpeg") {
        out = VideoReaderFactory::ReaderType::FFMPEG_RKMPP;
        return true;
    }
    std::cerr << "[VideoInput] unknown YOLOV8_VIDEO_READER=\"" << value
              << "\", ignored" << std::endl;
    return false;
}

bool hasVideoExtension(const std::string& uri) {
    std::string lower = uri;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower.find(".mp4") != std::string::npos ||
           lower.find(".avi") != std::string::npos ||
           lower.find(".mov") != std::string::npos ||
           lower.find(".mkv") != std::string::npos ||
           lower.find(".flv") != std::string::npos ||
           lower.find(".ts") != std::string::npos;
}

bool hasStreamPrefix(const std::string& uri) {
    std::string lower = uri;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower.rfind("rtsp://", 0) == 0 ||
           lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("rtmp://", 0) == 0 ||
           lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 ||
           lower.rfind("srt://", 0) == 0 ||
           lower.rfind("http://", 0) == 0 ||
           lower.rfind("https://", 0) == 0 ||
           lower.rfind("/dev/video", 0) == 0;
}
}

std::unique_ptr<InputSource> InputSource::create(const std::string& uri) {
    if (uri.empty()) return nullptr;

    bool isVideo = hasVideoExtension(uri) || hasStreamPrefix(uri);

    if (isVideo) {
        return std::make_unique<VideoInputSource>();
    }
    return std::make_unique<ImageInputSource>();
}

bool VideoInputSource::open(const std::string& uri) {
    uri_ = uri;
    read_failures_ = 0;
    is_stream_source_ = hasStreamPrefix(uri);

    // 优先尊重 YOLOV8_VIDEO_READER 显式指定。
    VideoReaderFactory::ReaderType prefer = VideoReaderFactory::ReaderType::OPENCV;
    const bool env_explicit = getReaderFromEnv(prefer);

    // 未显式指定时按输入类型自动分流：
    //   - 本地文件：FFmpegRKMPP 硬解（h264_rkmpp/hevc_rkmpp），不丢帧且速度优于软解
    //   - 实时流：RKMPP 硬解，低延迟优先
    const bool is_stream = hasStreamPrefix(uri);
    VideoReaderFactory::ReaderType primary;
    if (env_explicit) {
        primary = prefer;
    } else {
        primary = is_stream ? VideoReaderFactory::ReaderType::RKMPP
                            : VideoReaderFactory::ReaderType::FFMPEG_RKMPP;
    }
    active_reader_type_ = primary;
    const VideoReaderFactory::ReaderType fallback =
        (primary == VideoReaderFactory::ReaderType::RKMPP ||
         primary == VideoReaderFactory::ReaderType::FFMPEG_RKMPP)
            ? VideoReaderFactory::ReaderType::OPENCV
            : VideoReaderFactory::ReaderType::RKMPP;

    reader_ = VideoReaderFactory::createReader(primary);
    VideoReaderFactory::ReaderType active = primary;
    if (!reader_->open(uri)) {
        reader_ = VideoReaderFactory::createReader(fallback);
        if (!reader_->open(uri)) {
            std::cerr << "[VideoInput] Failed: " << uri << std::endl;
            reader_.reset();
            return false;
        }
        active = fallback;
    }
    using_rkmpp_ = (active == VideoReaderFactory::ReaderType::RKMPP);
    const char* reader_name = "opencv";
    if (active == VideoReaderFactory::ReaderType::RKMPP) {
        reader_name = "rkmpp";
    } else if (active == VideoReaderFactory::ReaderType::FFMPEG_RKMPP) {
        reader_name = "ffmpeg_rkmpp";
    }
    std::cout << "[VideoInput] reader=" << reader_name << std::endl;

    info_.width = reader_->getWidth();
    info_.height = reader_->getHeight();
    info_.fps = reader_->getFPS();
    info_.totalFrames = reader_->getTotalFrames();
    std::cout << "[VideoInput] " << info_.width << "x" << info_.height
              << "@" << info_.fps << "fps, " << info_.totalFrames << " frames" << std::endl;
    return true;
}

bool VideoInputSource::read(cv::Mat& frame) {
    if (!reader_) return false;
    if (reader_->read(frame)) {
        read_failures_ = 0;
        return true;
    }
    read_failures_++;
    if (using_rkmpp_ && read_failures_ >= 5) {
        auto rkmpp = dynamic_cast<RKMPPVideoReader*>(reader_.get());
        if (rkmpp && rkmpp->isEofReached()) {
            return false;
        }
        if (is_stream_source_) {
            // 实时流断流：退避重连（阻塞直到恢复），恢复后立即重读一帧
            read_failures_ = 0;
            if (reconnectStreamBlocking()) {
                return read(frame);
            }
            return false;
        }
        reader_->release();
        reader_ = VideoReaderFactory::createReader(VideoReaderFactory::ReaderType::OPENCV);
        if (reader_ && reader_->open(uri_)) {
            using_rkmpp_ = false;
            read_failures_ = 0;
            info_.width = reader_->getWidth();
            info_.height = reader_->getHeight();
            info_.fps = reader_->getFPS();
            info_.totalFrames = reader_->getTotalFrames();
            std::cout << "[VideoInput] switch reader=rkmpp->opencv" << std::endl;
            return reader_->read(frame);
        }
        reader_.reset();
    }
    return false;
}

bool VideoInputSource::read(image_buffer_t& frame) {
    if (!reader_) return false;
    if (reader_->read(frame)) {
        read_failures_ = 0;
        return true;
    }
    read_failures_++;
    if (using_rkmpp_ && read_failures_ >= 5) {
        auto rkmpp = dynamic_cast<RKMPPVideoReader*>(reader_.get());
        if (rkmpp && rkmpp->isEofReached()) {
            return false;
        }
        if (is_stream_source_) {
            // 实时流断流：退避重连（阻塞直到恢复），恢复后立即重读一帧
            read_failures_ = 0;
            if (reconnectStreamBlocking()) {
                return read(frame);
            }
            return false;
        }
        reader_->release();
        reader_ = VideoReaderFactory::createReader(VideoReaderFactory::ReaderType::OPENCV);
        if (reader_ && reader_->open(uri_)) {
            using_rkmpp_ = false;
            read_failures_ = 0;
            info_.width = reader_->getWidth();
            info_.height = reader_->getHeight();
            info_.fps = reader_->getFPS();
            info_.totalFrames = reader_->getTotalFrames();
            std::cout << "[VideoInput] switch reader=rkmpp->opencv" << std::endl;
            return reader_->read(frame);
        }
        reader_.reset();
    }
    return false;
}

// 实时流断流后阻塞重连：按 RK_PIPE_STREAM_RECONNECT_MS（默认 3000ms）间隔重试，
// 直到重新打开成功；RK_PIPE_STREAM_RECONNECT_MAX_S 可设置总超时（秒，0=无限），
// 超时后放弃重连并返回 false，让上层正常结束流水线。Ctrl+C 可中断。
bool VideoInputSource::reconnectStreamBlocking() {
    int delay_ms = 3000;
    const char* ms_env = std::getenv("RK_PIPE_STREAM_RECONNECT_MS");
    if (ms_env && *ms_env) {
        const int v = std::atoi(ms_env);
        if (v > 0) {
            delay_ms = v;
        }
    }
    int max_s = 0;
    const char* max_env = std::getenv("RK_PIPE_STREAM_RECONNECT_MAX_S");
    if (max_env && *max_env) {
        const int v = std::atoi(max_env);
        if (v > 0) {
            max_s = v;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    std::cerr << "[VideoInput] stream lost, reconnecting every " << delay_ms << "ms: " << uri_ << std::endl;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        if (max_s > 0) {
            const double elapsed_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (elapsed_s >= static_cast<double>(max_s)) {
                std::cerr << "[VideoInput] reconnect timeout after " << max_s << "s, giving up: " << uri_
                          << std::endl;
                return false;
            }
        }
        reader_.reset();
        auto new_reader = VideoReaderFactory::createReader(active_reader_type_);
        if (new_reader && new_reader->open(uri_)) {
            reader_ = std::move(new_reader);
            using_rkmpp_ = (active_reader_type_ == VideoReaderFactory::ReaderType::RKMPP);
            read_failures_ = 0;
            info_.width = reader_->getWidth();
            info_.height = reader_->getHeight();
            info_.fps = reader_->getFPS();
            info_.totalFrames = reader_->getTotalFrames();
            ++reconnect_count_;
            std::cout << "[VideoInput] stream reconnected: " << uri_ << std::endl;
            return true;
        }
        std::cerr << "[VideoInput] reconnect failed, retrying in " << delay_ms << "ms: " << uri_ << std::endl;
    }
}

FrameInfo VideoInputSource::getInfo() const { return info_; }
bool VideoInputSource::isOpened() const { return reader_ && reader_->isOpened(); }
void VideoInputSource::getInputStats(std::uint64_t* dropped_frames, std::uint64_t* reconnects) const {
    if (dropped_frames) {
        *dropped_frames = reader_ ? reader_->droppedFrameCount() : 0;
    }
    if (reconnects) {
        *reconnects = reconnect_count_;
    }
}
void VideoInputSource::release() {
    if (reader_) {
        reader_->release();
        reader_.reset();
    }
    read_failures_ = 0;
    using_rkmpp_ = false;
    uri_.clear();
}

bool ImageInputSource::open(const std::string& uri) {
    struct stat st;
    if (::stat(uri.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        // 显式单文件（带斜杠的图片路径此前会被 opendir 误判而打不开）
        files_.push_back(uri);
    } else if (uri.find_last_of("/") == std::string::npos) {
        files_.push_back(uri);
    } else {
        DIR* d = opendir(uri.c_str());
        if (d) {
            struct dirent* ent;
            while ((ent = readdir(d))) {
                std::string f = ent->d_name;
                if (f != "." && f != "..") {
                    std::string ext = f.substr(f.find_last_of('.') + 1);
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == "jpg" || ext == "jpeg" || ext == "png" ||
                        ext == "bmp" || ext == "tiff") {
                        files_.push_back(uri + "/" + f);
                    }
                }
            }
            closedir(d);
        }
    }

    std::sort(files_.begin(), files_.end());
    if (!files_.empty()) {
        cv::Mat img = cv::imread(files_.front());
        if (!img.empty()) {
            width_ = img.cols;
            height_ = img.rows;
        }
    }
    std::cout << "[ImageInput] " << files_.size() << " images" << std::endl;
    return !files_.empty();
}

bool ImageInputSource::read(cv::Mat& frame) {
    if (index_ >= files_.size()) return false;
    frame = cv::imread(files_[index_++]);
    return !frame.empty();
}

FrameInfo ImageInputSource::getInfo() const {
    FrameInfo fi;
    fi.width = width_;
    fi.height = height_;
    fi.fps = 30.0;
    fi.totalFrames = (int)files_.size();
    return fi;
}

bool ImageInputSource::isOpened() const { return !files_.empty(); }
void ImageInputSource::release() { files_.clear(); index_ = 0; }

std::string ImageInputSource::currentSourceName() const {
    if (index_ > 0 && index_ <= files_.size()) {
        return files_[index_ - 1];
    }
    return {};
}
