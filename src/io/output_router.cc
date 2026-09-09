#include "../../include/io/output_router.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include "../../include/config/app_config.h"
#include "preprocess.h"
#include "utils/draw_utils.h"

namespace {

std::vector<int> parseWebPreviewPorts(const std::string& ports_csv) {
    std::vector<int> ports;
    std::stringstream ss(ports_csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }),
                   item.end());
        if (item.empty()) {
            continue;
        }
        const int port = std::atoi(item.c_str());
        if (port > 0) {
            ports.push_back(port);
        }
    }
    return ports;
}

// 解析 WebRTC 播放 URL 列表：以 '|' 分隔（URL 内含 ':' 与 '/'，不能用逗号），
// 顺序与窗格对齐（0=本服务，1..N=extra_stream_ports）。空位保留（对应窗格回退 MJPEG）。
std::vector<std::string> parseWebrtcUrls(const std::string& urls_csv) {
    std::vector<std::string> urls;
    std::stringstream ss(urls_csv);
    std::string item;
    while (std::getline(ss, item, '|')) {
        item.erase(0, item.find_first_not_of(" \t\r\n"));
        item.erase(item.find_last_not_of(" \t\r\n") + 1);
        urls.push_back(item);
    }
    return urls;
}

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    return std::string(value) == "1" || std::string(value) == "true" || std::string(value) == "TRUE" ||
           std::string(value) == "yes" || std::string(value) == "on";
}

// 基准模式：跳过 web JPEG 编码等输出侧开销，只测解码+推理+pass-through 的纯上限
bool benchModeEnabled() {
    return envFlagEnabled("RK_PIPE_BENCH");
}

bool publishWebPreviewFrame(WebPreviewServer* web_preview,
                            const image_buffer_t& frame,
                            const seg_detect_result_list* seg_overlay) {
    if (!web_preview || frame.width <= 0 || frame.height <= 0 || !frame.virt_addr) {
        return false;
    }

    const bool stages_debug = []() {
        const char* v = std::getenv("RK_PIPE_DEBUG_PIPELINE_STAGES");
        return v && *v && std::string(v) != "0";
    }();
    auto t0 = std::chrono::steady_clock::now();

    cv::Mat bgr;
    bool pre_scaled = false;
    switch (frame.format) {
        case IMAGE_FORMAT_BGR888: {
            int stride = frame.width_stride > 0 ? frame.width_stride : frame.width;
            cv::Mat wrapped(frame.height, frame.width, CV_8UC3, frame.virt_addr, static_cast<std::size_t>(stride) * 3);
            bgr = wrapped.clone();
            break;
        }
        case IMAGE_FORMAT_RGB888: {
            int stride = frame.width_stride > 0 ? frame.width_stride : frame.width;
            cv::Mat wrapped(frame.height, frame.width, CV_8UC3, frame.virt_addr, static_cast<std::size_t>(stride) * 3);
            cv::cvtColor(wrapped, bgr, cv::COLOR_RGB2BGR);
            break;
        }
        case IMAGE_FORMAT_RGBA8888: {
            int stride = frame.width_stride > 0 ? frame.width_stride : frame.width;
            cv::Mat wrapped(frame.height, frame.width, CV_8UC4, frame.virt_addr, static_cast<std::size_t>(stride) * 4);
            cv::cvtColor(wrapped, bgr, cv::COLOR_RGBA2BGR);
            break;
        }
        case IMAGE_FORMAT_GRAY8: {
            int stride = frame.width_stride > 0 ? frame.width_stride : frame.width;
            cv::Mat wrapped(frame.height, frame.width, CV_8UC1, frame.virt_addr, stride);
            cv::cvtColor(wrapped, bgr, cv::COLOR_GRAY2BGR);
            break;
        }
        case IMAGE_FORMAT_YUV420SP_NV12:
        case IMAGE_FORMAT_YUV420SP_NV21: {
            // 优先 RGA 硬件转换（省 CPU cvtColor），失败回退 CPU
            // preview_scale<1.0 时 RGA 一次完成 NV12→BGR+降采样，编码线程不再二次缩放
            const double scale = web_preview->previewScale();
            static std::atomic<bool> rga_nv12_disabled{false};  // 失败一次后锁定 CPU，避免 librga 每帧刷屏
            if (!rga_nv12_disabled && rga_nv12_to_bgr(frame, bgr, scale) == 0) {
                pre_scaled = (bgr.cols != frame.width || bgr.rows != frame.height);
                break;
            }
            if (!rga_nv12_disabled) {
                rga_nv12_disabled = true;
                std::fprintf(stderr,
                             "[OutputRouter] RGA NV12->BGR unavailable, fallback to CPU cvtColor\n");
            }
            int stride = frame.width_stride > 0 ? frame.width_stride : frame.width;
            int hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;
            cv::Mat yuv(hstride + hstride / 2, stride, CV_8UC1, frame.virt_addr);
            int cvt_code = frame.format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2BGR_NV21
                                                                      : cv::COLOR_YUV2BGR_NV12;
            cv::cvtColor(yuv, bgr, cvt_code);
            if (bgr.cols != frame.width || bgr.rows != frame.height) {
                bgr = bgr(cv::Rect(0, 0, frame.width, frame.height)).clone();
            }
            break;
        }
        default:
            return false;
    }

    // 延迟叠加：预览降采样时掩膜由输出线程画在缩放后的 BGR 帧上（坐标按缩放比例换算）
    if (seg_overlay && !bgr.empty()) {
        drawSegResultsBGR(bgr, *seg_overlay, pre_scaled ? web_preview->previewScale() : 1.0);
    }

    web_preview->publishFrame(std::move(bgr), pre_scaled);

    if (stages_debug) {
        static thread_local double sum_cvt = 0.0, sum_pub = 0.0;
        static thread_local int pub_count = 0;
        auto t1 = std::chrono::steady_clock::now();
        sum_cvt += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_pub += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        ++pub_count;
        if ((pub_count % 200) == 0) {
            std::fprintf(stderr,
                         "[rk_pipe][publish] cvt=%.1fms publish=%.1fms total=%.1fms fmt=%d\n",
                         sum_cvt / pub_count,
                         sum_pub / pub_count,
                         (sum_cvt + sum_pub) / pub_count,
                         frame.format);
        }
    }
    return true;
}

}  // namespace

bool OutputRouter::init(const FrameInfo& info, const AppConfig& options) {
    std::string backend = options.output_backend;
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    bool backend_is_opencv = backend == "opencv";
    bool enable_display = options.gui;
    bool enable_write = !backend_is_opencv && !options.output_video_path.empty();
    bool enable_web_preview = options.web_preview;
    if (!enable_display && !enable_write && !enable_web_preview) {
        return false;
    }

    const double legacy_output_fps = options.output_fps > 0.0 ? options.output_fps : info.fps;
    if (enable_display || enable_write) {
        VideoOutput::Config video_config;
        const double display_fps = options.display_fps > 0.0 ? options.display_fps : legacy_output_fps;
        const double write_fps = options.stream_fps > 0.0 ? options.stream_fps : legacy_output_fps;
        video_config.windowName = "YOLO Detection";
        video_config.windowWidth = info.width;
        video_config.windowHeight = info.height;
        video_config.targetFPS = legacy_output_fps;
        video_config.displayFPS = display_fps;
        video_config.writeFPS = write_fps;
        video_config.enableDisplay = enable_display;
        video_config.enableWrite = enable_write;
        video_config.maxQueueSize *= 2;
        video_config.outputPath = enable_write ? options.output_video_path : std::string();
        video_config.qualityMode = options.output_quality;
        video_config.bitrateMbps = options.output_bitrate_mbps;
        video_config.crf = options.output_crf;
        video_config.preset = options.output_preset;
        video_config.lowLatency = options.output_low_latency;
        video_config.dropFramesOnOverflow = options.drop_frames_on_overflow;

        auto output_type = backend_is_opencv ? VideoOutputFactory::OutputType::OPENCV
                                             : VideoOutputFactory::OutputType::FFMPEG;
        video_ = VideoOutputFactory::create(output_type, video_config);
        if (!video_ || !video_->init()) {
            video_.reset();
            return false;
        }
    }

    const bool disable_zero_copy_video = envFlagEnabled("RK_PIPE_DISABLE_ZERO_COPY_VIDEO");
    std::string web_preview_mode = "live";
    // 网络推流（rkmpp 硬编）同样走 NV12 零拷贝：buffer 由写线程持有到编码完成后才
    // 归还解码池，否则推流路径会把帧提前释放、被读线程覆盖，编码到无叠加框的旧内容。
    zero_copy_video_enabled_ = enable_write && !disable_zero_copy_video;
    if (enable_write && disable_zero_copy_video) {
        std::fprintf(stderr, "[OutputRouter] Zero-copy video disabled by RK_PIPE_DISABLE_ZERO_COPY_VIDEO\n");
    }

    if (enable_web_preview) {
        WebPreviewServer::Config web_config;
        web_config.bind_address = options.web_preview_bind;
        web_config.port = options.web_preview_port;
        web_config.jpeg_quality = options.web_preview_quality;
        // 本地文件按源帧率预览：totalFrames 有效即走 replay；fps 探测失败时兜底 30fps
        const bool want_replay = options.web_preview_mode == "replay" && info.totalFrames > 0;
        web_config.preview_mode = want_replay ? "replay" : "live";
        web_config.replay_fps = want_replay ? (info.fps > 0.0 ? info.fps : 30.0) : 0.0;
        web_config.source_fps = info.fps;
        web_config.frame_width = info.width;
        web_config.frame_height = info.height;
        web_config.eof_linger_ms = options.web_preview_eof_linger_ms;
        web_config.replay_pace = options.web_preview_replay_pace;
        web_config.preview_scale = options.web_preview_scale;
        web_preview_mode = web_config.preview_mode;
        if (options.web_preview_secondary_port > 0) {
            web_config.extra_stream_ports.push_back(options.web_preview_secondary_port);
        }
        const auto extra_stream_ports = parseWebPreviewPorts(options.web_preview_extra_ports);
        web_config.extra_stream_ports.insert(web_config.extra_stream_ports.end(),
                                             extra_stream_ports.begin(),
                                             extra_stream_ports.end());
        web_config.title = "rkpipe_preview";
        web_config.task = options.task;
        web_config.input_path = options.input_path;
        web_config.webrtc_urls = parseWebrtcUrls(options.web_preview_webrtc_urls);
        web_preview_ = std::make_unique<WebPreviewServer>();
        if (!web_preview_->start(web_config)) {
            web_preview_.reset();
            if (!video_) {
                return false;
            }
        }
    }
        return true;
}

void OutputRouter::write(int frame_index, cv::Mat frame) {
        if (web_preview_ && !benchModeEnabled() && !frame.empty()) {
        web_preview_->publishFrame(frame);
    }
    if (video_) {
        video_->pushFrame(frame_index, std::move(frame));
    }
}

bool OutputRouter::write(int frame_index, const image_buffer_t& frame) {
    bool web_written = false;
    if (web_preview_ && !benchModeEnabled()) {
        web_written = publishWebPreviewFrame(web_preview_.get(), frame, nullptr);
    }
        if (video_ && video_->pushFrame(frame_index, frame)) {
        return true;
    }
    if (!video_) {
        return web_written;
    }
    return false;
}

bool OutputRouter::write(int frame_index, const image_buffer_t& frame, const seg_detect_result_list* seg_overlay) {
    bool web_written = false;
    if (web_preview_ && !benchModeEnabled()) {
        web_written = publishWebPreviewFrame(web_preview_.get(), frame, seg_overlay);
    }
    if (video_ && video_->pushFrame(frame_index, frame)) {
        return true;
    }
    if (!video_) {
        return web_written;
    }
    return false;
}

bool OutputRouter::supportsZeroCopyDisplay() const {
    return false;
}

bool OutputRouter::supportsZeroCopyVideo() const {
    return zero_copy_video_enabled_ && video_ != nullptr;
}

double OutputRouter::displayFPS() const {
    return video_ ? video_->getDisplayFPS() : 0.0;
}

double OutputRouter::writeFPS() const {
    if (video_) {
        return video_->getWriteFPS();
    }
    return web_preview_ ? web_preview_->publishFPS() : 0.0;
}

void OutputRouter::stop() {
    if (web_preview_) {
        web_preview_->stop();
        web_preview_.reset();
    }
    if (video_) {
        video_->stop();
        video_.reset();
    }
}

bool OutputRouter::enabled() const {
    return video_ != nullptr || web_preview_ != nullptr;
}

void OutputRouter::setInputStatSource(InputStatSource source) {
    if (web_preview_) {
        web_preview_->setInputStatSource(std::move(source));
    }
}

std::uint64_t OutputRouter::webPublishedFrames() const {
    return web_preview_ ? web_preview_->publishedFrames() : 0;
}

std::uint64_t OutputRouter::webDroppedFrames() const {
    return web_preview_ ? web_preview_->droppedFrames() : 0;
}

double OutputRouter::webEncodeMs() const {
    return web_preview_ ? web_preview_->encodeMs() : 0.0;
}

int OutputRouter::webClientCount() const {
    return web_preview_ ? web_preview_->clientCount() : 0;
}

long long OutputRouter::videoWrittenFrames() const {
    return video_ ? video_->getWrittenFrames() : 0;
}

long long OutputRouter::videoDroppedFrames() const {
    return video_ ? video_->getDroppedFrames() : 0;
}
