#include "io/video_output.h"
#include "io/video_reader.h"
#include <iostream>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

namespace {

std::string toLowerCopy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string getExtension(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    return toLowerCopy(path.substr(pos));
}

bool isNetworkUrl(const std::string& path) {
    std::string lower = toLowerCopy(path);
    return lower.rfind("rtmp://", 0) == 0 ||
           lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("rtsp://", 0) == 0 ||
           lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 ||
           lower.rfind("srt://", 0) == 0;
}

bool isRtmpUrl(const std::string& path) {
    std::string lower = toLowerCopy(path);
    return lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0;
}

std::string forceRtmpPublishUrl(const std::string& path) {
    return path;
}

// SRT 推流地址规范化：ffmpeg 的 srt://host:port/path 不会自动生成 streamid，
// MediaMTX 等服务器要求 streamid 指定流路径（否则握手被拒）。
// 这里把直观形式 `srt://host:port/live/key` 转换为 MediaMTX 可接受的完整 URL：
//   srt://host:port?mode=caller&streamid=#!::r=live/key,m=publish
// 已带 query 或无路径的 URL 原样返回（调用方可直接给完整格式）。
std::string normalizeSrtPublishUrl(const std::string& path) {
    std::string lower = toLowerCopy(path);
    if (lower.rfind("srt://", 0) != 0 || path.find('?') != std::string::npos) {
        return path;
    }
    const std::string rest = path.substr(6);  // 去掉 "srt://"（6 字符）
    const size_t slash = rest.find('/');
    if (slash == std::string::npos) {
        return path;
    }
    const std::string host_port = rest.substr(0, slash);
    const std::string stream_path = rest.substr(slash + 1);
    if (stream_path.empty()) {
        return path;
    }
    return "srt://" + host_port + "?mode=caller&streamid=#!::r=" + stream_path + ",m=publish";
}

const char* getOutputFormatName(const std::string& path) {
    std::string lower = toLowerCopy(path);
    if (lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0) {
        return "flv";
    }
    if (lower.rfind("rtsp://", 0) == 0 || lower.rfind("rtsps://", 0) == 0) {
        return "rtsp";
    }
    if (lower.rfind("udp://", 0) == 0 || lower.rfind("srt://", 0) == 0) {
        return "mpegts";
    }
    return nullptr;
}

std::vector<int> buildCodecCandidates(const std::string& path, int fallback) {
    std::string ext = getExtension(path);
    if (ext == ".mp4" || ext == ".m4v") {
        return {cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fallback};
    }
    if (ext == ".avi") {
        return {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                fallback};
    }
    return {fallback,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G')};
}

std::string avErrorToString(int errnum) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(errnum, buffer, sizeof(buffer));
    return std::string(buffer);
}

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool shouldUseRkMppForOutput(const std::string& outputPath) {
    if (envFlagEnabled("RK_PIPE_DISABLE_RKMPP")) {
        return false;
    }
    if (envFlagEnabled("RK_PIPE_FORCE_RKMPP")) {
        return true;
    }
    // RKMPP 编码器在 RK3588 上接近 0 CPU 占用，本地文件与网络推流
    // （RTMP/RTSP/UDP/SRT）均默认硬编；RK_PIPE_DISABLE_RKMPP 仍可整体关闭，
    // 打不开编码器时 init() 内会回退 libx264。
    (void)outputPath;
    return true;
}

bool convertYuv420spToBgrForDisplay(const image_buffer_t& frame, cv::Mat* out_bgr) {
    if (!out_bgr || frame.virt_addr == nullptr) {
        return false;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    const int width = frame.width;
    const int height = frame.height;
    const int width_stride = frame.width_stride > 0 ? frame.width_stride : width;
    const int height_stride = frame.height_stride > 0 ? frame.height_stride : height;
    if (width <= 0 || height <= 0 || width_stride < width || height_stride < height) {
        return false;
    }

    cv::Mat bgr;
    const int cvt_code = frame.format == IMAGE_FORMAT_YUV420SP_NV21
                             ? cv::COLOR_YUV2BGR_NV21
                             : cv::COLOR_YUV2BGR_NV12;

    if (height_stride > height || width_stride > width) {
        cv::Mat contiguous_yuv(height * 3 / 2, width, CV_8UC1);
        const uint8_t* src_y = frame.virt_addr;
        uint8_t* dst_y = contiguous_yuv.data;
        for (int y = 0; y < height; ++y) {
            std::memcpy(dst_y + y * width, src_y + y * width_stride, width);
        }

        const uint8_t* src_uv = frame.virt_addr + width_stride * height_stride;
        uint8_t* dst_uv = contiguous_yuv.data + width * height;
        for (int y = 0; y < height / 2; ++y) {
            std::memcpy(dst_uv + y * width, src_uv + y * width_stride, width);
        }

        cv::cvtColor(contiguous_yuv, bgr, cvt_code);
    } else {
        cv::Mat yuv(height * 3 / 2, width, CV_8UC1, frame.virt_addr, width_stride);
        cv::cvtColor(yuv, bgr, cvt_code);
    }

    if (bgr.cols != width || bgr.rows != height) {
        bgr = bgr(cv::Rect(0, 0, width, height)).clone();
    }
    *out_bgr = std::move(bgr);
    return !out_bgr->empty();
}

std::string normalizeQualityMode(const std::string& value) {
    std::string lower = toLowerCopy(value);
    if (lower == "speed" || lower == "fast" || lower == "low") {
        return "speed";
    }
    if (lower == "fidelity" || lower == "hi_fi" || lower == "hifi" ||
        lower == "ultra" || lower == "max" || lower == "lossless" ||
        lower == "visually_lossless" || lower == "visually-lossless") {
        return "fidelity";
    }
    if (lower == "source" || lower == "source_like" || lower == "source-like" ||
        lower == "quality" || lower == "high") {
        return "source";
    }
    return "balanced";
}

struct SoftwareEncodeTuning {
    std::string quality_mode = "balanced";
    std::string preset = "fast";
    int crf = 18;
    bool low_latency = false;
    int64_t bit_rate = 8LL * 1000 * 1000;
    int64_t rc_max_rate = 8LL * 1000 * 1000;
    int64_t rc_buffer_size = 16LL * 1000 * 1000;
    int gop_size = 60;
};

SoftwareEncodeTuning buildSoftwareEncodeTuning(const VideoOutput::Config& config,
                                               bool is_network,
                                               int width,
                                               int height,
                                               double fps) {
    SoftwareEncodeTuning tuning;
    tuning.quality_mode = normalizeQualityMode(config.qualityMode);

    double bpp = 0.16;
    int64_t min_bitrate = 4LL * 1000 * 1000;
    int64_t max_bitrate = 20LL * 1000 * 1000;

    if (tuning.quality_mode == "speed") {
        tuning.preset = "veryfast";
        tuning.crf = 22;
        tuning.low_latency = true;
        bpp = is_network ? 0.10 : 0.12;
        min_bitrate = is_network ? 4LL * 1000 * 1000 : 3LL * 1000 * 1000;
        max_bitrate = is_network ? 12LL * 1000 * 1000 : 16LL * 1000 * 1000;
    } else if (tuning.quality_mode == "fidelity") {
        tuning.preset = "slow";
        tuning.crf = 10;
        tuning.low_latency = false;
        bpp = is_network ? 0.38 : 0.34;
        min_bitrate = is_network ? 20LL * 1000 * 1000 : 10LL * 1000 * 1000;
        max_bitrate = is_network ? 50LL * 1000 * 1000 : 30LL * 1000 * 1000;
    } else if (tuning.quality_mode == "source") {
        tuning.preset = "medium";
        tuning.crf = 14;
        tuning.low_latency = false;
        bpp = is_network ? 0.24 : 0.22;
        min_bitrate = is_network ? 12LL * 1000 * 1000 : 6LL * 1000 * 1000;
        max_bitrate = is_network ? 30LL * 1000 * 1000 : 24LL * 1000 * 1000;
    } else {
        tuning.preset = "fast";
        tuning.crf = 18;
        tuning.low_latency = false;
        bpp = is_network ? 0.16 : 0.18;
        min_bitrate = is_network ? 8LL * 1000 * 1000 : 4LL * 1000 * 1000;
        max_bitrate = is_network ? 20LL * 1000 * 1000 : 20LL * 1000 * 1000;
    }

    tuning.bit_rate = static_cast<int64_t>(width) * height * fps * bpp;
    tuning.bit_rate = std::max(min_bitrate, std::min(tuning.bit_rate, max_bitrate));

    if (config.bitrateMbps > 0.0) {
        tuning.bit_rate = static_cast<int64_t>(std::llround(config.bitrateMbps * 1000.0 * 1000.0));
    }
    tuning.rc_max_rate = tuning.bit_rate;
    tuning.rc_buffer_size = tuning.bit_rate * (tuning.low_latency ? 1 : (tuning.quality_mode == "fidelity" ? 3 : 2));

    if (config.crf >= 0) {
        tuning.crf = config.crf;
    }
    if (!config.preset.empty()) {
        tuning.preset = config.preset;
    }
    if (config.lowLatency) {
        tuning.low_latency = true;
        if (tuning.preset == "medium") {
            tuning.preset = "fast";
        }
    }

    int fps_int = std::max(1, static_cast<int>(std::round(fps)));
    tuning.gop_size = tuning.low_latency ? fps_int : std::max(fps_int, fps_int * 2);
    return tuning;
}

}

struct VideoOutput::RKMPPWriter {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVStream* stream = nullptr;
    SwsContext* sws_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    // RTSP+H264：h264_rkmpp 默认 AVCC（长度前缀）输出，rtp muxer 对它的 NAL 边界
    // 误判，产生非法 FU-A 分片（mediamtx 报 "invalid FU-A packet (non-starting)"，
    // 拉流端花屏/闪烁）。显式挂 h264_mp4toannexb 转 Annex-B 修复。
    AVBSFContext* h264_annexb_bsf = nullptr;
    int64_t pts = 0;
    int64_t last_muxed_dts = AV_NOPTS_VALUE;
    int64_t last_muxed_pts = AV_NOPTS_VALUE;
    int width = 0;
    int height = 0;
    bool opened = false;
    std::atomic<bool> interrupt_enabled{false};
    std::chrono::steady_clock::time_point io_deadline;
    int open_timeout_ms = 3000;
    int write_timeout_ms = 2000;

    static int interruptCallback(void* opaque) {
        auto* self = static_cast<RKMPPWriter*>(opaque);
        if (!self) return 0;
        if (!self->interrupt_enabled.load()) return 0;
        return std::chrono::steady_clock::now() > self->io_deadline ? 1 : 0;
    }

    void normalizePacketTimestamps(AVPacket* pkt) const {
        if (!pkt) return;
        if (pkt->pts == AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE) {
            pkt->pts = pkt->dts;
        }
        if (pkt->dts == AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE) {
            pkt->dts = pkt->pts;
        }
        if (pkt->pts == AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE) {
            pkt->pts = last_muxed_pts == AV_NOPTS_VALUE ? 0 : (last_muxed_pts + 1);
            pkt->dts = last_muxed_dts == AV_NOPTS_VALUE ? pkt->pts : (last_muxed_dts + 1);
        }

        if (last_muxed_dts != AV_NOPTS_VALUE && pkt->dts <= last_muxed_dts) {
            pkt->dts = last_muxed_dts + 1;
        }
        if (pkt->pts < pkt->dts) {
            pkt->pts = pkt->dts;
        }
    }

    // 把编码输出包写入容器（RTSP+H264 时先过 h264_mp4toannexb）。成功返回 true，
    // 失败返回 false（包已被 unref，不可再使用）。
    bool writePacket(AVPacket* pkt) {
        av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
        normalizePacketTimestamps(pkt);
        pkt->stream_index = stream->index;
        io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(write_timeout_ms);
        int wret = av_interleaved_write_frame(fmt_ctx, pkt);
        if (wret >= 0) {
            last_muxed_dts = pkt->dts;
            last_muxed_pts = pkt->pts;
        }
        av_packet_unref(pkt);
        if (wret < 0) {
            std::cerr << "[VideoOutput] FFmpeg write frame failed: " << avErrorToString(wret) << std::endl;
            return false;
        }
        return true;
    }

    bool muxPacket(AVPacket* pkt) {
        if (!h264_annexb_bsf) {
            return writePacket(pkt);
        }
        // av_bsf_send_packet 转移 pkt 引用（pkt 随即置空，可被下一轮复用）
        if (av_bsf_send_packet(h264_annexb_bsf, pkt) < 0) {
            av_packet_unref(pkt);
            return false;
        }
        int bret = 0;
        while ((bret = av_bsf_receive_packet(h264_annexb_bsf, pkt)) == 0) {
            if (!writePacket(pkt)) {
                return false;
            }
        }
        return bret == AVERROR(EAGAIN) || bret == AVERROR_EOF || bret == 0;
    }

    bool open(const VideoOutput::Config& config) {
        close();
        width = config.windowWidth;
        height = config.windowHeight;
        if (width <= 0 || height <= 0 || config.outputPath.empty()) {
            return false;
        }

        std::string output_url = normalizeSrtPublishUrl(forceRtmpPublishUrl(config.outputPath));

        if (isNetworkUrl(output_url)) {
            static std::once_flag network_init_flag;
            std::call_once(network_init_flag, []() { avformat_network_init(); });
        }

        double fps = config.targetFPS > 0.0 ? config.targetFPS : 30.0;
        int fps_int = static_cast<int>(std::round(fps));
        if (fps_int <= 0) fps_int = 30;

        const char* format_name = getOutputFormatName(output_url);
        int ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, format_name, output_url.c_str());
        if (ret < 0 || !fmt_ctx) {
            if (ret < 0) {
                std::cerr << "[VideoOutput] FFmpeg output context failed: " << avErrorToString(ret) << std::endl;
            }
            return false;
        }

        // 输出编码：RK_PIPE_OUTPUT_CODEC=h264|hevc（默认 h264）。
        // H.264/H.265 均优先走 RK3588 VPU 硬编（h264_rkmpp/hevc_rkmpp），失败回退软编。
        const char* codec_env = std::getenv("RK_PIPE_OUTPUT_CODEC");
        const bool want_hevc = codec_env && *codec_env && std::string(codec_env) == "hevc";
        const AVCodec* codec = nullptr;
        if (config.useRkMpp) {
            codec = avcodec_find_encoder_by_name(want_hevc ? "hevc_rkmpp" : "h264_rkmpp");
        }
        if (!codec) {
            codec = want_hevc ? avcodec_find_encoder(AV_CODEC_ID_HEVC)
                              : avcodec_find_encoder_by_name("libx264");
        }
        if (!codec) {
            codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (!codec) {
            std::cerr << "[VideoOutput] FFmpeg encoder not found" << std::endl;
            close();
            return false;
        }

        stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!stream) {
            close();
            return false;
        }

        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            close();
            return false;
        }

        codec_ctx->codec_id = codec->id;
        codec_ctx->width = width;
        codec_ctx->height = height;
        codec_ctx->time_base = av_make_q(1, fps_int);
        codec_ctx->framerate = av_make_q(fps_int, 1);
        codec_ctx->pix_fmt = config.useRkMpp ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P;
        codec_ctx->max_b_frames = 0;

        bool is_network = isNetworkUrl(config.outputPath);
        SoftwareEncodeTuning software_tuning;
        if (!config.useRkMpp) {
            software_tuning = buildSoftwareEncodeTuning(config, is_network, width, height, fps);
            codec_ctx->bit_rate = software_tuning.bit_rate;
            codec_ctx->rc_max_rate = software_tuning.rc_max_rate;
            codec_ctx->rc_buffer_size = software_tuning.rc_buffer_size;
            codec_ctx->gop_size = software_tuning.gop_size;
        } else {
            // 画质档位复用软件编码的 bitrate 推算（bpp 按 quality mode 分档）
            software_tuning = buildSoftwareEncodeTuning(config, is_network, width, height, fps);
            codec_ctx->bit_rate = software_tuning.bit_rate;
            if (is_network) {
                if (codec_ctx->bit_rate < 8000000) codec_ctx->bit_rate = 8000000;
                if (codec_ctx->bit_rate > 20000000) codec_ctx->bit_rate = 20000000;
                codec_ctx->rc_max_rate = codec_ctx->bit_rate;
                codec_ctx->rc_buffer_size = codec_ctx->bit_rate * 2;
            } else {
                if (codec_ctx->bit_rate < 4000000) codec_ctx->bit_rate = 4000000;
                if (codec_ctx->bit_rate > 20000000) codec_ctx->bit_rate = 20000000;
            }
            codec_ctx->gop_size = software_tuning.gop_size;
        }

        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        if (config.useRkMpp) {
            // rkmpp 把 SPS/PPS 写入 extradata（AVCC）：FLV 封装进 avcC 头、
            // TS/RTSP 由 h264_mp4toannexb 自动转 Annex B，保证拉流端秒开不花屏。
            codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        AVDictionary* codec_opts = nullptr;
        if (config.useRkMpp) {
            // 默认：网络推流 CBR（码率稳定、低延迟），本地文件 VBR（保画质）。
            // 可用 RK_PIPE_RKMPP_RC_MODE / _SURFACES / _PROFILE / _LEVEL 覆盖。
            const char* rc_mode_env = std::getenv("RK_PIPE_RKMPP_RC_MODE");
            std::string rc_mode = (rc_mode_env && *rc_mode_env)
                                      ? std::string(rc_mode_env)
                                      : (is_network ? "CBR" : "VBR");
            av_dict_set(&codec_opts, "rc_mode", rc_mode.c_str(), 0);
            const char* surfaces_env = std::getenv("RK_PIPE_RKMPP_SURFACES");
            if (surfaces_env && *surfaces_env) {
                av_dict_set(&codec_opts, "surfaces", surfaces_env, 0);
            }
            if (!want_hevc) {
                // hevc_rkmpp 仅支持 main profile，H.264 才应用 profile 覆盖
                const char* profile_env = std::getenv("RK_PIPE_RKMPP_PROFILE");
                if (profile_env && *profile_env) {
                    av_dict_set(&codec_opts, "profile", profile_env, 0);
                }
            }
            const char* level_env = std::getenv("RK_PIPE_RKMPP_LEVEL");
            if (level_env && *level_env) {
                av_dict_set(&codec_opts, "level", level_env, 0);
            }
            std::cout << "[VideoOutput] " << (want_hevc ? "hevc_rkmpp" : "h264_rkmpp")
                      << " rc_mode=" << rc_mode
                      << " bitrate=" << (codec_ctx->bit_rate / 1000000.0) << "Mbps"
                      << " gop=" << codec_ctx->gop_size
                      << " global_header=1"
                      << (rc_mode_env ? " (env RK_PIPE_RKMPP_RC_MODE)" : "")
                      << std::endl;
        } else if (codec->name && std::strcmp(codec->name, "libx264") == 0) {
            av_dict_set(&codec_opts, "preset", software_tuning.preset.c_str(), 0);
            av_dict_set(&codec_opts, "crf", std::to_string(software_tuning.crf).c_str(), 0);
            av_dict_set(&codec_opts, "profile", "high", 0);
            av_dict_set(&codec_opts, "repeat-headers", "1", 0);
            if (software_tuning.low_latency) {
                av_dict_set(&codec_opts, "tune", "zerolatency", 0);
            }
            std::cout << "[VideoOutput] libx264 tune mode=" << software_tuning.quality_mode
                      << " preset=" << software_tuning.preset
                      << " crf=" << software_tuning.crf
                      << " bitrate=" << (software_tuning.bit_rate / 1000000.0) << "Mbps"
                      << " low_latency=" << (software_tuning.low_latency ? "1" : "0")
                      << std::endl;
        }

        ret = avcodec_open2(codec_ctx, codec, &codec_opts);
        av_dict_free(&codec_opts);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg encoder open failed: " << avErrorToString(ret) << std::endl;
            close();
            return false;
        }

        ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg codec parameters failed: " << avErrorToString(ret) << std::endl;
            close();
            return false;
        }

        stream->time_base = codec_ctx->time_base;

        if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            interrupt_enabled.store(true);
            io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(open_timeout_ms);
            fmt_ctx->interrupt_callback.callback = &RKMPPWriter::interruptCallback;
            fmt_ctx->interrupt_callback.opaque = this;

            AVDictionary* io_opts = nullptr;
            if (isRtmpUrl(output_url)) {
                // RTMP 直播推流必须显式 rtmp_live=1，否则部分服务器会等流结束才发布
                av_dict_set(&io_opts, "rtmp_live", "1", 0);
            }
            ret = avio_open2(&fmt_ctx->pb, output_url.c_str(), AVIO_FLAG_WRITE, &fmt_ctx->interrupt_callback, &io_opts);
            av_dict_free(&io_opts);
            if (ret < 0) {
                std::cerr << "[VideoOutput] FFmpeg open output failed: " << avErrorToString(ret) << std::endl;
                close();
                return false;
            }
        }

        io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(open_timeout_ms);
        AVDictionary* header_opts = nullptr;
        if (fmt_ctx->oformat->name && std::strcmp(fmt_ctx->oformat->name, "flv") == 0) {
            // FLV 直播：禁止在 trailer 回写 duration/filesize（不可 seek，写坏会中断流）
            av_dict_set(&header_opts, "flvflags", "no_duration_filesize", 0);
        }

        ret = avformat_write_header(fmt_ctx, &header_opts);
        av_dict_free(&header_opts);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg write header failed: " << avErrorToString(ret) << std::endl;
            close();
            return false;
        }

        // RTSP(RTP muxer)+H264：显式转 Annex-B，避免 h264_rkmpp 的 AVCC 包被
        // rtp muxer 错误分片成非法 FU-A（板端实测 mediamtx 大量 "invalid FU-A
        // packet (non-starting)" 处理错误，拉流端花屏/闪烁）。RTMP/FLV 保持 AVCC。
        if (config.useRkMpp && !want_hevc && format_name &&
            (std::strcmp(format_name, "rtsp") == 0 || std::strcmp(format_name, "rtsps") == 0)) {
            const AVBitStreamFilter* filter = av_bsf_get_by_name("h264_mp4toannexb");
            if (filter) {
                if (av_bsf_alloc(filter, &h264_annexb_bsf) == 0 && stream->codecpar) {
                    avcodec_parameters_copy(h264_annexb_bsf->par_in, stream->codecpar);
                    if (av_bsf_init(h264_annexb_bsf) < 0) {
                        av_bsf_free(&h264_annexb_bsf);
                    }
                } else {
                    av_bsf_free(&h264_annexb_bsf);
                }
            }
            if (h264_annexb_bsf) {
                std::cout << "[VideoOutput] RTSP H264: applied h264_mp4toannexb (fixes invalid FU-A)" << std::endl;
            }
        }

        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet) {
            close();
            return false;
        }
        frame->format = codec_ctx->pix_fmt;
        frame->width = width;
        frame->height = height;
        ret = av_frame_get_buffer(frame, 32);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg frame buffer failed: " << avErrorToString(ret) << std::endl;
            close();
            return false;
        }

        sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGR24,
                                 width, height, codec_ctx->pix_fmt,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);

        // Ensure we handle NV12 input correctly if sws_ctx is used for it (though writeNV12 uses manual copy currently)
        // If we want to support scaling/conversion from NV12 input via sws_scale later, we would need a separate context or reconfiguration.
        // For now, the manual copy in writeNV12 handles NV12/NV21 -> NV12 pass-through correctly.

        if (!sws_ctx) {
            std::cerr << "[VideoOutput] FFmpeg sws context failed" << std::endl;
            close();
            return false;
        }

        opened = true;
        pts = 0;
        last_muxed_dts = AV_NOPTS_VALUE;
        last_muxed_pts = AV_NOPTS_VALUE;
        return true;
    }

    bool write(const cv::Mat& input) {
        if (!opened || !codec_ctx || !frame || !packet || !sws_ctx) return false;

        cv::Mat src = input;
        cv::Mat resized;
        if (src.cols != width || src.rows != height) {
            cv::resize(src, resized, cv::Size(width, height));
            src = resized;
        }

        int ret = av_frame_make_writable(frame);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg frame writable failed: " << avErrorToString(ret) << std::endl;
            return false;
        }

        const uint8_t* src_data[4] = { src.data, nullptr, nullptr, nullptr };
        int src_linesize[4] = { static_cast<int>(src.step[0]), 0, 0, 0 };
        sws_scale(sws_ctx, src_data, src_linesize, 0, height, frame->data, frame->linesize);
        frame->pts = pts++;

        io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(write_timeout_ms);
        ret = avcodec_send_frame(codec_ctx, frame);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg send frame failed: " << avErrorToString(ret) << std::endl;
            return false;
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                std::cerr << "[VideoOutput] FFmpeg receive packet failed: " << avErrorToString(ret) << std::endl;
                return false;
            }
            if (!muxPacket(packet)) {
                return false;
            }
        }

        return true;
    }

    bool writeNV12(const image_buffer_t& input) {
        if (!opened || !codec_ctx || !frame || !packet) return false;
        if (input.virt_addr == nullptr) return false;
        if (input.width != width || input.height != height) return false;

        int ret = av_frame_make_writable(frame);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg frame writable failed: " << avErrorToString(ret) << std::endl;
            return false;
        }

        int src_stride = input.width_stride > 0 ? input.width_stride : input.width;
        int dst_stride_y = frame->linesize[0];
        int dst_stride_uv = frame->linesize[1];

        const uint8_t* src_y = input.virt_addr;
        const uint8_t* src_uv = input.virt_addr + src_stride * (input.height_stride > 0 ? input.height_stride : input.height);

        for (int y = 0; y < height; ++y) {
            memcpy(frame->data[0] + y * dst_stride_y, src_y + y * src_stride, width);
        }

        bool is_nv21 = input.format == IMAGE_FORMAT_YUV420SP_NV21;
        for (int y = 0; y < height / 2; ++y) {
            uint8_t* dst = frame->data[1] + y * dst_stride_uv;
            const uint8_t* src = src_uv + y * src_stride;
            if (!is_nv21) {
                memcpy(dst, src, width);
            } else {
                for (int x = 0; x < width; x += 2) {
                    dst[x] = src[x + 1];
                    dst[x + 1] = src[x];
                }
            }
        }

        frame->pts = pts++;

        io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(write_timeout_ms);
        ret = avcodec_send_frame(codec_ctx, frame);
        if (ret < 0) {
            std::cerr << "[VideoOutput] FFmpeg send frame failed: " << avErrorToString(ret) << std::endl;
            return false;
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                std::cerr << "[VideoOutput] FFmpeg receive packet failed: " << avErrorToString(ret) << std::endl;
                return false;
            }
            av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
            normalizePacketTimestamps(packet);
            packet->stream_index = stream->index;
            io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(write_timeout_ms);
            int wret = av_interleaved_write_frame(fmt_ctx, packet);
            if (wret >= 0) {
                last_muxed_dts = packet->dts;
                last_muxed_pts = packet->pts;
            }
            av_packet_unref(packet);
            if (wret < 0) {
                std::cerr << "[VideoOutput] FFmpeg write frame failed: " << avErrorToString(wret) << std::endl;
                return false;
            }
        }

        return true;
    }

    void close() {
        if (opened && codec_ctx && fmt_ctx) {
            avcodec_send_frame(codec_ctx, nullptr);
            while (avcodec_receive_packet(codec_ctx, packet) == 0) {
                muxPacket(packet);
            }
            io_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(write_timeout_ms);
            av_write_trailer(fmt_ctx);
        }

        if (h264_annexb_bsf) {
            av_bsf_free(&h264_annexb_bsf);
        }
        if (packet) {
            av_packet_free(&packet);
        }
        if (frame) {
            av_frame_free(&frame);
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        if (fmt_ctx) {
            if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && fmt_ctx->pb) {
                avio_closep(&fmt_ctx->pb);
            }
            avformat_free_context(fmt_ctx);
            fmt_ctx = nullptr;
        }
        stream = nullptr;
        opened = false;
        interrupt_enabled.store(false);
        pts = 0;
        last_muxed_dts = AV_NOPTS_VALUE;
        last_muxed_pts = AV_NOPTS_VALUE;
    }
};

int VideoOutput::calculateOptimalQueueSize(int width, int height, int maxMemoryMB)
{
    size_t frameSize = estimateFrameMemory(width, height);
    if (frameSize == 0) return 20;

    int maxFrames = static_cast<int>((maxMemoryMB * 1024 * 1024) / frameSize);
    maxFrames = std::max(2, std::min(maxFrames, 20));

    return maxFrames;
}

size_t VideoOutput::estimateFrameMemory(int width, int height)
{
    if (width <= 0 || height <= 0) return 0;
    return static_cast<size_t>(width) * height * 3;
}

VideoOutput::VideoOutput(Config config)
    : config_(std::move(config))
{
}

VideoOutput::~VideoOutput()
{
    stop();
}

bool VideoOutput::init()
{
    if (initialized_) return true;

    if (config_.maxQueueSize <= 0) {
        config_.maxQueueSize = calculateOptimalQueueSize(
            config_.windowWidth, config_.windowHeight, config_.maxMemoryMB);
    }

    displayEnabled_ = config_.enableDisplay;
    writeEnabled_ = config_.enableWrite && !config_.outputPath.empty();
    if (config_.dropFramesOnOverflow >= 0) {
        // 显式背压策略优先（0=阻塞背压，1=队满丢帧）
        dropFramesOnOverflow_ = config_.dropFramesOnOverflow != 0;
    } else {
        dropFramesOnOverflow_ = writeEnabled_ && isNetworkUrl(config_.outputPath);
    }

    if (writeEnabled_) {
        const bool networkOutput = isNetworkUrl(config_.outputPath);

        std::cout << "[VideoOutput] Writer mode: " << (config_.useRkMpp ? "RKMPP" : "FFmpeg/OpenCV fallback")
                  << ", Output: " << config_.outputPath;
        if (!config_.useRkMpp && networkOutput) {
            std::cout << " (RKMPP disabled for network streaming stability)";
        }
        std::cout << std::endl;

        const bool preferAvWriter = config_.useRkMpp || networkOutput;
        if (preferAvWriter) {
            auto openAvWriter = [this]() {
                rkmppWriter_ = std::make_unique<RKMPPWriter>();
                return rkmppWriter_->open(config_);
            };

            if (!openAvWriter()) {
                rkmppWriter_.reset();
                if (config_.useRkMpp) {
                    std::cerr << "[VideoOutput] RKMPP writer init failed, fallback to software FFmpeg: "
                              << config_.outputPath << std::endl;
                    config_.useRkMpp = false;
                    if (!openAvWriter()) {
                        rkmppWriter_.reset();
                    }
                }
            }

            if (!rkmppWriter_ && networkOutput) {
                std::cerr << "[VideoOutput] Failed to open libav network writer: "
                          << config_.outputPath << std::endl;
                writeEnabled_ = false;
            }
        }
        if (!rkmppWriter_ && !config_.useRkMpp && writeEnabled_) {
            auto codecs = buildCodecCandidates(config_.outputPath, config_.fourcc);
            auto tryOpen = [&](int apiPreference) {
                for (int fcc : codecs) {
                    videoWriter_.open(config_.outputPath, apiPreference, fcc, config_.targetFPS,
                                      cv::Size(config_.windowWidth, config_.windowHeight), true);
                    if (videoWriter_.isOpened()) {
                        config_.fourcc = fcc;
                        break;
                    }
                }
            };
            if (networkOutput) {
                tryOpen(cv::CAP_FFMPEG);
            } else {
                tryOpen(cv::CAP_ANY);
                if (!videoWriter_.isOpened()) {
                    tryOpen(cv::CAP_FFMPEG);
                }
            }
            if (!videoWriter_.isOpened()) {
                std::cerr << "[VideoOutput] Failed to open video writer: " << config_.outputPath << std::endl;
                writeEnabled_ = false;
            } else {
                videoWriter_.set(cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
                videoWriter_.set(cv::VIDEOWRITER_PROP_QUALITY, config_.quality);
            }
        }
    }

    if (displayEnabled_) {
        cv::namedWindow(config_.windowName, cv::WINDOW_NORMAL);
        if (config_.windowWidth > 0 && config_.windowHeight > 0) {
            cv::resizeWindow(config_.windowName, config_.windowWidth, config_.windowHeight);
        }
    }

    running_ = true;
    if (config_.displayFPS <= 0.0) {
        config_.displayFPS = config_.targetFPS;
    }
    if (config_.writeFPS <= 0.0) {
        config_.writeFPS = config_.targetFPS;
    }
    targetDisplayFPS_ = std::max(1.0, std::min(config_.displayFPS, 120.0));
    targetWriteFPS_ = std::max(1.0, std::min(config_.writeFPS, 120.0));
    displayFpsCalcStart_ = std::chrono::steady_clock::now();
    initialized_ = true;

    if (displayEnabled_) {
        displayThread_ = std::thread(&VideoOutput::displayLoop, this);
    }

    if (writeEnabled_) {
        writeThread_ = std::thread(&VideoOutput::writeLoop, this);
    }

    size_t estimatedFrameMem = estimateFrameMemory(config_.windowWidth, config_.windowHeight);
    size_t maxQueueMem = estimatedFrameMem * config_.maxQueueSize;

    std::cout << "[VideoOutput] Initialized - Display FPS: " << config_.displayFPS
              << ", Stream FPS: " << config_.writeFPS
              << ", Queue size: " << config_.maxQueueSize
              << ", Max memory: " << (maxQueueMem / 1024 / 1024) << "MB" << std::endl;

    return true;
}

void VideoOutput::pushFrame(cv::Mat frame)
{
    pushFrame(-1, std::move(frame));
}

void VideoOutput::pushFrame(int frameIndex, cv::Mat frame)
{
    if (!running_ || frame.empty()) return;

    if (displayEnabled_) {
        cv::Mat displayFrame = frame;
        {
            std::lock_guard<std::mutex> lock(displayMutex_);
            if (static_cast<int>(displayQueue_.size()) >= config_.maxQueueSize) {
                displayQueue_.pop();
            }
            if (!displayFrame.empty()) {
                displayQueue_.push(QueuedFrame{frameIndex, std::move(displayFrame)});
            }
        }
        cvDisplay_.notify_one();
    }

    if (writeEnabled_) {
        std::unique_lock<std::mutex> lock(writeMutex_);
        if (!dropFramesOnOverflow_) {
            cvWrite_.wait(lock, [this]() {
                if (!running_) return true;
                return static_cast<int>(writeQueue_.size() + writeQueueYuv_.size()) < config_.maxQueueSize;
            });
            if (!running_) return;
        } else if (static_cast<int>(writeQueue_.size()) >= config_.maxQueueSize) {
            writeQueue_.pop();
            dropped_frames_.fetch_add(1);
        }
        if (!frame.empty()) {
            writeQueue_.push(QueuedFrame{frameIndex, std::move(frame)});
        }
        cvWrite_.notify_one();
    }
}

bool VideoOutput::pushFrame(int frameIndex, const image_buffer_t& frame)
{
    if (!running_ || !writeEnabled_ || !rkmppWriter_) return false;
    if (frame.virt_addr == nullptr) return false;
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) return false;
    if (frameIndex >= 0 && frameIndex <= lastWrittenIndex_.load()) return false;

    if (displayEnabled_) {
        cv::Mat displayFrame;
        if (convertYuv420spToBgrForDisplay(frame, &displayFrame)) {
            std::lock_guard<std::mutex> lock(displayMutex_);
            if (static_cast<int>(displayQueue_.size()) >= config_.maxQueueSize) {
                displayQueue_.pop();
            }
            displayQueue_.push(QueuedFrame{frameIndex, std::move(displayFrame)});
            cvDisplay_.notify_one();
        }
    }

    if (!config_.useRkMpp) {
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(writeMutex_);
        if (!dropFramesOnOverflow_) {
            cvWrite_.wait(lock, [this]() {
                if (!running_) return true;
                return static_cast<int>(writeQueue_.size() + writeQueueYuv_.size()) < config_.maxQueueSize;
            });
            if (!running_) return false;
        } else if (static_cast<int>(writeQueueYuv_.size()) >= config_.maxQueueSize) {
            auto dropped = std::move(writeQueueYuv_.front());
            writeQueueYuv_.pop();
            VideoReader::releaseFrame(dropped.frame);
            dropped_frames_.fetch_add(1);
        }
        writeQueueYuv_.push(QueuedFrameYuv{frameIndex, frame});
    }
    cvWrite_.notify_one();
    return true;
}

void VideoOutput::setDisplayEnabled(bool enabled)
{
    displayEnabled_ = enabled;
}

void VideoOutput::setWriteEnabled(bool enabled)
{
    writeEnabled_ = enabled;
}

void VideoOutput::setTargetFPS(double fps)
{
    targetDisplayFPS_ = std::max(1.0, std::min(fps, 120.0));
    targetWriteFPS_ = std::max(1.0, std::min(fps, 120.0));
}

double VideoOutput::getCurrentFPS() const
{
    return writeEnabled_ ? currentWriteFPS_.load() : currentDisplayFPS_.load();
}

double VideoOutput::getDisplayFPS() const
{
    return currentDisplayFPS_.load();
}

double VideoOutput::getWriteFPS() const
{
    return currentWriteFPS_.load();
}

bool VideoOutput::isRunning() const
{
    return running_.load();
}

void VideoOutput::stop()
{
    if (!running_.load()) {
        return;
    }

    if (writeEnabled_) {
        auto deadline = std::chrono::steady_clock::now() + (dropFramesOnOverflow_ ? std::chrono::seconds(5) : std::chrono::seconds(60));
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(writeMutex_);
                if (writeQueue_.empty() && writeQueueYuv_.empty()) {
                    break;
                }
            }
            cvWrite_.notify_one();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    if (!running_.exchange(false)) {
        return;
    }

    cvDisplay_.notify_all();
    cvWrite_.notify_all();

    if (displayThread_.joinable()) {
        displayThread_.join();
    }

    if (writeThread_.joinable()) {
        writeThread_.join();
    }

    if (displayEnabled_) {
        cv::destroyWindow(config_.windowName);
    }

    if (videoWriter_.isOpened()) {
        videoWriter_.release();
    }

    if (rkmppWriter_) {
        rkmppWriter_->close();
        rkmppWriter_.reset();
    }

    {
        std::lock_guard<std::mutex> lock(displayMutex_);
        while (!displayQueue_.empty()) {
            displayQueue_.pop();
        }
    }

    {
        std::lock_guard<std::mutex> lock(writeMutex_);
        while (!writeQueue_.empty()) {
            writeQueue_.pop();
        }
        while (!writeQueueYuv_.empty()) {
            auto queued = std::move(writeQueueYuv_.front());
            writeQueueYuv_.pop();
            VideoReader::releaseFrame(queued.frame);
        }
    }
}

int VideoOutput::getQueuedFrameCount() const
{
    std::lock_guard<std::mutex> lock(displayMutex_);
    return static_cast<int>(displayQueue_.size());
}

size_t VideoOutput::getCurrentMemoryUsage() const
{
    return currentMemoryUsage_.load();
}

void VideoOutput::displayLoop()
{
    auto nextFrameTime = std::chrono::steady_clock::now();

    while (running_) {
        QueuedFrame queued;
        bool backlog = false;

        {
            std::unique_lock<std::mutex> lock(displayMutex_);
            cvDisplay_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !displayQueue_.empty() || !running_; });

            if (!running_) break;

            if (!displayQueue_.empty()) {
                if (displayQueue_.size() > 1) {
                    backlog = true;
                    while (displayQueue_.size() > 1) {
                        displayQueue_.pop();
                    }
                }
                queued = std::move(displayQueue_.front());
                displayQueue_.pop();
            }
        }

        if (queued.frame.empty()) continue;
        int lastIndex = lastDisplayedIndex_.load();
        if (queued.index >= 0 && queued.index <= lastIndex) {
            continue;
        }
        if (queued.index >= 0) {
            lastDisplayedIndex_.store(queued.index);
        }

        cv::Mat frame = std::move(queued.frame);

        currentMemoryUsage_ = frame.total() * frame.elemSize();

        if (displayEnabled_) {
            cv::imshow(config_.windowName, frame);
            cv::waitKey(1);
        }

        if (writeEnabled_) {
            currentMemoryUsage_ = frame.total() * frame.elemSize();
        }

        if (!backlog) {
            double fps = targetDisplayFPS_.load();
            if (fps > 0) {
                auto frameDuration = std::chrono::milliseconds(static_cast<int>(1000.0 / fps));
                nextFrameTime = std::chrono::steady_clock::now() + frameDuration;

                if (nextFrameTime > std::chrono::steady_clock::now()) {
                    std::this_thread::sleep_until(nextFrameTime);
                }
            }
        }

        displayFrameCount_++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - displayFpsCalcStart_).count();
        if (elapsed >= 1000) {
            currentDisplayFPS_ = displayFrameCount_ * 1000.0 / elapsed;
            displayFrameCount_ = 0;
            displayFpsCalcStart_ = now;
        }
    }
}

void VideoOutput::writeLoop()
{
    static const bool debug_enabled = []() {
        const char* v = std::getenv("RK_PIPE_DEBUG_OUTPUT");
        return v && *v && std::string(v) != "0";
    }();

    static const int slow_write_ms = []() {
        const char* v = std::getenv("RK_PIPE_OUTPUT_SLOW_WRITE_MS");
        if (!v || !*v) return 50;
        int out = std::atoi(v);
        return out <= 0 ? 50 : out;
    }();

    static const int stall_write_ms = []() {
        const char* v = std::getenv("RK_PIPE_OUTPUT_STALL_WRITE_MS");
        if (!v || !*v) return 200;
        int out = std::atoi(v);
        return out <= 0 ? 200 : out;
    }();

    auto nextFrameTime = std::chrono::steady_clock::now();
    int localFrameCount = 0;
    auto localFpsStart = std::chrono::steady_clock::now();
    while (running_) {
        QueuedFrame queuedFrame;
        QueuedFrameYuv queuedYuv;
        {
            std::unique_lock<std::mutex> lock(writeMutex_);
            cvWrite_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !writeQueue_.empty() || !writeQueueYuv_.empty() || !running_; });

            if (!running_) break;

            if (!writeQueueYuv_.empty() && !writeQueue_.empty()) {
                int yuvIndex = writeQueueYuv_.front().index;
                int rgbIndex = writeQueue_.front().index;
                if (rgbIndex >= 0 && (yuvIndex < 0 || rgbIndex <= yuvIndex)) {
                    queuedFrame = std::move(writeQueue_.front());
                    writeQueue_.pop();
                } else {
                    queuedYuv = std::move(writeQueueYuv_.front());
                    writeQueueYuv_.pop();
                }
                cvWrite_.notify_one();
            } else if (!writeQueueYuv_.empty()) {
                queuedYuv = std::move(writeQueueYuv_.front());
                writeQueueYuv_.pop();
                cvWrite_.notify_one();
            } else if (!writeQueue_.empty()) {
                queuedFrame = std::move(writeQueue_.front());
                writeQueue_.pop();
                cvWrite_.notify_one();
            }
        }

        bool wrote = false;
        if (queuedYuv.frame.virt_addr) {
            int lastIndex = lastWrittenIndex_.load();
            if (queuedYuv.index >= 0 && queuedYuv.index <= lastIndex) {
                VideoReader::releaseFrame(queuedYuv.frame);
                continue;
            }
            if (rkmppWriter_) {
                auto t0 = std::chrono::steady_clock::now();
                wrote = rkmppWriter_->writeNV12(queuedYuv.frame);
                if (debug_enabled) {
                    auto t1 = std::chrono::steady_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                    if (ms >= slow_write_ms) {
                        int wq = 0;
                        int yq = 0;
                        {
                            std::lock_guard<std::mutex> lock(writeMutex_);
                            wq = static_cast<int>(writeQueue_.size());
                            yq = static_cast<int>(writeQueueYuv_.size());
                        }
                        fprintf(stderr,
                                "[rk_pipe][output] writeNV12 %s cost=%lldms idx=%d q(rgb=%d yuv=%d) target=%.2f cur=%.2f drop=%d\n",
                                ms >= stall_write_ms ? "STALL" : "slow",
                                static_cast<long long>(ms),
                                queuedYuv.index,
                                wq,
                                yq,
                                targetWriteFPS_.load(),
                                currentWriteFPS_.load(),
                                dropFramesOnOverflow_ ? 1 : 0);
                    }
                }
            }
            VideoReader::releaseFrame(queuedYuv.frame);
            if (wrote && queuedYuv.index >= 0) {
                lastWrittenIndex_.store(queuedYuv.index);
            }
        } else if (!queuedFrame.frame.empty()) {
            int lastIndex = lastWrittenIndex_.load();
            if (queuedFrame.index >= 0 && queuedFrame.index <= lastIndex) {
                continue;
            }
            if (rkmppWriter_) {
                auto t0 = std::chrono::steady_clock::now();
                wrote = rkmppWriter_->write(queuedFrame.frame);
                if (debug_enabled) {
                    auto t1 = std::chrono::steady_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                    if (ms >= slow_write_ms) {
                        int wq = 0;
                        int yq = 0;
                        {
                            std::lock_guard<std::mutex> lock(writeMutex_);
                            wq = static_cast<int>(writeQueue_.size());
                            yq = static_cast<int>(writeQueueYuv_.size());
                        }
                        fprintf(stderr,
                                "[rk_pipe][output] writeBGR %s cost=%lldms idx=%d q(rgb=%d yuv=%d) target=%.2f cur=%.2f drop=%d\n",
                                ms >= stall_write_ms ? "STALL" : "slow",
                                static_cast<long long>(ms),
                                queuedFrame.index,
                                wq,
                                yq,
                                targetWriteFPS_.load(),
                                currentWriteFPS_.load(),
                                dropFramesOnOverflow_ ? 1 : 0);
                    }
                }
            } else if (videoWriter_.isOpened()) {
                videoWriter_.write(queuedFrame.frame);
                wrote = true;
            }
            if (wrote && queuedFrame.index >= 0) {
                lastWrittenIndex_.store(queuedFrame.index);
            }
        }

        if (!wrote) {
            continue;
        }
        written_frames_.fetch_add(1);

        if (dropFramesOnOverflow_) {
            double fps = targetWriteFPS_.load();
            if (fps > 0) {
                auto frameDuration = std::chrono::milliseconds(static_cast<int>(1000.0 / fps));
                nextFrameTime += frameDuration;
                if (nextFrameTime > std::chrono::steady_clock::now()) {
                    std::this_thread::sleep_until(nextFrameTime);
                } else {
                    nextFrameTime = std::chrono::steady_clock::now();
                }
            }
        }

        localFrameCount++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - localFpsStart).count();
        if (elapsed >= 1000) {
            currentWriteFPS_ = localFrameCount * 1000.0 / elapsed;
            localFrameCount = 0;
            localFpsStart = now;
        }
    }
}

std::unique_ptr<VideoOutput> VideoOutputFactory::create(OutputType type, VideoOutput::Config config)
{
    switch (type) {
        case OutputType::OPENCV:
            return std::make_unique<VideoOutput>(std::move(config));
        case OutputType::FFMPEG:
            config.useRkMpp = shouldUseRkMppForOutput(config.outputPath);
            return std::make_unique<VideoOutput>(std::move(config));
        case OutputType::GSTREAMER:
            return std::make_unique<VideoOutput>(std::move(config));
        case OutputType::RKMPP:
            config.useRkMpp = shouldUseRkMppForOutput(config.outputPath);
            return std::make_unique<VideoOutput>(std::move(config));
        default:
            return std::make_unique<VideoOutput>(std::move(config));
    }
}
