#pragma once

// 输出链路纯逻辑（G2 第二坡从 video_output.cc 提取，无 FFmpeg 依赖，CI 可测）：
//   URL 分类 / muxer 选择 / SRT 发布地址规范化 / rkmpp 启停开关 / 软编码参数推算。

#include <cstdint>
#include <string>

// 网络推流 URL（rtmp/rtmps/rtsp/rtsps/udp/srt）
bool isNetworkUrl(const std::string& path);
bool isRtmpUrl(const std::string& path);

// URL → 封装名：rtmp→flv / rtsp→rtsp / udp·srt→mpegts；本地文件返回 nullptr
const char* getOutputFormatName(const std::string& path);

// SRT 推流地址规范化：`srt://host:port/live/key` →
//   `srt://host:port?mode=caller&streamid=#!::r=live/key,m=publish`
// （MediaMTX 要求 streamid 指定流路径；已带 query 或无路径的 URL 原样返回）
std::string normalizeSrtPublishUrl(const std::string& path);

// 环境变量开关（1/true/yes/on，大小写不敏感）
bool envFlagEnabled(const char* name);

// 输出编码器选择：显式 DISABLE 优先，其次 FORCE，最后网络 URL 默认放行
bool shouldUseRkMppForOutput(const std::string& outputPath);

// 画质档位别名归一（speed/balanced/source/fidelity 及历史别名），未知值回退 balanced
std::string normalizeQualityMode(const std::string& value);

// 软编码参数推算的输入（从 VideoOutput::Config 摘取，解耦避免拉入整个头）
struct EncodeTuningParams {
    std::string qualityMode = "balanced";
    double bitrateMbps = 0.0;  // >0 时覆盖 bpp 推算
    int crf = -1;              // >=0 时覆盖档位 CRF
    std::string preset;        // 非空时覆盖档位 preset
    bool lowLatency = false;
};

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

// bpp 按档位×网络/本地分档推算 bitrate，钳到档位上下限，再套用显式覆盖
SoftwareEncodeTuning buildSoftwareEncodeTuning(const EncodeTuningParams& params,
                                               bool is_network,
                                               int width,
                                               int height,
                                               double fps);
