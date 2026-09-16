#include "io/output_tuning.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace {

std::string toLowerCopy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

}  // namespace

bool isNetworkUrl(const std::string& path) {
    const std::string lower = toLowerCopy(path);
    return lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("rtsp://", 0) == 0 || lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 || lower.rfind("srt://", 0) == 0;
}

bool isRtmpUrl(const std::string& path) {
    const std::string lower = toLowerCopy(path);
    return lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0;
}

const char* getOutputFormatName(const std::string& path) {
    const std::string lower = toLowerCopy(path);
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

std::string normalizeSrtPublishUrl(const std::string& path) {
    const std::string lower = toLowerCopy(path);
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

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    const std::string normalized = toLowerCopy(value);
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool shouldUseRkMppForOutput(const std::string& outputPath) {
    if (envFlagEnabled("RK_PIPE_DISABLE_RKMPP")) {
        return false;
    }
    if (envFlagEnabled("RK_PIPE_FORCE_RKMPP")) {
        return true;
    }
    return isNetworkUrl(outputPath);
}

std::string normalizeQualityMode(const std::string& value) {
    const std::string lower = toLowerCopy(value);
    if (lower.empty()) {
        return "balanced";
    }
    if (lower == "speed" || lower == "fast" || lower == "low_latency" ||
        lower == "low-latency") {
        return "speed";
    }
    if (lower == "fidelity" || lower == "visually_lossless" || lower == "visually-lossless") {
        return "fidelity";
    }
    if (lower == "source" || lower == "source_like" || lower == "source-like" ||
        lower == "quality" || lower == "high") {
        return "source";
    }
    return "balanced";
}

SoftwareEncodeTuning buildSoftwareEncodeTuning(const EncodeTuningParams& params,
                                               bool is_network,
                                               int width,
                                               int height,
                                               double fps) {
    SoftwareEncodeTuning tuning;
    tuning.quality_mode = normalizeQualityMode(params.qualityMode);

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

    if (params.bitrateMbps > 0.0) {
        tuning.bit_rate = static_cast<int64_t>(std::llround(params.bitrateMbps * 1000.0 * 1000.0));
    }
    tuning.rc_max_rate = tuning.bit_rate;
    tuning.rc_buffer_size =
        tuning.bit_rate * (tuning.low_latency ? 1 : (tuning.quality_mode == "fidelity" ? 3 : 2));

    if (params.crf >= 0) {
        tuning.crf = params.crf;
    }
    if (!params.preset.empty()) {
        tuning.preset = params.preset;
    }
    if (params.lowLatency) {
        tuning.low_latency = true;
        if (tuning.preset == "medium") {
            tuning.preset = "fast";
        }
    }

    const int fps_int = std::max(1, static_cast<int>(std::round(fps)));
    tuning.gop_size = tuning.low_latency ? fps_int : std::max(fps_int, fps_int * 2);
    return tuning;
}
