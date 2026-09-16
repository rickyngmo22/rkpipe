#include "llm/llm_config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <opencv2/opencv.hpp>

namespace {

bool readString(const cv::FileStorage& fs, const char* key, std::string& value) {
    const cv::FileNode node = fs[key];
    if (!node.empty()) {
        node >> value;
        return true;
    }
    return false;
}

bool readFloat(const cv::FileStorage& fs, const char* key, float& value) {
    const cv::FileNode node = fs[key];
    if (!node.empty()) {
        node >> value;
        return true;
    }
    return false;
}

bool readDouble(const cv::FileStorage& fs, const char* key, double& value) {
    const cv::FileNode node = fs[key];
    if (!node.empty()) {
        node >> value;
        return true;
    }
    return false;
}

bool readInt(const cv::FileStorage& fs, const char* key, int& value) {
    const cv::FileNode node = fs[key];
    if (!node.empty()) {
        node >> value;
        return true;
    }
    return false;
}

bool readBool(const cv::FileStorage& fs, const char* key, bool& value) {
    int int_value = value ? 1 : 0;
    if (readInt(fs, key, int_value)) {
        value = int_value != 0;
        return true;
    }
    return false;
}

}  // namespace

bool LlmConfig::loadFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    config_path = filename;

    readBool(fs, "llm_enabled", enabled);
    readString(fs, "llm_provider", provider);
    readFloat(fs, "llm_min_conf", min_conf);
    readDouble(fs, "llm_interval_s", interval_s);
    readBool(fs, "llm_submit_all", submit_all);
    readInt(fs, "llm_queue_size", queue_size);
    readInt(fs, "llm_max_results_keep", max_results_keep);
    readString(fs, "llm_results_path", results_path);
    readString(fs, "llm_stats_jsonl_path", stats_jsonl_path);
    readString(fs, "llm_model", model);
    readString(fs, "llm_endpoint", endpoint);
    readString(fs, "llm_api_key", api_key);
    readString(fs, "llm_api_key_env", api_key_env);
    readDouble(fs, "llm_timeout_s", timeout_s);
    readDouble(fs, "llm_temperature", temperature);
    readInt(fs, "llm_max_tokens", max_tokens);
    readBool(fs, "llm_send_image", send_image);
    readDouble(fs, "llm_image_max_side", image_max_side);
    readString(fs, "llm_label_path", label_path);

    // 基础钳位，避免配置笔误把队列/限速打崩
    if (queue_size < 1) queue_size = 1;
    if (queue_size > 64) queue_size = 64;
    if (interval_s < 0.0) interval_s = 0.0;
    if (max_results_keep < 1) max_results_keep = 1;
    if (timeout_s < 1.0) timeout_s = 1.0;
    if (max_tokens < 16) max_tokens = 16;
    if (image_max_side < 64.0) image_max_side = 64.0;
    return true;
}

std::vector<std::string> LlmConfig::loadLabels() const {
    std::vector<std::string> labels;
    if (label_path.empty()) {
        return labels;
    }
    std::ifstream in(label_path);
    if (!in.is_open()) {
        return labels;
    }
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

LlmConfig LlmConfig::fromDefaultLocation() {
    LlmConfig config;
    const char* env_path = std::getenv("RKPIPE_LLM_CONFIG");
    std::string path = env_path && *env_path ? env_path : "./llm.yaml";
    // 先探测存在性：无 llm.yaml 是常态（模块静默禁用），
    // 避免 FileStorage 每次启动都打 OpenCV 打不开文件的 ERROR 行
    std::ifstream probe(path);
    if (!probe.is_open()) {
        return LlmConfig{};
    }
    if (!config.loadFromFile(path)) {
        return LlmConfig{};  // 文件存在但解析失败：同样按禁用处理
    }
    return config;
}
