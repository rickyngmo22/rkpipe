// OpenAI 兼容 chat/completions HTTP provider：
//   - provider=cloud：云端多模态/文本模型（智谱 GLM、DashScope 及任意兼容网关）
//   - provider=rkllm：板端 RKLLM 本地服务（同样暴露 OpenAI 兼容协议，无需 API key）
// 带图像的请求按 GLM-4.5V / OpenAI vision 的 image_url(data URL) 约定组装 messages。
// 本文件仅在 CMake 探测到 libcurl 时参与编译；否则由 provider_mock.cc 降级。

#include <curl/curl.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <utility>

#include <nlohmann/json.hpp>

#include "llm/llm_provider.h"
#include "llm/llm_config.h"

namespace llm {

namespace {

constexpr const char* kDefaultRkllmEndpoint = "http://127.0.0.1:8080/v1/chat/completions";

std::string base64Encode(const unsigned char* data, std::size_t size) {
    static const char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (std::size_t i = 0; i < size; i += 3) {
        const unsigned int n = (data[i] << 16) | (i + 1 < size ? data[i + 1] << 8 : 0) |
                               (i + 2 < size ? data[i + 2] : 0);
        out += kTable[(n >> 18) & 63];
        out += kTable[(n >> 12) & 63];
        out += i + 1 < size ? kTable[(n >> 6) & 63] : '=';
        out += i + 2 < size ? kTable[n & 63] : '=';
    }
    return out;
}

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* body = static_cast<std::string*>(userdata);
    body->append(ptr, size * nmemb);
    return size * nmemb;
}

std::string resolveApiKey(const LlmConfig& config) {
    if (!config.api_key.empty()) {
        return config.api_key;
    }
    const char* env_key = config.api_key_env.empty() ? nullptr : std::getenv(config.api_key_env.c_str());
    return env_key ? env_key : "";
}

// curl_global_init/init 全局一次性；CURL easy handle 每请求新建（复检/对话并发频率低）
std::once_flag g_curl_once;

class OpenAiCompatProvider : public LlmProvider {
public:
    OpenAiCompatProvider(const LlmConfig& config, std::string default_endpoint, bool vision)
        : config_(config), default_endpoint_(std::move(default_endpoint)), vision_(vision) {}

    LlmResponse generate(const LlmRequest& request) override {
        LlmResponse response;
        const auto begin = std::chrono::steady_clock::now();

        std::call_once(g_curl_once, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });

        CURL* curl = curl_easy_init();
        if (!curl) {
            response.error = "curl_easy_init failed";
            return response;
        }

        const std::string body = buildRequestBody(request);
        const std::string endpoint = config_.endpoint.empty() ? default_endpoint_ : config_.endpoint;
        const std::string api_key = resolveApiKey(config_);
        std::string response_body;

        curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        if (!api_key.empty()) {
            headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
        }

        curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(config_.timeout_s * 1000.0));
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        const CURLcode code = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        response.elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

        if (code != CURLE_OK) {
            response.error = "HTTP request failed: " + std::string(curl_easy_strerror(code));
            return response;
        }
        if (http_code != 200) {
            response.error = "HTTP " + std::to_string(http_code) + ": " + response_body.substr(0, 300);
            return response;
        }

        try {
            const nlohmann::json parsed = nlohmann::json::parse(response_body);
            response.text = parsed.at("choices").at(0).at("message").at("content").get<std::string>();
        } catch (const std::exception& e) {
            response.error = std::string("unexpected response schema: ") + e.what();
            return response;
        }
        response.ok = true;
        return response;
    }

    bool supportsVision() const override { return vision_; }
    const char* name() const override { return config_.provider.c_str(); }

private:
    // messages 组装：带图像时 content 为 [{image_url},{text}] 数组（OpenAI vision 约定，
    // GLM-4.5V 等兼容该格式）；纯文本时 content 为字符串
    std::string buildRequestBody(const LlmRequest& request) const {
        nlohmann::json messages = nlohmann::json::array();
        if (!request.system.empty()) {
            messages.push_back({{"role", "system"}, {"content", request.system}});
        }

        if (!request.image_jpeg.empty()) {
            nlohmann::json content = nlohmann::json::array();
            content.push_back({
                {"type", "image_url"},
                {"image_url", {{"url", "data:image/jpeg;base64," +
                                          base64Encode(request.image_jpeg.data(), request.image_jpeg.size())}}},
            });
            content.push_back({{"type", "text"}, {"text", request.prompt}});
            messages.push_back({{"role", "user"}, {"content", content}});
        } else {
            messages.push_back({{"role", "user"}, {"content", request.prompt}});
        }

        nlohmann::json body = {
            {"model", config_.model.empty() ? "default" : config_.model},
            {"messages", messages},
            {"temperature", config_.temperature},
            {"max_tokens", config_.max_tokens},
            {"stream", false},
        };
        return body.dump();
    }

    LlmConfig config_;
    std::string default_endpoint_;
    bool vision_;
};

}  // namespace

LlmProvider* createOpenAiCompatProvider(const LlmConfig& config) {
    if (config.provider == "rkllm") {
        // 板端 RKLLM 本地服务：纯文本模型，默认端点见 tools/setup_rkllm.sh
        return new OpenAiCompatProvider(config, kDefaultRkllmEndpoint, false);
    }
    return new OpenAiCompatProvider(config, "", true);
}

}  // namespace llm
