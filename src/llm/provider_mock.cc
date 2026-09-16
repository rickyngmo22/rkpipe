#include <cstdio>
#include <utility>

#include "llm/llm_provider.h"

#include "llm/llm_config.h"

namespace llm {

// OpenAI 兼容 HTTP provider（provider_cloud.cc 实现；未链 libcurl 的构建不提供）
#ifndef RK_PIPE_LLM_NO_CLOUD
LlmProvider* createOpenAiCompatProvider(const LlmConfig& config);
#endif

namespace {

// 离线模拟 provider：不发起任何网络/推理调用，返回确定性伪结论，
// 让复检/对话全链路（队列、落盘、web 端点）在无 API key 的板端即可验证。
class MockProvider : public LlmProvider {
public:
    LlmResponse generate(const LlmRequest& request) override {
        LlmResponse response;
        response.ok = true;
        if (!request.image_jpeg.empty()) {
            // 复检请求：返回固定结构的伪结论（字段与真实 VLM 输出约定一致）
            response.text =
                "{\"verdict\": \"real_defect\", \"confidence\": 0.86, "
                "\"description\": \"模拟复检（provider=mock）：检出目标判定为疑似真实缺陷，"
                "请配置 llm_provider=cloud 与 API key 获取真实模型结论。\"}";
        } else {
            response.text =
                "模拟回答（provider=mock）：已收到 " + std::to_string(request.prompt.size()) +
                " 字符的提问上下文。配置 llm_provider=cloud（或 rkllm）后此处将返回真实大模型回答。";
        }
        return response;
    }

    bool supportsVision() const override { return true; }
    const char* name() const override { return "mock"; }
};

// 未链 libcurl 的构建（RK_PIPE_LLM_NO_CLOUD）：cloud/rkllm 请求直接返回明确错误，
// 保证模块其余链路（队列/落盘/端点）仍可运行与测试
#ifdef RK_PIPE_LLM_NO_CLOUD
class NoCloudProvider : public LlmProvider {
public:
    explicit NoCloudProvider(std::string provider) : provider_(std::move(provider)) {}
    LlmResponse generate(const LlmRequest&) override {
        LlmResponse response;
        response.ok = false;
        response.error = "provider '" + provider_ + "' unavailable: built without libcurl";
        return response;
    }
    bool supportsVision() const override { return false; }
    const char* name() const override { return provider_.c_str(); }

private:
    std::string provider_;
};
#endif

}  // namespace

LlmProvider* createProvider(const LlmConfig& config) {
    const std::string& provider = config.provider;
    if (provider == "cloud" || provider == "rkllm") {
#ifdef RK_PIPE_LLM_NO_CLOUD
        return new NoCloudProvider(provider);
#else
        return createOpenAiCompatProvider(config);
#endif
    }
    if (provider != "mock") {
        std::printf("[Llm] unknown provider '%s', fallback to mock\n", provider.c_str());
    }
    return new MockProvider();
}

}  // namespace llm
