#pragma once

// LLM Provider 抽象：统一 mock（离线测试）/ cloud（OpenAI 兼容 HTTP）/ rkllm
// （板端 RKLLM 服务，同为 OpenAI 兼容协议）三种后端。
//
// 说明：RKLLM 走"本地 OpenAI 兼容服务"路线（rknn-llm 仓库自带 rkllm-server 示例，
// 见 tools/setup_rkllm.sh），而不是 C API dlopen 直连——后者结构体布局随 SDK 版本
// 变化且板上无 librkllmrt 无法验证，HTTP 路线零新增原生依赖、可测、可替换。

#include <string>
#include <vector>

struct LlmConfig;

namespace llm {

// 一次生成请求：prompt 文本 + 可选图像（JPEG 字节，多模态模型用）
struct LlmRequest {
    std::string system;                     // 可选 system 提示词
    std::string prompt;
    std::vector<unsigned char> image_jpeg;  // 非空且 provider 支持视觉时随请求发送
};

// 一次生成结果
struct LlmResponse {
    bool ok = false;
    std::string text;    // ok 时为模型输出
    std::string error;   // !ok 时为错误描述
    double elapsed_ms = 0.0;
};

class LlmProvider {
public:
    virtual ~LlmProvider() = default;

    // 实现必须线程安全（复检 worker 与 chat 端点可能并发调用）
    virtual LlmResponse generate(const LlmRequest& request) = 0;

    // 是否支持图像输入（文本模型返回 false，analyzer 据此跳过截图编码）
    virtual bool supportsVision() const = 0;

    virtual const char* name() const = 0;
};

// 按 config.provider 分派；未知 provider 回退 mock。永不返回空。
LlmProvider* createProvider(const LlmConfig& config);

// OpenAI 兼容 HTTP provider（provider_cloud.cc 实现；未链 libcurl 的构建不提供，
// createProvider 内部按 RK_PIPE_LLM_NO_CLOUD 降级为报错实现）
LlmProvider* createOpenAiCompatProvider(const LlmConfig& config);

}  // namespace llm
