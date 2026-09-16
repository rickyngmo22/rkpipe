#pragma once

// LLM 分析模块配置。
//
// 设计约束：闭源核心按成员布局消费 AppConfig，在其上追加字段会破坏与
// 预编译 librkpipe_core.a 的 ABI 兼容，因此 LLM 配置走独立结构体 + 独立 YAML，
// 与主流水线配置文件完全解耦。
//
// 加载约定（见 fromDefaultLocation）：
//   1. 环境变量 RKPIPE_LLM_CONFIG 指定的路径优先；
//   2. 否则回退当前工作目录 ./llm.yaml；
//   3. 文件不存在或未写 llm_enabled: 1 时模块静默禁用（零开销，不报错）。

#include <string>
#include <vector>

struct LlmConfig {
    bool enabled = false;
    // mock：离线模拟响应（无网络/API key 也能全链路验证）
    // cloud：OpenAI 兼容 chat/completions HTTP（智谱 GLM / DashScope / 任意兼容网关）
    // rkllm：板端 RKLLM 服务（OpenAI 兼容协议，默认 http://127.0.0.1:8080/v1/chat/completions）
    std::string provider = "mock";
    std::string config_path;  // 实际加载的配置文件路径（诊断用）

    // ---- 提交策略（finalizePipelineFrameOutput 挂接点）----
    float min_conf = 0.5f;    // 帧内存在 conf < min_conf 的检出才提交（borderline 复检）
    double interval_s = 2.0;  // 两次提交的最小间隔（限速，控制 LLM 调用频率/成本）
    bool submit_all = false;  // true = 不做 min_conf 过滤，帧内有检出即提交
    int queue_size = 4;       // 有界队列深度（满则丢最旧，绝不阻塞输出线程）
    int max_results_keep = 200;   // 内存保留的最近结论条数（/api/llm/status 展示）
    std::string results_path;     // 非空时逐条追加 JSONL（供离线报告工具消费）
    std::string stats_jsonl_path; // 非空时尾读该 detections JSONL 聚合统计（对话上下文增强）

    // ---- 模型 / 服务 ----
    std::string model;        // cloud: 模型名（如 glm-4.5v）；rkllm: .rkllm 模型名
    std::string endpoint;     // OpenAI 兼容 chat/completions 完整 URL
    std::string api_key;      // Bearer token；留空则从 api_key_env 环境变量读取
    std::string api_key_env = "LLM_API_KEY";
    double timeout_s = 60.0;
    double temperature = 0.2;
    int max_tokens = 512;
    bool send_image = true;       // 随复检请求附缺陷区域 JPEG（需多模态模型）
    double image_max_side = 640;  // 发送前把截图长边缩放到该值（省带宽与时延）

    // ---- 标签 ----
    std::string label_path;  // 类别 id->名称 列表文件（每行一个名称）；空则输出 class_<id>

    // FileStorage 扁平键解析（键名带 llm_ 前缀，风格与 AppConfig 一致）
    bool loadFromFile(const std::string& filename);

    // label_path 逐行读类别名；文件不存在返回空表
    std::vector<std::string> loadLabels() const;

    // 按约定路径发现配置：文件不存在时返回默认构造（enabled=false）
    static LlmConfig fromDefaultLocation();
};
