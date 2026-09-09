#include "io/result_sink.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#include "config/app_config.h"
#include "core/pipeline_frame.h"
#include "core/task_result_json.h"

namespace {

// 过渡期配置:AppConfig 与闭源核心布局耦合不可加字段,逐帧结果 JSONL 路径
// 先经环境变量下发;YAML 键 result_jsonl_path 随配套核心 Release 协同启用
constexpr char kResultJsonlEnv[] = "RK_PIPE_RESULT_JSONL";

struct FrameResultSinkState {
    std::mutex mutex;
    std::ofstream file;
    std::string path;
    bool failed = false;
};

FrameResultSinkState& sinkState() {
    static FrameResultSinkState state;
    return state;
}

}  // namespace

void configureFrameResultSink(const AppConfig& options) {
    (void)options;  // 下个核心 Release 起:优先读 options.result_jsonl_path(协同启用),环境变量作覆盖
    const char* env_path = std::getenv(kResultJsonlEnv);
    if (env_path == nullptr || *env_path == '\0') {
        return;
    }
    FrameResultSinkState& s = sinkState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file.is_open() || !s.path.empty()) {
        return;  // 已初始化:一次进程一条汇,不随重复调用重开
    }
    s.path = env_path;
    s.file.open(s.path, std::ios::out | std::ios::trunc);
    if (!s.file.is_open()) {
        s.failed = true;
        std::printf("[rkpipe] %s 打开失败: %s(逐帧结果不落盘,流水线继续)\n",
                    kResultJsonlEnv, s.path.c_str());
        return;
    }
    std::printf("[rkpipe] 逐帧结果 JSONL(schema v1,docs/event_payload.md): %s\n",
                s.path.c_str());
}

void writeFrameResult(const PipelineFrame& frame) {
    FrameResultSinkState& s = sinkState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.failed || s.path.empty() || !s.file.is_open() || !frame.hasResult) {
        return;
    }
    const std::string line = buildFrameResultJson(frame);
    if (!(s.file << line << '\n')) {
        s.failed = true;
        s.file.close();
        std::printf("[rkpipe] 逐帧结果 JSONL 写入失败,已停用: %s\n", s.path.c_str());
    }
}

void shutdownFrameResultSink() {
    FrameResultSinkState& s = sinkState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file.is_open()) {
        s.file.flush();
        s.file.close();
    }
    s.path.clear();
    s.failed = false;
}
