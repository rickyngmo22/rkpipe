#pragma once

// 结构化日志宏（rk_pipe 日志规范沉淀）：
//   RKLOG_D/W/I/E(module, fmt, ...) → stderr，自动带 [module][级别] 前缀
// 既有约定即此风格（[rk_pipe][alert]、[daemon][llm]...），新代码统一走宏，
// 旧代码随改随迁（全量机械替换风险大于收益，不一次性刷）。
//
// module 约定小写无括号：alert/event/thermal/llm/daemon/output/...
// 示例：RKLOG_W(alert, "webhook send failed: %s", host.c_str());

#include <cstdio>

#define RKLOG_IMPL(level, module, ...)                       \
    do {                                                     \
        std::fprintf(stderr, "[" module "][" level "] ");    \
        std::fprintf(stderr, __VA_ARGS__);                   \
        std::fprintf(stderr, "\n");                          \
    } while (0)

#define RKLOG_D(module, ...) RKLOG_IMPL("debug", module, __VA_ARGS__)
#define RKLOG_I(module, ...) RKLOG_IMPL("info", module, __VA_ARGS__)
#define RKLOG_W(module, ...) RKLOG_IMPL("warn", module, __VA_ARGS__)
#define RKLOG_E(module, ...) RKLOG_IMPL("error", module, __VA_ARGS__)
