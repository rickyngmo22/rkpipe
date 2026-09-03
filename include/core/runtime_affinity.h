#pragma once

#include <sched.h>

#include <cstdlib>
#include <sstream>
#include <string>

// CPU 亲和性工具（RK3588: cpu0-3=A55 小核, cpu4-7=A76 大核）。
// 用法（环境变量）：
//   RK_PIPE_PIN_WORKER=4-7   推理 worker 线程绑定大核
//   RK_PIPE_PIN_IO=0-3       reader/输出/编码等 IO 线程绑定小核
// 格式："0,2-3,7" 支持逗号分隔的单个 CPU 与区间。绑定失败静默忽略。

inline bool parseCpuSpec(const std::string& spec, cpu_set_t& set) {
    CPU_ZERO(&set);
    std::string token;
    std::istringstream ss(spec);
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        const std::size_t dash = token.find('-');
        if (dash == std::string::npos) {
            const int cpu = std::atoi(token.c_str());
            if (cpu < 0 || cpu >= CPU_SETSIZE) {
                return false;
            }
            CPU_SET(cpu, &set);
        } else {
            const int lo = std::atoi(token.substr(0, dash).c_str());
            const int hi = std::atoi(token.substr(dash + 1).c_str());
            if (lo < 0 || hi < lo || hi >= CPU_SETSIZE) {
                return false;
            }
            for (int c = lo; c <= hi; ++c) {
                CPU_SET(c, &set);
            }
        }
    }
    return true;
}

// 将当前线程绑定到 spec 指定的 CPU 列表。spec 为空或失败时返回 false。
inline bool pinCurrentThreadToCpus(const std::string& spec) {
    if (spec.empty()) {
        return false;
    }
    cpu_set_t set;
    if (!parseCpuSpec(spec, set)) {
        return false;
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// 从环境变量读取 CPU 列表并绑定当前线程。未设置/非法/权限不足时静默忽略。
inline bool pinCurrentThreadFromEnv(const char* env_name) {
    const char* v = std::getenv(env_name);
    if (!v || !*v) {
        return false;
    }
    return pinCurrentThreadToCpus(std::string(v));
}
