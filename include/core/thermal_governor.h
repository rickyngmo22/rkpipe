#pragma once

#include <atomic>
#include <string>

// 热降档策略（纯逻辑，温度由调用方注入，可单测）。
//
// 无风扇 RK3588 盒子长期高负载会触发热降频：NPU/CPU 一旦降频，推理吞吐
// 骤降且队列积压恶化。与其撞上"降频→积压→更热"的恶性循环，不如在温度
// 接近阈值时主动限制推理帧率上限（经 AdaptiveFrameSkipState 的 external_cap）。
//
// 档位（level 0~3）：
//   temp >= high_c               → level = min(3, 1 + (temp-high)/step)
//   temp <  resume_c             → 每次调用恢复一级（渐进回满）
//   resume_c <= temp < high_c    → 保持现状（迟滞区，防抖动）
// 档位系数：1.0 / 0.6 / 0.4 / min_ratio
//
// 环境变量（在 app_runner 侧解析）：
//   RK_PIPE_THERMAL_MAX_C   降档阈值（默认 85，0=禁用采样）
//   RK_PIPE_THERMAL_RESUME_C 恢复阈值（默认 80）

class ThermalGovernor {
public:
    struct Params {
        double high_c = 85.0;
        double resume_c = 80.0;
        double step_c = 5.0;
        double min_ratio = 0.25;
    };

    ThermalGovernor() = default;
    explicit ThermalGovernor(const Params& p) : p_(p) {}

    // temp_c 传负数（读取失败）时保持现状。base_fps 为正常上限（通常 = 目标帧率）。
    // 返回当前允许的 fps 上限。
    double update(double temp_c, double base_fps);
    double currentFps() const { return current_fps_.load(); }
    int level() const { return level_.load(); }
    double thresholdC() const { return p_.high_c; }

private:
    Params p_;
    std::atomic<int> level_{0};
    std::atomic<double> current_fps_{0.0};
};

// 板温采样器（子进程版）：fork+exec 一个 /bin/sh 循环，周期读
// /sys/class/thermal 并把最高温（毫摄氏度）写入 /dev/shm 文件；
// 父进程按需（内部 2s 节流）读该 tmpfs 文件。
//
// 为什么是子进程：实测在 NPU/MPP 已初始化的进程里，任何线程读
// /sys/class/thermal 都会使主流水线确定性死锁（疑似 RKNPU cooling device
// 与内核 thermal framework 的锁交互，用户态无解）。子进程经 exec 后地址
// 空间完全重置、无任何驱动上下文，读 sysfs 与父进程完全隔离。
//
// 父进程读的是 /dev/shm（tmpfs），与 thermal framework 无交互，安全。

class ThermalSampler {
public:
    // high_c<=0 = 禁用（start 直接返回）
    void start(double high_c, int period_ms = 2000);
    void stop();  // 终止子进程并清理 /dev/shm 文件；未启动时无害
    // 最近一次采样温度（摄氏度；未采样过/读取失败为 -1）
    double lastTempC();

private:
    std::string shm_path_;      // /dev/shm/rkpipe_thermal_<ppid>
    double cached_c_ = -1.0;    // 内部节流缓存
    double last_read_s_ = -1e9; // 上次读文件的稳态时钟（秒）
    bool child_started_ = false;
    int child_pid_ = -1;
};
