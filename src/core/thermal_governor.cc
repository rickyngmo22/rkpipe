#include "core/thermal_governor.h"

#include <sys/prctl.h>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>

namespace {
constexpr double kLevelRatio[4] = {1.0, 0.6, 0.4, 0.25};
}  // namespace

double ThermalGovernor::update(double temp_c, double base_fps) {
    if (base_fps <= 0.0) {
        current_fps_.store(0.0);
        return current_fps_.load();
    }
    if (temp_c >= 0.0) {
        if (temp_c >= p_.high_c) {
            const int over = static_cast<int>((temp_c - p_.high_c) / std::max(p_.step_c, 0.1));
            level_.store(std::min(3, 1 + over));
        } else if (temp_c < p_.resume_c) {
            level_.store(std::max(0, level_.load() - 1));  // 渐进恢复（调用周期 ~2s）
        }
        // 迟滞区保持现状
    }
    current_fps_.store(base_fps * kLevelRatio[level_.load()]);
    return current_fps_.load();
}

namespace {
constexpr double kThermalReadMinIntervalS = 2.0;

// 父进程读子进程写出的 /dev/shm 文件（tmpfs，与 thermal framework 无交互）
bool readShmTempC(const std::string& path, double* out_c) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    double milli = 0.0;
    const int n = std::fscanf(f, "%lf", &milli);
    std::fclose(f);
    if (n != 1 || milli <= 0) {
        return false;
    }
    *out_c = milli / 1000.0;
    return true;
}
}  // namespace

void ThermalSampler::start(double high_c, int period_ms) {
    if (high_c <= 0.0 || child_started_) {
        return;
    }
    shm_path_ = "/dev/shm/rkpipe_thermal_" + std::to_string(static_cast<long>(::getpid()));
    std::string period_s = std::to_string(std::max(1, period_ms / 1000));
    // 子进程 shell 循环：取全部热区最高毫摄氏度写入 /dev/shm（tmpfs）
    std::string cmd =
        "while :; do cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null"
        " | sort -n | tail -1 > " + shm_path_ + ".tmp 2>/dev/null"
        " && mv " + shm_path_ + ".tmp " + shm_path_ + " 2>/dev/null; sleep " + period_s + "; done";
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "[rk_pipe][thermal] fork 采样进程失败，热降档禁用\n");
        return;
    }
    if (pid == 0) {
        // 子进程：父退出时随之退出；关闭全部继承 fd（RKNN/MPP 的 dma-buf
        // fd 副本会干扰父进程 NPU fence 信号——fork+exec 的标准卫生实践），
        // 再 exec 重置地址空间
        ::prctl(PR_SET_PDEATHSIG, SIGKILL);
        for (int fd = 3; fd < 256; ++fd) {
            ::close(fd);
        }
        ::execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);  // exec 失败
    }
    child_pid_ = pid;
    child_started_ = true;
}

void ThermalSampler::stop() {
    if (!child_started_) {
        return;
    }
    ::kill(child_pid_, SIGTERM);
    int status = 0;
    ::waitpid(child_pid_, &status, 0);
    child_started_ = false;
    std::remove(shm_path_.c_str());
    std::remove((shm_path_ + ".tmp").c_str());
}

double ThermalSampler::lastTempC() {
    const double now_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_s - last_read_s_ >= kThermalReadMinIntervalS) {
        last_read_s_ = now_s;
        double c = 0.0;
        if (readShmTempC(shm_path_, &c)) {
            cached_c_ = c;
        }
    }
    return cached_c_;
}
