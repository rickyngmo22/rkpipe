// 板载负载采集与文件尾部读取（自 daemon_main.cc 拆出）：只读 /proc、/sys 与普通文件。
#include <algorithm>
#include <sys/statvfs.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "daemon/daemon_internal.h"

static bool readCpuTimes(unsigned long long& idle, unsigned long long& total) {
    std::ifstream f("/proc/stat");
    std::string cpu;
    unsigned long long u = 0, n = 0, s = 0, i = 0, io = 0, irq = 0, sirq = 0, st = 0;
    if (!(f >> cpu >> u >> n >> s >> i >> io >> irq >> sirq >> st)) {
        return false;
    }
    idle = i + io;
    total = u + n + s + i + io + irq + sirq + st;
    return true;
}

static double cpuUsagePercent() {
    unsigned long long i1 = 0, t1 = 0, i2 = 0, t2 = 0;
    if (!readCpuTimes(i1, t1)) {
        return 0.0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (!readCpuTimes(i2, t2)) {
        return 0.0;
    }
    const double dt = static_cast<double>(t2 - t1);
    if (dt <= 0.0) {
        return 0.0;
    }
    return 100.0 * (1.0 - static_cast<double>(i2 - i1) / dt);
}

static double maxCpuTempC() {
    double mx = 0.0;
    for (int z = 0; z < 16; ++z) {
        char path[64];
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", z);
        std::ifstream f(path);
        long t = 0;
        if (f && (f >> t)) {
            const double c = static_cast<double>(t) / 1000.0;
            if (c > mx) {
                mx = c;
            }
        }
    }
    return mx;
}

// NPU 三核负载（%）：读 /sys/kernel/debug/rknpu/load（需 root），格式 "NPU load:  Core0:  25%, Core1:  15%, Core2:  5%,"
static std::vector<double> npuLoadPcts() {
    std::vector<double> out;
    std::ifstream f("/sys/kernel/debug/rknpu/load");
    if (!f) {
        return out;
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    const std::string text = buf.str();
    for (int c = 0; c < 3; ++c) {
        const std::string key = "Core" + std::to_string(c) + ":";
        const size_t p = text.find(key);
        if (p == std::string::npos) {
            out.push_back(0.0);
            continue;
        }
        size_t q = p + key.size();
        while (q < text.size() && (text[q] == ' ' || text[q] == '\t')) {
            ++q;
        }
        out.push_back(std::atof(text.c_str() + q));
    }
    return out;
}

// 板载负载 JSON：{"load":[l1,l5,l15],"cpu_usage":x,"mem_total_mb":..,"mem_avail_mb":..,"mem_used_pct":..,"temp_c":..,"npu_load":[c0,c1,c2]}
std::string boardLoadJson() {
    double load1 = 0, load5 = 0, load15 = 0;
    {
        std::ifstream f("/proc/loadavg");
        if (f) {
            f >> load1 >> load5 >> load15;
        }
    }
    long mem_total_kb = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string key;
        long value = 0;
        std::string unit;
        while (f >> key >> value >> unit) {
            if (key == "MemTotal:") {
                mem_total_kb = value;
                break;
            }
        }
    }
    const long long mem_avail_kb = memAvailKb();
    const double mem_used_pct =
        (mem_total_kb > 0) ? 100.0 * static_cast<double>(mem_total_kb - mem_avail_kb) / mem_total_kb : 0.0;
    const auto npu = npuLoadPcts();
    std::ostringstream j;
    j << "{\"load\":[" << load1 << "," << load5 << "," << load15 << "],"
      << "\"cpu_usage\":" << cpuUsagePercent() << ","
      << "\"mem_total_mb\":" << (mem_total_kb / 1024) << ","
      << "\"mem_avail_mb\":" << (mem_avail_kb / 1024) << ","
      << "\"mem_used_pct\":" << mem_used_pct << ","
      << "\"temp_c\":" << maxCpuTempC() << ","
      << "\"npu_load\":[";
    for (size_t i = 0; i < npu.size(); ++i) {
        if (i > 0) {
            j << ",";
        }
        j << npu[i];
    }
    // 磁盘余量（dump_dir 所在文件系统：快照/事件/日志都在它下面）
    struct statvfs vfs {};
    if (::statvfs(g_opt.dump_dir.c_str(), &vfs) == 0) {
        const double bs = static_cast<double>(vfs.f_frsize);
        const double total_mb = bs * static_cast<double>(vfs.f_blocks) / (1024.0 * 1024.0);
        const double free_mb = bs * static_cast<double>(vfs.f_bavail) / (1024.0 * 1024.0);
        const double used_pct = total_mb > 0.0 ? 100.0 * (total_mb - free_mb) / total_mb : 0.0;
        j << "],\"disk_total_mb\":" << static_cast<long long>(total_mb)
          << ",\"disk_free_mb\":" << static_cast<long long>(free_mb)
          << ",\"disk_used_pct\":" << static_cast<long long>(used_pct);
    } else {
        j << "]";
    }
    j << "}";
    return j.str();
}

// 返回文件最后 n 行（用于任务日志查看）
std::string tailFile(const std::string& path, int n) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return "";
    }
    std::streamoff end = in.tellg();
    if (end <= 0) {
        return "";
    }
    std::streamoff pos = end;
    int lines = 0;
    while (pos > 0) {
        in.seekg(pos - 1);
        char c = 0;
        in.get(c);
        --pos;
        if (c == '\n' || c == '\r') {  // 进度行用 \r 覆盖（无 \n），两种都当作换行
            ++lines;
            if (lines > n) {
                break;
            }
        }
    }
    if (pos >= end) {
        return "";
    }
    in.clear();
    in.seekg(pos);
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string out = ss.str();
    // 去掉段首多出的半个换行
    if (lines > n && !out.empty() && (out[0] == '\n' || out[0] == '\r')) {
        out.erase(0, 1);
    }
    // 清洗日志：按 \r/\n 拆行（进度行用 \r 覆盖，拆成独立行），保留全部内容，
    // 每 10 帧的处理信息逐行展示。
    std::istringstream in_clean(out);
    std::string line;
    std::vector<std::string> raw_lines;
    while (std::getline(in_clean, line)) {
        std::string seg;
        for (const char ch : line) {  // 进度行用 \r 覆盖（无 \n），需再按 \r 拆分
            if (ch == '\r') {
                if (!seg.empty()) {
                    raw_lines.push_back(seg);
                }
                seg.clear();
            } else {
                seg += ch;
            }
        }
        if (!seg.empty()) {
            raw_lines.push_back(seg);
        }
    }
    std::ostringstream cleaned;
    for (const auto& l : raw_lines) {
        cleaned << l << "\n";
    }
    return cleaned.str();
}

// ---------------------------------------------------------------------------
// 磁盘治理（housekeeping）：快照 GC + 任务日志轮转。
// 快照无界增长会写满 eMMC（系统性故障）；日志轮转必须保持同 inode——子进程
// 持有 O_APPEND fd，rename 旧文件会使其继续写旧 inode 泄漏磁盘空间。
// 只处理本项目命名模式（snapshot_*.jpg）与显式配置过的目录/日志，不越界。
// ---------------------------------------------------------------------------
std::vector<std::string> collectSnapshotDirs() {
    std::vector<std::string> dirs;
    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    for (const auto& t : g_tasks) {
        for (const auto& kv : t->spec.passthrough) {
            if (kv.first != "alert_snapshot_dir" && kv.first != "snapshot_dir") {
                continue;
            }
            if (kv.second.empty() ||
                std::find(dirs.begin(), dirs.end(), kv.second) != dirs.end()) {
                continue;
            }
            dirs.push_back(kv.second);
        }
    }
    return dirs;
}
