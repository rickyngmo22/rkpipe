// rk_pipe_daemon：多任务常驻守护进程（B3 REST 服务化 + C2 模型热切换）
//
// 职责：
//   - 读取与 stream_multi_web 相同的 streams.json 配置，fork 监督 N 个 console_detector 子进程
//   - REST API（默认 :8099）：
//       GET    /api/tasks                    任务列表（id/task/input/model/port/pid/status/uptime/restarts）
//       GET    /api/tasks/<id>               单任务详情
//       POST   /api/tasks                    从 query 参数动态启停新任务（task/input/model/threads/port/...）
//       DELETE /api/tasks/<id>               优雅停止任务
//       POST   /api/tasks/<id>/restart       重启任务（模型热切换入口）
//       POST   /api/reload                   重载配置文件，热切换：新增/删除/重启变更任务
//       GET    /api/tasks/<id>/status.json   代理子进程 /status.json（只读结果 API）
//       GET    /api/health                   存活探针
//   - 事件闭环：POST /internal/event（子进程告警上报→JSONL 落盘→LLM 复检→转发业务 webhook）、
//       GET /api/events 事件查询、POST /api/events/<id>/review 人工复核
//   - 动态任务持久化：REST 创建的任务写 dynamic_tasks.json，重启自动恢复
//   - 配置热切换：--watch <s>（默认 2s）轮询配置文件 mtime，变化自动 reload
//   - --restart：子进程崩溃自动拉起（指数退避，10 分钟滑动窗口内最多 5 次）
//
// 编译：cmake --build build --target rk_pipe_daemon

#include <arpa/inet.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <poll.h>
#include <sys/statvfs.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "daemon/daemon_internal.h"
#include "daemon/events_store.h"
#include "daemon/llm_review.h"
#include "daemon/mini_json.h"
#include "utils/json_escape.h"

// ---- 全局状态（声明见 daemon/daemon_internal.h，定义唯一在本单元）----


void gcSnapshotDir(const std::string& dir, long long max_bytes, int max_age_days) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return;
    }
    const auto now = std::filesystem::file_time_type::clock::now();
    struct Entry {
        std::filesystem::path path;
        std::uintmax_t size = 0;
        std::filesystem::file_time_type mtime{};
    };
    std::vector<Entry> entries;
    std::uintmax_t total = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!e.is_regular_file(ec)) {
            continue;
        }
        const std::string name = e.path().filename().string();
        // 快照与事件片段同目录：两类一并治理
        const bool is_snapshot =
            name.rfind("snapshot_", 0) == 0 &&
            name.size() >= 4 && name.compare(name.size() - 4, 4, ".jpg") == 0;
        const bool is_clip = name.rfind("clip_", 0) == 0 &&
                             name.size() >= 4 && name.compare(name.size() - 4, 4, ".avi") == 0;
        if (!is_snapshot && !is_clip) {
            continue;
        }
        const auto mtime = e.last_write_time(ec);
        const auto size = e.file_size(ec);
        if (ec) {
            continue;
        }
        total += size;
        if (max_age_days > 0 && now - mtime > std::chrono::hours(24 * max_age_days)) {
            std::filesystem::remove(e.path(), ec);
            total -= size;
            continue;
        }
        entries.push_back(Entry{e.path(), size, mtime});
    }
    if (max_bytes > 0 && total > max_bytes) {
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
        for (const auto& en : entries) {
            if (total <= max_bytes) {
                break;
            }
            std::error_code rec;
            const auto sz = std::filesystem::file_size(en.path, rec);
            if (std::filesystem::remove(en.path, rec)) {
                total = total > sz ? total - sz : 0;
            }
        }
    }
}

void rotateLogTail(const std::string& path, long long max_bytes) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0 || st.st_size <= max_bytes) {
        return;
    }
    const auto keep = static_cast<size_t>(max_bytes / 2);
    std::string tail;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return;
        }
        in.seekg(-static_cast<std::streamoff>(keep), std::ios::end);
        std::ostringstream ss;
        ss << in.rdbuf();
        tail = ss.str();
        // 对齐到行边界（丢掉半行）
        const size_t nl = tail.find('\n');
        if (nl != std::string::npos) {
            tail.erase(0, nl + 1);
        }
    }
    if (tail.empty()) {
        return;
    }
    // 同 inode 截断重写：子进程的 O_APPEND fd 继续有效，后续追加落在新 EOF
    const int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        return;
    }
    ::ftruncate(fd, 0);
    ::lseek(fd, 0, SEEK_SET);
    const ssize_t w = ::write(fd, tail.data(), static_cast<size_t>(tail.size()));
    (void)w;
    ::close(fd);
    std::fprintf(stderr, "[daemon] rotated log %s (kept %zu bytes)\n", path.c_str(), tail.size());
}

void housekeepingLoop() {
    // 首轮先等一小段，避开启动风暴（spawn/warmup 抢内存与 CPU）
    std::this_thread::sleep_for(std::chrono::seconds(std::min(g_opt.housekeep_s, 30)));
    while (!g_shutdown.load()) {
        const long long snap_bytes =
            g_opt.snapshot_max_mb > 0 ? g_opt.snapshot_max_mb * (1ll << 20) : 0;
        if (snap_bytes > 0 || g_opt.snapshot_max_age_days > 0) {
            for (const auto& dir : collectSnapshotDirs()) {
                gcSnapshotDir(dir, snap_bytes, g_opt.snapshot_max_age_days);
            }
        }
        if (g_opt.log_max_mb > 0) {
            std::lock_guard<std::mutex> lock(g_tasks_mutex);
            for (const auto& t : g_tasks) {
                rotateLogTail(t->log_path, g_opt.log_max_mb * (1ll << 20));
            }
        }
        for (int i = 0; i < g_opt.housekeep_s && !g_shutdown.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

static void onSignal(int) {
    g_shutdown.store(true);
}

static void supervisionLoop() {
    while (!g_shutdown.load()) {
        // 回收已退出子进程：仅记录退出原因并安排重启（指数退避），
        // 实际拉起在下方统一的"到期拉起"块中执行
        {
            std::lock_guard<std::mutex> lock(g_tasks_mutex);
            for (auto& t : g_tasks) {
                if (t->pid <= 0) {
                    continue;
                }
                int status = 0;
                const pid_t r = ::waitpid(t->pid, &status, WNOHANG);
                if (r != t->pid) {
                    continue;
                }
                t->pid = -1;
                if (g_shutdown.load()) {
                    continue;
                }
                const bool normal_exit = WIFEXITED(status);
                const int code = normal_exit ? WEXITSTATUS(status)
                                             : (WIFSIGNALED(status) ? WTERMSIG(status) : -1);
                std::fprintf(stderr, "[daemon] task %d exited (code=%d%s)\n", t->id, code,
                             normal_exit ? "" : " by signal");
                // 仅异常退出才自动重启（崩溃/非 0 退出码/被信号杀死）；
                // 正常跑完（exit 0，如文件播完）不重启，避免播完又被拉起占用资源。
                const bool crashed = !normal_exit || code != 0;
                if (!crashed || !g_opt.restart_on_failure || !t->wanted.load()) {
                    t->pending_restart = false;
                    continue;
                }
                // 滑动窗口限次：10 分钟内最多 5 次重启，超限放弃（避免 crash-loop 空转）
                const auto now = std::chrono::steady_clock::now();
                while (!t->restart_window.empty() &&
                       now - t->restart_window.front() > std::chrono::seconds(600)) {
                    t->restart_window.pop_front();
                }
                if (t->restart_window.size() >= 5) {
                    t->give_up = true;
                    t->pending_restart = false;
                    std::fprintf(stderr,
                                 "[daemon] task %d crash-looping (5 restarts in 10min), giving up; "
                                 "manual restart / config reload to recover\n",
                                 t->id);
                    continue;
                }
                t->restart_window.push_back(now);
                ++t->restarts;
                t->pending_restart = true;
                const auto backoff =
                    std::chrono::seconds(std::min(60, 1 << std::min(t->restarts, 6)));  // 2,4,8..60s
                t->next_restart_at = now + backoff;
                std::fprintf(stderr, "[daemon] task %d will restart in %lds (restart #%d)\n", t->id,
                             static_cast<long>(std::chrono::duration_cast<std::chrono::seconds>(backoff).count()),
                             t->restarts);
            }

            // 退避到期后拉起（含 spawn 失败后的 5s 重试，如内存护栏暂时不足）
            if (g_opt.restart_on_failure) {
                const auto now = std::chrono::steady_clock::now();
                for (auto& t : g_tasks) {
                    if (!t->wanted.load() || t->give_up || !t->pending_restart || t->pid > 0) {
                        continue;
                    }
                    if (now < t->next_restart_at) {
                        continue;
                    }
                    t->pending_restart = false;
                    if (!spawnTask(t.get())) {
                        t->pending_restart = true;
                        t->next_restart_at = now + std::chrono::seconds(5);
                    }
                }
            }
        }

        // 配置文件热切换（streams.json + 动态任务持久化合并）
        if (g_opt.watch_s > 0.0 && !g_opt.streams_file.empty()) {
            const int64_t mtime = fileMtime(g_opt.streams_file);
            if (mtime > 0 && mtime != g_config_mtime.load()) {
                g_config_mtime.store(mtime);
                std::printf("[daemon] config changed, hot-swapping\n");
                reloadAndApply();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    // 关闭：优雅停止所有子进程
    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    for (auto& t : g_tasks) {
        if (t->pid > 0) {
            ::kill(t->pid, SIGTERM);
        }
    }
    // 最多等 3s
    for (int i = 0; i < 30; ++i) {
        bool any = false;
        for (auto& t : g_tasks) {
            if (t->pid > 0) {
                any = true;
            }
        }
        if (!any) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (auto& t : g_tasks) {
        if (t->pid > 0) {
            ::kill(t->pid, SIGKILL);
        }
    }
    std::printf("[daemon] all tasks stopped, bye\n");
}

static void printUsage(const char* prog) {
    std::printf(
        "Usage: %s --streams-file <path> [options]\n"
        "\n"
        "常驻守护：fork 监督多个 console_detector 任务，暴露 REST API（B3）并支持配置热切换（C2）。\n"
        "\n"
        "Options:\n"
        "  --streams-file <path>  多路配置 JSON（与 stream_multi_web 相同格式，必填）\n"
        "                           每条 stream 可带 output=<推流地址>（rtmp/rtsp/udp/srt）；\n"
        "                           可带 codec=h264|hevc 切换推流编码（H.265 需播放器支持）；\n"
        "                           兼容 output 尾部 `,codec=hevc` 的写法（自动剥离）\n"
        "                           有推流时监控页预览自动用 WebRTC(经 MediaMTX)，否则 MJPEG\n"
        "  --api-port <n>         REST API 端口 (default: 8099)\n"
        "  --bind <addr>          API/预览 bind 地址 (default: 0.0.0.0)\n"
        "  --base-port <n>        预览端口起始 (default: 8090)\n"
        "  --quality <1-100>      MJPEG 质量 (default: 60，内嵌监控画面够用)\n"
        "  --dump-dir <path>      生成的 yaml/log 目录 (default: /tmp/rk_pipe_daemon)\n"
        "  --bin <path>           console_detector 路径\n"
        "  --root <path>          rk_pipe 根目录（模型/label 查找）\n"
        "  --metrics-file <path>  给每个子进程注入 RK_PIPE_METRICS_FILE=<path>.<id>\n"
        "  --watch <s>            配置轮询热切换间隔，0=关闭 (default: 2)\n"
        "  --restart              子进程崩溃自动拉起（指数退避 2..60s，10 分钟内最多 5 次）\n"
        "  --forward-url <url>    业务 webhook：/internal/event 收到的事件在复检完成后\n"
        "                         （或未启用复检时）POST 到该地址，body 增补 llm 结论字段\n"
        "  --flywheel-dir <dir>   误报训练样本归档目录（复检 verdict=false 时归档快照+manifest，\n"
        "                         亦可用环境变量 RK_PIPE_FLYWHEEL_DIR）\n"
        "  --events-file <path>   事件 JSONL 落盘路径 (default: <dump-dir>/events.jsonl)\n"
        "  --dynamic-file <path>  REST 动态任务持久化路径 (default: <dump-dir>/dynamic_tasks.json)\n"
        "  --api-token <t>        写操作鉴权 token（非 GET 需携带；GET 只读放行，env RK_PIPE_API_TOKEN）\n"
        "  --snapshot-max-mb <n>  快照目录容量上限，超出删最旧 (default: 1024，0=关闭)\n"
        "  --snapshot-max-age-days <n> 快照保留天数 (default: 7，0=关闭)\n"
        "  --log-max-mb <n>       任务日志上限，尾部保留式轮转 (default: 16，0=关闭)\n"
        "  --housekeep-s <s>      磁盘治理周期 (default: 600)\n"
        "  -h, --help             帮助\n"
        "\n"
        "事件闭环 / LLM 复检（环境变量，api_key 不进命令行与 streams.json）:\n"
        "  子进程告警指向 daemon 即接入：alert_webhook_url=http://127.0.0.1:<api-port>/internal/event\n"
        "  RK_PIPE_LLM_BASE_URL / RK_PIPE_LLM_MODEL  OpenAI 兼容 VLM（空=关闭复检，仅落盘+转发）\n"
        "  RK_PIPE_LLM_API_KEY / RK_PIPE_LLM_EXTRA_PROMPT / RK_PIPE_LLM_TIMEOUT_MS\n"
        "  RK_PIPE_LLM_MAX_CONCURRENCY / RK_PIPE_LLM_RATE_PER_MIN / RK_PIPE_LLM_QUEUE_DEPTH\n"
        "  RK_PIPE_LLM_DROP_FALSE=1   verdict=false 的误报不再转发业务 webhook\n"
        "  GET /api/events?task=&type=&since=&limit=   事件查询（含复检结论）\n"
        "  GET /timeline                                事件时间线页（过滤/统计/复核徽章）\n"
        "\n"
        "REST API 示例:\n"
        "  curl http://<ip>:8099/api/tasks\n"
        "  curl -X POST 'http://<ip>:8099/api/tasks?task=pose&input=rtsp://x&threads=2&port=8093'\n"
        "  curl -X DELETE http://<ip>:8099/api/tasks/2\n"
        "  curl -X POST http://<ip>:8099/api/tasks/1/restart\n"
        "  curl -X POST http://<ip>:8099/api/reload     # 重载配置(模型热切换)\n"
        "  curl http://<ip>:8099/api/tasks/1/status.json\n"
        "\n"
        "事件/告警（每路可选，事件规则引擎 D1/D2 + webhook 告警）:\n"
        "  streams.json 每路可直接写扁平键（与子进程 yaml 键名一致），或嵌套对象：\n"
        "    {\"task\":\"detect\",\"input\":\"rtsp://x\",\"tracking\":1,\n"
        "     \"event_line\":\"450,740,1250,680\",\"event_line_cross\":1,\n"
        "     \"event_region\":\"820,480,1230,500,1300,810,750,810\",\"event_intrusion\":1,\n"
        "     \"alert\":{\"enabled\":1,\"webhook_url\":\"http://<host>:9000/alert\",\n"
        "               \"interval_s\":5,\"snapshot_dir\":\"./snapshots\"}}\n"
        "  嵌套 alert 对象: enabled/webhook_url/min_count/interval_s/snapshot_dir(告警抓拍)/\n"
        "                   dedup_interval_s/dedup_state_dir/periodic_snapshot_dir(周期快照)/snapshot_interval_s\n"
        "  也支持 extra_yaml 字符串原样追加任意 yaml 字段（高级）；REST 同名 query 参数\n"
        "  可用于 POST /api/tasks 与 /api/tasks/<id>/update（传空值=删除该键）\n",
        prog);
}


int main(int argc, char** argv) {
    // 兼容 --flag=value 写法：拆成两个参数
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            const size_t eq = a.find('=');
            if (eq != std::string::npos) {
                args.push_back(a.substr(0, eq));
                args.push_back(a.substr(eq + 1));
                continue;
            }
        }
        args.push_back(a);
    }
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string arg = args[i];
        const auto next = [&]() -> std::string {
            return i + 1 < args.size() ? args[++i] : "";
        };
        if (arg == "--streams-file") {
            g_opt.streams_file = next();
        } else if (arg == "--api-port") {
            g_opt.api_port = std::atoi(next().c_str());
        } else if (arg == "--bind") {
            g_opt.bind = next();
        } else if (arg == "--base-port") {
            g_opt.base_port = std::atoi(next().c_str());
        } else if (arg == "--quality") {
            g_opt.quality = std::atoi(next().c_str());
        } else if (arg == "--dump-dir") {
            g_opt.dump_dir = next();
        } else if (arg == "--bin") {
            g_opt.bin_path = next();
        } else if (arg == "--root") {
            g_opt.root_dir = next();
        } else if (arg == "--metrics-file") {
            g_opt.metrics_file = next();
        } else if (arg == "--watch") {
            g_opt.watch_s = std::atof(next().c_str());
        } else if (arg == "--restart") {
            g_opt.restart_on_failure = true;
        } else if (arg == "--flywheel-dir") {
            g_opt.flywheel_dir = next();
        } else if (arg == "--forward-url") {
            g_opt.forward_url = next();
        } else if (arg == "--events-file") {
            g_opt.events_file = next();
        } else if (arg == "--dynamic-file") {
            g_opt.dynamic_file = next();
        } else if (arg == "--api-token") {
            g_opt.api_token = next();
        } else if (arg == "--snapshot-max-mb") {
            g_opt.snapshot_max_mb = std::atoll(next().c_str());
        } else if (arg == "--snapshot-max-age-days") {
            g_opt.snapshot_max_age_days = std::atoi(next().c_str());
        } else if (arg == "--log-max-mb") {
            g_opt.log_max_mb = std::atoll(next().c_str());
        } else if (arg == "--housekeep-s") {
            g_opt.housekeep_s = std::atoi(next().c_str());
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    if (g_opt.streams_file.empty()) {
        std::fprintf(stderr, "Error: --streams-file is required\n");
        printUsage(argv[0]);
        return 1;
    }
    if (const char* env = std::getenv("RK_PIPE_BIN_PATH"); env && env[0]) {
        g_opt.bin_path = env;
    }
    if (const char* env = std::getenv("RK_PIPE_ROOT_DIR"); env && env[0]) {
        g_opt.root_dir = env;
    }
    if (g_opt.api_token.empty()) {
        if (const char* env = std::getenv("RK_PIPE_API_TOKEN"); env && env[0]) {
            g_opt.api_token = env;
        }
    }
    if (const char* env = std::getenv("RK_PIPE_HOUSEKEEP_S"); env && env[0]) {
        const int v = std::atoi(env);
        if (v > 0) {
            g_opt.housekeep_s = v;
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(g_opt.dump_dir, ec);

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGCHLD, SIG_DFL);  // 使用 waitpid WNOHANG 轮询回收

    // -------------------------------------------------------------------
    // 事件闭环初始化：事件存储 + 业务 webhook 转发 + LLM 复检（环境变量配置）
    // -------------------------------------------------------------------
    if (g_opt.events_file.empty()) {
        g_opt.events_file = g_opt.dump_dir + "/events.jsonl";
    }
    if (g_opt.dynamic_file.empty()) {
        g_opt.dynamic_file = g_opt.dump_dir + "/dynamic_tasks.json";
    }
    static EventStore event_store(g_opt.events_file, 2048, 32ull << 20);
    g_events = &event_store;
    {
        const char* env_dir = std::getenv("RK_PIPE_FLYWHEEL_DIR");
        const std::string dir = !g_opt.flywheel_dir.empty() ? g_opt.flywheel_dir
                                : (env_dir && *env_dir) ? std::string(env_dir) : std::string();
        static EventFlywheel flywheel(dir);
        g_flywheel = &flywheel;
    }
    if (!g_opt.forward_url.empty()) {
        if (g_opt.forward_url.rfind("https://", 0) == 0) {
            g_forward_url_full_ = g_opt.forward_url;  // https 走 libcurl（utils/http_post）
            g_forward_enabled = true;
        } else if (parseHttpUrl(g_opt.forward_url, &g_forward_host, &g_forward_port, &g_forward_path)) {
            g_forward_enabled = true;
        } else {
            std::fprintf(stderr, "[daemon][warn] --forward-url 仅支持 http(s)://，事件转发已禁用\n");
        }
    }
    static LlmReviewer reviewer(LlmReviewer::fromEnv(), g_opt.dump_dir, g_events,
                                [](long id, const std::string& body) {
                                    if (!g_forward_enabled) {
                                        return;
                                    }
                                    const bool ok = !g_forward_url_full_.empty()
                                                        ? httpPostUrl(g_forward_url_full_, body,
                                                                      "application/json", 5000)
                                                        : httpPostJson(g_forward_host, g_forward_port,
                                                                       g_forward_path, body, 5000);
                                    std::fprintf(stderr, "[daemon] event %ld forwarded: %s\n", id,
                                                 ok ? "ok" : "failed");
                                });
    g_reviewer = reviewer.enabled() ? &reviewer : nullptr;

    const int configured = reloadAndApply();
    if (configured == 0) {
        std::fprintf(stderr, "[daemon][warn] no stream configured in %s\n", g_opt.streams_file.c_str());
    }
    g_config_mtime.store(fileMtime(g_opt.streams_file));

    std::thread http_thread(acceptLoop);
    http_thread.detach();  // accept() 阻塞，进程退出时随主线程终止
    std::thread housekeeping_thread(housekeepingLoop);
    housekeeping_thread.detach();
    supervisionLoop();
    return 0;
}
