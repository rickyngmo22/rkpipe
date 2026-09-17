#pragma once

// rk_pipe_daemon 内部共享声明（私有头，不随 SDK 安装）。
// types/状态与跨编译单元函数集中在此（自守护进程入口拆分）。
// 全局状态(g_*)定义在守护进程入口，各单元经 extern 引用。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

#include "io/http_request.h"
#include "utils/http_post.h"
#include "daemon/events_store.h"
#include "daemon/flywheel.h"
#include "daemon/llm_review.h"
#include "daemon/mini_json.h"

// ---- 监控页（daemon_pages.cc）----
std::string timelineHtml();
std::string monitorHtml();

// ---- 板载负载与文件工具（daemon_sysinfo.cc）----
std::string boardLoadJson();
std::string tailFile(const std::string& path, int n);

// ---- HTTP 收发原语（daemon_http.cc）----
bool sendAll(int fd, const void* data, size_t size);
void sendResponse(int fd, const std::string& status, const std::string& content_type, const std::string& body);
void sendJson(int fd, const std::string& body, const std::string& status = "200 OK");
bool parseHttpUrl(const std::string& url, std::string* host, int* port, std::string* path);
int connectTimeout(const std::string& host, int port, int timeout_ms);
std::string httpGetBody(const std::string& host, int port, const std::string& path, int timeout_ms = 1500);
bool httpPostJson(const std::string& host, int port, const std::string& path, const std::string& body,
                  int timeout_ms = 1500);

// ---- 任务模型（守护进程各单元共用）----
struct TaskSpec {
    std::string id;               // 可选：任务身份标识（允许完全相同的 input/task/model 跑多路）
    std::string task = "detect";
    std::string input;
    std::string model;
    std::string threads = "3";
    std::string port;             // 空=自动分配
    std::string pace = "0";
    std::string npu_core_mask;
    std::string npu_core_start;   // 可选：NPU 轮转起始核(0-2)；空=按累计线程数自动错开
    std::string web_preview_scale;
    std::string output;           // 推流地址（rtmp/rtsp/udp/srt 或本地文件），空=仅预览
    std::string codec;            // 输出编码 h264|hevc（空=默认 h264）
    std::string tracking;         // 可选：0=关 1=开（所有任务含 detect 默认关，与 stream_multi_web 一致）；空=0
    std::string aux_model;        // 可选：多任务组合（Y5）辅助模型路径；空=不启用
    std::string aux_task = "detect";
    // 事件/告警透传（事件规则引擎 D1/D2 + webhook 告警）：键名与子进程 yaml 一致
    // （event_line/event_line_cross/event_region/event_intrusion/.../alert_enabled/alert_webhook_url/...）。
    // streams.json 每路可直接写扁平 event_*/alert_* 键，或嵌套 events{}/alert{} 对象（见 loadConfig）；
    // REST 侧同名 query 参数可用于 POST /api/tasks 与 /api/tasks/<id>/update
    std::vector<std::pair<std::string, std::string>> passthrough;
    std::string extra_yaml;       // 高级逃生口：原样追加到生成 yaml 末尾（可覆盖内置键，谨慎使用）
    // 命名事件规则（streams.json 每路 rules 数组 → 子进程 yaml event_rules 列表）。
    // 每条规则为键值对列表，支持键：id/type/line/region/classes/direction/
    // dwell_seconds/absence_seconds/seconds/min_count（值统一字符串化，数字写 yaml 不带引号）
    std::vector<std::vector<std::pair<std::string, std::string>>> rules;
};

// 设置/删除一条透传键（value 为空 = 删除该键，用于 REST 更新时清除）

struct Task {
    int id = 0;
    TaskSpec spec;
    int port = 0;
    std::string yaml_path;
    std::string log_path;
    pid_t pid = -1;
    int restarts = 0;
    int est_mem_mb = 0;   // 启动前资源检查估算的内存占用（MB，观测用）
    int64_t start_time_s = 0;
    std::atomic<bool> wanted{true};
    // REST 动态创建的任务：持久化到 dynamic_tasks.json，daemon 重启后自动恢复
    bool dynamic = false;
    // 崩溃自动拉起（指数退避 + 10 分钟滑动窗口限次）
    bool pending_restart = false;
    bool give_up = false;
    std::chrono::steady_clock::time_point next_restart_at{};
    std::deque<std::chrono::steady_clock::time_point> restart_window;
};

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

struct DaemonOptions {
    std::string streams_file;
    std::string bind = "0.0.0.0";
    int api_port = 8099;
    int base_port = 8090;
    int quality = 60;  // 监控面板内嵌多路 MJPEG，画质不需太高，省带宽
    // 运行目录：子进程 yaml/log、events.jsonl、dynamic_tasks.json 都落在此。
    // 空 = <root_dir>/daemon_runs（即 /userdata 下，而非 /tmp）——/tmp 会被系统
    // 清理，REST 热加的任务与事件历史会随之丢失。
    std::string dump_dir;
    std::string bin_path;   // 空 = <root_dir>/build/console_detector
    std::string root_dir;   // 空 = 由 /proc/self/exe 反推仓根
    std::string metrics_file;
    double watch_s = 2.0;
    bool restart_on_failure = false;
    // 事件闭环：业务 webhook 转发地址（子进程告警指向 /internal/event 后，
    // 未启用复检或复检完成时 POST 到该地址；空=仅落盘可查）
    std::string forward_url;
    std::string flywheel_dir;  // 空=禁用；也接受环境变量 RK_PIPE_FLYWHEEL_DIR
    std::string events_file;   // 空=<dump_dir>/events.jsonl
    std::string dynamic_file;  // 空=<dump_dir>/dynamic_tasks.json
    // 磁盘治理（housekeeping，0=关闭对应项）：快照无界增长会写满 eMMC
    long long snapshot_max_mb = 1024;   // 各快照目录容量上限，超出删最旧
    int snapshot_max_age_days = 7;      // 快照保留天数
    long long log_max_mb = 16;          // 任务日志上限（尾部保留式轮转，同 inode）
    int housekeep_s = 600;              // housekeeping 周期（秒）
    std::string api_token;              // 非空时写操作（非 GET）需 token，GET 只读放行
};


// ---- 全局状态（定义在守护进程入口）----
extern DaemonOptions g_opt;
extern std::mutex g_tasks_mutex;
extern std::vector<std::unique_ptr<Task>> g_tasks;
extern std::atomic<bool> g_shutdown;
extern std::atomic<int64_t> g_config_mtime;
extern int g_next_id;
extern EventStore* g_events;
extern class EventFlywheel* g_flywheel;
extern LlmReviewer* g_reviewer;
extern std::string g_forward_url_full_;  // https 转发时的完整 URL（非空走 libcurl）
extern std::string g_forward_host;
extern int g_forward_port;
extern std::string g_forward_path;
extern bool g_forward_enabled;

// ---- 任务规格/生命周期（daemon_supervision.cc）----
// query 解析（parseQuery/urlDecode）声明移至 io/http_request.h（本头已包含）
long long memAvailKb();
std::vector<std::string> collectSnapshotDirs();  // daemon_sysinfo.cc
std::string yamlQuoted(const std::string& value);
std::string yamlScalar(const std::string& value);
bool isPassthroughKey(const std::string& key);
std::string passthroughValueToString(const mini_json::JsonValue& v);
void setPassthrough(TaskSpec& spec, const std::string& key, const std::string& value);
bool isEventRuleKey(const std::string& key);
std::string defaultModelForTask(const std::string& task, const std::string& root);
std::string previewModeForInput(const std::string& input);
std::string deriveWebrtcUrl(const std::string& output, const std::string& codec);
bool isNetworkOutputUrl(const std::string& output);
std::string extractCodecFromOutput(const std::string& output, std::string& codec);
std::string taskBlockYaml(const std::string& task, const std::string& root, bool track_on);
int autoNpuCoreOffset(const Task& task);
bool writeYaml(const Task& task, const std::vector<int>& extra_ports);
int allocPort();
int estimateTaskMemMb(const TaskSpec& spec);
bool resourceBudgetOk(const TaskSpec& spec, int* out_est_mb);
bool spawnTask(Task* task);
void stopTask(Task* task);
bool tcpPortFree(int port);
int64_t fileMtime(const std::string& path);
std::vector<TaskSpec> loadConfig(const std::string& path = "");
bool sameSpec(const TaskSpec& a, const TaskSpec& b);
bool taskSpecSameIdentity(const TaskSpec& a, const TaskSpec& b);
int nextTaskId(const std::string& spec_id);
std::string taskSpecJson(const TaskSpec& s);
std::string taskToJson(const Task& t);
void saveDynamicTasks();
void applyConfig(const std::vector<TaskSpec>& specs);
int reloadAndApply(const std::string& override_file = "");

// ---- Prometheus /metrics（daemon_prometheus.cc）----
struct PrometheusTaskSample {
    int id = 0;
    std::string task;        // 任务类型（label）
    std::string state;       // running / exited / stopped
    int64_t uptime_s = 0;
    bool child_ok = false;   // 子进程 status.json 是否可获取（不可得仅输出 task_info）
    double publish_fps = 0;
    long published_frames = 0;
    long dropped_frames = 0;
    long input_reconnects = 0;
};
std::string formatPrometheusMetrics(const std::string& board_json,
                                    const std::vector<PrometheusTaskSample>& tasks);

// ---- REST 服务（daemon_api.cc）----
void acceptLoop();
