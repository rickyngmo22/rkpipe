// 任务规格解析 / YAML 生成 / 子进程拉起停止 / 配套加载与热切换（自守护进程入口拆出）。
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
#include "io/http_request.h"

using mini_json::JsonValue;
using mini_json::JsonParser;

// ---- 全局状态（daemon_internal.h 中 extern 声明；定义唯一在本单元）----
DaemonOptions g_opt;
std::mutex g_tasks_mutex;
std::vector<std::unique_ptr<Task>> g_tasks;
std::atomic<bool> g_shutdown{false};
std::atomic<int64_t> g_config_mtime{0};
int g_next_id = 1;
// 事件闭环（main 里初始化，进程生命周期内常驻）
EventStore* g_events = nullptr;
EventFlywheel* g_flywheel = nullptr;
LlmReviewer* g_reviewer = nullptr;
std::string g_forward_url_full_;
std::string g_forward_host;
int g_forward_port = 0;
std::string g_forward_path = "/";
bool g_forward_enabled = false;

#include "daemon/events_store.h"
#include "daemon/llm_review.h"
#include "daemon/mini_json.h"
#include "utils/json_escape.h"
#include "daemon/daemon_internal.h"

using mini_json::JsonValue;
using mini_json::JsonParser;
#include "daemon/mini_json.h"
#include "utils/json_escape.h"

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}


// ---------------------------------------------------------------------------
// 事件/告警透传（event_*/alert_* → 子进程 yaml）与 extra_yaml
// ---------------------------------------------------------------------------
// 透传键白名单：event_*/alert_*（与子进程 yaml 键名一致）+ extra_yaml。
// 键名限定 [a-z0-9_]，防 REST query 注入任意 yaml 键
bool isPassthroughKey(const std::string& key) {
    if (key == "extra_yaml") {
        return true;
    }
    if (key.rfind("record_", 0) == 0) {  // 录像分段治理键透传给子进程
        return true;
    }
    if (key.rfind("event_", 0) != 0 && key.rfind("alert_", 0) != 0) {
        return false;
    }
    for (const char c : key) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
            return false;
        }
    }
    return true;
}

// JSON 标量 → 透传字符串值（bool: 1/0；number: %g 紧凑字面量；string: 原样；其余忽略）
std::string passthroughValueToString(const JsonValue& v) {
    switch (v.type) {
        case JsonValue::Type::Bool: return v.b ? "1" : "0";
        case JsonValue::Type::Number: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%g", v.num);
            return buf;
        }
        case JsonValue::Type::String: return v.str;
        default: return "";
    }
}

// yaml 双引号字符串（转义 + 剥离控制字符，防值内换行注入 yaml 键）
std::string yamlQuoted(const std::string& value) {
    std::string out = "\"";
    for (const char c : value) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (static_cast<unsigned char>(c) >= 0x20) {
            out += c;
        }
    }
    out += "\"";
    return out;
}

// yaml 标量输出：纯数字字面量不加引号（cv::FileStorage 按 int/double 读取
// event_dwell_seconds/alert_min_count 等，带引号会变 string 读不进来）；其余加双引号
std::string yamlScalar(const std::string& value) {
    bool numeric = !value.empty();
    for (const char c : value) {
        if (!((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')) {
            numeric = false;
            break;
        }
    }
    return numeric ? value : yamlQuoted(value);
}

// 设置/删除一条透传键（value 为空 = 删除该键，用于 REST 更新时清除）
// （定义在 TaskSpec 之后）

// ---------------------------------------------------------------------------
// 任务定义
// ---------------------------------------------------------------------------
void setPassthrough(TaskSpec& spec, const std::string& key, const std::string& value) {
    spec.passthrough.erase(std::remove_if(spec.passthrough.begin(), spec.passthrough.end(),
                                          [&](const std::pair<std::string, std::string>& kv) {
                                              return kv.first == key;
                                          }),
                           spec.passthrough.end());
    if (!value.empty()) {
        spec.passthrough.emplace_back(key, value);
    }
}

// 事件规则字段白名单（streams.json rules 数组项 → 子进程 yaml event_rules 键）
bool isEventRuleKey(const std::string& key) {
    static const char* kAllowed[] = {"id",         "type",           "line",       "region",
                                     "classes",    "direction",      "dwell_seconds",
                                     "absence_seconds", "seconds",    "min_count",
                                     "fall_seconds", "fall_aspect", "llm_hint"};
    for (const char* k : kAllowed) {
        if (key == k) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// YAML 生成（与 stream_multi_web 保持一致）
// ---------------------------------------------------------------------------
std::string defaultModelForTask(const std::string& task, const std::string& root) {
    if (task == "detect") return root + "/model/yolov5s.rknn";
    if (task == "pose") return root + "/model/yolov8_pose.rknn";
    if (task == "obb") return root + "/model/yolov8_obb.rknn";
    if (task == "seg") return root + "/model/yolo11n_seg.rknn";
    if (task == "sem") return root + "/model/yolo26n_sem.rknn";
    return root + "/model/yolov5s.rknn";
}

std::string previewModeForInput(const std::string& input) {
    if (input.rfind("rtsp://", 0) == 0 || input.rfind("rtmp://", 0) == 0 || input.rfind("http://", 0) == 0 ||
        input.rfind("https://", 0) == 0 || input.rfind("/dev/video", 0) == 0) {
        return "live";
    }
    return "replay";
}

// 由推流地址推导 WebRTC 观看地址（MediaMTX 约定）：
//   rtmp://HOST:1935/live/key -> http://HOST:8889/live/key/
//   srt://HOST:8890/live/key  -> http://HOST:8889/live/key/（兼容 streamid 完整形式）
// 其余协议返回空（该窗格回退 MJPEG）。端口用 RK_PIPE_WEBRTC_PORT 覆盖。
// HEVC 编码时默认返回空（浏览器 WebRTC 对 HEVC 支持不稳定，Chrome 桌面多不支持，
// 会显示 "codecs not supported by client"）；export RK_PIPE_WEBRTC_HEVC=1 可强制保留
// WebRTC（如 Safari 播放 HEVC）。
std::string deriveWebrtcUrl(const std::string& output, const std::string& codec) {
    std::string lower = output;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (codec == "hevc") {
        const char* hevc_env = std::getenv("RK_PIPE_WEBRTC_HEVC");
        const bool force_hevc = hevc_env && *hevc_env && std::string(hevc_env) != "0";
        if (!force_hevc) {
            return "";
        }
    }
    const bool is_rtmp = lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0;
    const bool is_srt = lower.rfind("srt://", 0) == 0;
    if (!is_rtmp && !is_srt) {
        return "";
    }
    const std::string rest = output.substr(output.find("://") + 3);
    // 流路径：RTMP 直接取斜杠后；SRT 兼容直观路径或 streamid 的 r= 段
    std::string path;
    if (is_srt) {
        const size_t qmark = rest.find('?');
        const size_t slash = rest.find('/');
        if (qmark != std::string::npos) {
            const std::string query = rest.substr(qmark + 1);
            std::string sid;
            const size_t sp = query.find("streamid=");
            if (sp != std::string::npos) {
                sid = query.substr(sp + 9);  // "streamid=" 共 9 字符
                const size_t amp = sid.find('&');
                if (amp != std::string::npos) {
                    sid = sid.substr(0, amp);
                }
            }
            const size_t rp = sid.find("r=");
            if (rp != std::string::npos) {
                sid = sid.substr(rp + 2);
                const size_t comma = sid.find(',');
                if (comma != std::string::npos) {
                    sid = sid.substr(0, comma);
                }
                path = sid;
            }
        } else if (slash != std::string::npos) {
            path = rest.substr(slash + 1);
        }
        if (path.empty()) {
            return "";
        }
    } else {
        const size_t slash = rest.find('/');
        path = slash == std::string::npos ? "" : rest.substr(slash + 1);
    }
    const size_t hpe = rest.find_first_of("/?");
    const std::string host_port = hpe == std::string::npos ? rest : rest.substr(0, hpe);
    const size_t colon = host_port.rfind(':');
    const std::string host = colon == std::string::npos ? host_port : host_port.substr(0, colon);
    const char* port_env = std::getenv("RK_PIPE_WEBRTC_PORT");
    const std::string wport = (port_env && *port_env) ? port_env : "8889";
    return "http://" + host + ":" + wport + "/" + path + "/";
}

bool isNetworkOutputUrl(const std::string& output) {
    std::string lower = output;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("rtsp://", 0) == 0 || lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 || lower.rfind("srt://", 0) == 0 ||
           lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0;
}

// 兼容 `output` 地址尾部追加 `,codec=h264|hevc|h265` 的写法：
// 将该片段从地址中剥离（否则会污染 SRT streamid / 流路径），解析出的编码写入 codec。
// 显式 codec 字段优先，此函数只作为兜底。
std::string extractCodecFromOutput(const std::string& output, std::string& codec) {
    std::string url = output;
    const std::string marker = ",codec=";
    const size_t pos = url.rfind(marker);
    if (pos != std::string::npos) {
        const std::string val = url.substr(pos + marker.size());
        if (val == "h264" || val == "hevc" || val == "h265") {
            codec = (val == "hevc" || val == "h265") ? "hevc" : "h264";
            url.erase(pos);
        }
    }
    return url;
}

std::string taskBlockYaml(const std::string& task, const std::string& root, bool track_on) {
    // 各任务一视同仁：仅 JSON/REST 显式 tracking=1 才开（detect 同样默认关，与 stream_multi_web 一致）
    if (task == "detect") {
        return "label_path: \"" + root + "/model/coco_80_labels_list.txt\"\n"
               "obj_class_num: 80\n"
               "enable_tracking: " + (track_on ? "1" : "0") + "\n";
    }
    if (task == "pose") {
        return "label_path: \"" + root + "/model/coco_80_labels_list.txt\"\n"
               "obj_class_num: 1\n"
               "enable_tracking: " + (track_on ? "1" : "0") + "\n";
    }
    if (task == "obb") {
        return "label_path: \"" + root + "/model/yolov8_obb_labels_list.txt\"\n"
               "obj_class_num: 15\n"
               "conf_threshold: 0.4\n"
               "nms_threshold: 0.5\n"
               "enable_tracking: " + (track_on ? "1" : "0") + "\n";
    }
    if (task == "seg") {
        return "label_path: \"" + root + "/model/coco_80_labels_list.txt\"\n"
               "obj_class_num: 80\n"
               "seg_mask: 1\n"
               "enable_tracking: " + (track_on ? "1" : "0") + "\n";
    }
    if (task == "sem") {
        return "label_path: \"" + root + "/model/cityscapes_19_labels_list.txt\"\n"
               "obj_class_num: 19\n";
    }
    std::string block;
    if (track_on) {
        block += "track_iou_threshold: 0.30\n"
                 "track_max_missed: 35\n"
                 "track_min_confirm_hits: 4\n"
                 "track_reid_iou_threshold: 0.12\n"
                 "track_center_distance_threshold: 2.4\n"
                 "track_box_smooth_alpha: 0.62\n"
                 "track_render_max_missed: 1\n"
                 "track_high_conf_threshold: 0.50\n"
                 "track_low_conf_threshold: 0.25\n"
                 "track_stage2_iou_threshold: 0.50\n"
                 "track_size_ratio_threshold: 0.30\n"
                 "track_max_tracks: 256\n"
                 "track_algorithm: \"ocsort\"\n";
    }
    return block;
}

// ---------------------------------------------------------------------------
// 子进程管理
// ---------------------------------------------------------------------------
// 自动 NPU 起始核：按任务 id 顺序累计线程数做全局偏移。
// 多路 = 多进程，若每路都从 core 0 起轮转，threads=2 时全部挤在前两个 NPU，第三个闲置。
// 调用方需持有 g_tasks_mutex（writeYaml/spawnTask 均由持锁路径调用）。
int autoNpuCoreOffset(const Task& task) {
    int offset = 0;
    for (const auto& t : g_tasks) {
        if (t->id == task.id) break;
        offset += std::atoi(t->spec.threads.empty() ? "3" : t->spec.threads.c_str());
    }
    return offset % 3;
}

bool writeYaml(const Task& task, const std::vector<int>& extra_ports) {
    std::ofstream out(task.yaml_path);
    if (!out.is_open()) {
        return false;
    }
    const std::string& input = task.spec.input;
    out << "%YAML:1.0\n---\n";
    out << "model_path: \"" << task.spec.model << "\"\n";
    out << "input_path: \"" << input << "\"\n";
    out << "output_video_path: \"" << task.spec.output << "\"\n";
    out << "output_backend: \"ffmpeg\"\n";
    out << "web_preview: 1\n";
    out << "web_preview_bind: \"" << g_opt.bind << "\"\n";
    out << "web_preview_port: " << task.port << "\n";
    out << "web_preview_quality: " << g_opt.quality << "\n";
    out << "web_preview_mode: \"" << previewModeForInput(input) << "\"\n";
    out << "web_preview_eof_linger_ms: 1500\n";
    out << "web_preview_replay_pace: " << (task.spec.pace.empty() ? "0" : task.spec.pace) << "\n";
    out << "task: \"" << task.spec.task << "\"\n";
    out << "mode: \"pipeline\"\n";
    out << "thread_count: " << (task.spec.threads.empty() ? "3" : task.spec.threads) << "\n";
    out << "gui: 0\n";
    out << "local: 0\n";
    if (!task.spec.npu_core_mask.empty()) {
        out << "npu_core_mask: " << task.spec.npu_core_mask << "\n";
    }
    // 显式 npu_core_start 优先；否则按累计线程数自动错开起始核（与 stream_multi_web 一致）
    const std::string cs = task.spec.npu_core_start;
    const bool cs_valid = cs == "0" || cs == "1" || cs == "2";
    const std::string core_start = cs_valid ? cs : std::to_string(autoNpuCoreOffset(task));
    out << "npu_core_start: " << core_start << "\n";
    if (!task.spec.web_preview_scale.empty()) {
        out << "web_preview_scale: " << task.spec.web_preview_scale << "\n";
    }
    // 首任务聚合所有路预览到主页
    if (task.id == 1 && !extra_ports.empty()) {
        std::string csv;
        for (size_t i = 0; i < extra_ports.size(); ++i) {
            if (i) csv += ",";
            csv += std::to_string(extra_ports[i]);
        }
        out << "web_preview_extra_ports: \"" << csv << "\"\n";
    }
    // WebRTC 观看地址（推流经 MediaMTX 网关）：主任务聚合页按窗格顺序写全部，其余写自己
    std::string wrtc_csv;
    if (task.id == 1) {
        for (size_t i = 0; i < g_tasks.size(); ++i) {
            if (i) wrtc_csv += "|";
            wrtc_csv += deriveWebrtcUrl(g_tasks[i]->spec.output, g_tasks[i]->spec.codec);
        }
    } else {
        wrtc_csv = deriveWebrtcUrl(task.spec.output, task.spec.codec);
    }
    if (!wrtc_csv.empty()) {
        out << "web_preview_webrtc_urls: \"" << wrtc_csv << "\"\n";
    }
    // 多任务组合（Y5）：辅助模型 + 辅助任务 → aux_model_path / aux_task
    if (!task.spec.aux_model.empty()) {
        out << "aux_model_path: \"" << task.spec.aux_model << "\"\n";
        out << "aux_task: \"" << task.spec.aux_task << "\"\n";
    }
    const bool track_on = task.spec.tracking == "1";
    out << taskBlockYaml(task.spec.task, g_opt.root_dir, track_on);
    // 命名事件规则：streams.json rules 数组 → 子进程 yaml event_rules 列表
    if (!task.spec.rules.empty()) {
        // 数字键裸输出（int/double 语义），字符串键恒加引号（classes:"2" 等数字样字符串
        // 若裸输出会被 FileNode 读成 int，仍可转 string，但显式引号语义更稳）
        static const char* kNumericKeys[] = {"dwell_seconds", "absence_seconds", "seconds",
                                             "min_count",    "fall_seconds",    "fall_aspect"};
        out << "event_rules:\n";
        for (const auto& rule_kv : task.spec.rules) {
            bool first = true;
            for (const auto& kv : rule_kv) {
                bool numeric_key = false;
                for (const char* nk : kNumericKeys) {
                    if (kv.first == nk) {
                        numeric_key = true;
                        break;
                    }
                }
                out << (first ? "  - " : "    ") << kv.first << ": "
                    << (numeric_key ? yamlScalar(kv.second) : yamlQuoted(kv.second)) << "\n";
                first = false;
            }
        }
    }
    // 事件/告警透传字段（键名与子进程 yaml 一致；数字字面量不加引号保类型语义）
    for (const auto& kv : task.spec.passthrough) {
        out << kv.first << ": " << yamlScalar(kv.second) << "\n";
    }
    // extra_yaml 原样追加（高级用法：可覆盖上面的内置键，谨慎使用）
    if (!task.spec.extra_yaml.empty()) {
        out << task.spec.extra_yaml;
        if (task.spec.extra_yaml.back() != '\n') {
            out << "\n";
        }
    }
    return true;
}

int allocPort() {
    // 调用方需持有 g_tasks_mutex
    // 跳过已有任务端口与 daemon 自身端口（api_port 被任务 web_preview 占用
    // 会使 daemon 重启后无法 bind——实测踩过）
    int p = g_opt.base_port;
    for (;;) {
        bool used = (p == g_opt.api_port);
        if (!used) {
            for (const auto& t : g_tasks) {
                if (t->port == p) {
                    used = true;
                    break;
                }
            }
        }
        if (!used) {
            return p;
        }
        ++p;
    }
}

bool tcpPortFree(int port);  // 定义见 stopTask 之后

// ---------------------------------------------------------------------------
// 启动前资源检查（T2.2）：多进程防 RKNN errno:12（内存分配过大）。
// 在 spawn 前估算任务内存预算并与 /proc/meminfo MemAvailable 比对，
// 剩余内存低于保底值（RK_PIPE_MEM_MIN_FREE_MB，默认 256MB）时拒绝启动并告警；
// RK_PIPE_SKIP_MEM_CHECK=1 可关闭检查。
// ---------------------------------------------------------------------------
long long fileSizeBytes(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        return -1;
    }
    return static_cast<long long>(st.st_size);
}

long long memAvailKb() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long value = 0;
    std::string unit;
    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value;
        }
    }
    return -1;
}

// 任务内存预算估算（MB）：模型权重 ×2（权重+激活/推理缓冲）+ 每线程帧缓冲（~8MB，
// 双缓冲 1080p NV12）+ 固定开销（RKNN runtime / OpenCV / daemon 自身，约 220MB）。
int estimateTaskMemMb(const TaskSpec& spec) {
    const std::string model = spec.model.empty() ? defaultModelForTask(spec.task, g_opt.root_dir) : spec.model;
    const long long bytes = fileSizeBytes(model);
    const int model_mb = bytes > 0 ? static_cast<int>((bytes + (1 << 20) - 1) >> 20) : 30;
    int threads = std::atoi(spec.threads.empty() ? "3" : spec.threads.c_str());
    if (threads < 1) {
        threads = 3;
    }
    return model_mb * 2 + threads * 8 + 220;
}

bool resourceBudgetOk(const TaskSpec& spec, int* out_est_mb) {
    const int est_mb = estimateTaskMemMb(spec);
    if (out_est_mb) {
        *out_est_mb = est_mb;
    }
    const char* skip = std::getenv("RK_PIPE_SKIP_MEM_CHECK");
    if (skip && *skip && std::string(skip) != "0") {
        return true;
    }
    const long long avail = memAvailKb();
    if (avail < 0) {
        return true;  // 读不到内存信息不拦截
    }
    const char* floor_env = std::getenv("RK_PIPE_MEM_MIN_FREE_MB");
    const long long floor_mb = (floor_env && *floor_env) ? std::atoll(floor_env) : 256;
    const long long left_mb = avail / 1024 - est_mb;
    if (left_mb < floor_mb) {
        std::fprintf(stderr,
                     "[daemon][warn] resource budget: task '%s' est +%dMB, MemAvailable=%lldMB, "
                     "left=%lldMB < floor %lldMB. Refusing to spawn "
                     "(RK_PIPE_SKIP_MEM_CHECK=1 to bypass, RK_PIPE_MEM_MIN_FREE_MB to tune).\n",
                     spec.task.c_str(), est_mb, avail / 1024, left_mb, floor_mb);
        return false;
    }
    return true;
}

bool spawnTask(Task* task) {
    std::vector<int> extra;
    for (const auto& t : g_tasks) {
        if (t->id != task->id && t->pid > 0) {
            extra.push_back(t->port);
        }
    }
    if (task->port <= 0) {
        task->port = allocPort();
    }
    // 重启/热切换场景：等旧子进程释放预览端口（最多 4s），避免新子进程 WebPreviewServer bind 失败导致无预览
    for (int i = 0; i < 40 && !tcpPortFree(task->port); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (task->spec.model.empty()) {
        task->spec.model = defaultModelForTask(task->spec.task, g_opt.root_dir);
    }
    // 启动前资源检查（T2.2）：内存预算不足则拒绝 spawn（防多进程 RKNN errno:12）
    if (!resourceBudgetOk(task->spec, &task->est_mem_mb)) {
        return false;
    }
    if (!writeYaml(*task, extra)) {
        std::fprintf(stderr, "[daemon] write yaml failed: %s\n", task->yaml_path.c_str());
        return false;
    }

    const pid_t child = ::fork();
    if (child < 0) {
        return false;
    }
    if (child == 0) {
        // 子进程：日志重定向后 exec
        const int log_fd = ::open(task->log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (log_fd >= 0) {
            ::dup2(log_fd, STDOUT_FILENO);
            ::dup2(log_fd, STDERR_FILENO);
            ::close(log_fd);
        }
        if (!g_opt.metrics_file.empty()) {
            ::setenv("RK_PIPE_METRICS_FILE", (g_opt.metrics_file + "." + std::to_string(task->id)).c_str(), 1);
        }
        // 推流模式（网络输出或显式 RK_PIPE_FORCE_RKMPP=1）放开 rkmpp 硬编 + NV12 零拷贝，
        // 否则硬编被禁用、且零拷贝 buffer 提前归还导致推流端丢叠加框。
        const char* force = std::getenv("RK_PIPE_FORCE_RKMPP");
        const bool force_rkmpp = force && *force && std::string(force) != "0";
        const bool is_push = force_rkmpp || isNetworkOutputUrl(task->spec.output);
        if (!is_push) {
            ::setenv("RK_PIPE_DISABLE_RKMPP", "1", 1);
            ::setenv("RK_PIPE_DISABLE_ZERO_COPY_VIDEO", "1", 1);
        }
        // 编码切换：hevc = rkmpp 硬编 H.265（否则默认 h264_rkmpp）
        if (!task->spec.codec.empty()) {
            ::setenv("RK_PIPE_OUTPUT_CODEC", task->spec.codec.c_str(), 1);
        }
        // CPU 亲和性不再默认注入：多路并发时所有核本就被占满，强制绑到 4 个大核
        // 会造成过订阅/线程饥饿（表现像卡死）。需要时可自行 export：
        //   RK_PIPE_PIN_WORKER=4-7   推理 worker 绑 A76 大核（适合路数少、追求单路帧率）
        //   RK_PIPE_PIN_ENCODE=7     JPEG 编码绑大核（A55 编码太慢会拖垮预览）
        ::execl(g_opt.bin_path.c_str(), "console_detector", "--config", task->yaml_path.c_str(),
                static_cast<char*>(nullptr));
        ::fprintf(stderr, "[daemon] exec failed: %s\n", g_opt.bin_path.c_str());
        _exit(127);
    }

    task->pid = child;
    task->start_time_s = static_cast<int64_t>(std::time(nullptr));
    std::printf("[daemon] started task %d (task=%s input=%s port=%d pid=%d est_mem=%dMB)\n", task->id,
                task->spec.task.c_str(), task->spec.input.c_str(), task->port, child, task->est_mem_mb);
    return true;
}

void stopTask(Task* task) {
    if (task->pid > 0) {
        ::kill(task->pid, SIGTERM);
        // 等最多 3s 优雅退出
        for (int i = 0; i < 30; ++i) {
            int status = 0;
            if (::waitpid(task->pid, &status, WNOHANG) == task->pid) {
                task->pid = -1;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (task->pid > 0) {
            ::kill(task->pid, SIGKILL);
            // 轮询回收（最多 3s）：避免阻塞式 waitpid 在子进程卡硬件调用（D 状态）时挂死 daemon
            for (int i = 0; i < 30; ++i) {
                if (::waitpid(task->pid, nullptr, WNOHANG) == task->pid) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            task->pid = -1;
        }
    }
    task->wanted.store(false);
}

// 探测 tcp 端口当前是否可 bind（用于 spawn 前确认旧进程已释放端口）
bool tcpPortFree(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return true;
    }
    int opt = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    const int r = ::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    ::close(fd);
    return r == 0;
}

// ---------------------------------------------------------------------------
// 配置加载 / 热切换
// ---------------------------------------------------------------------------
int64_t fileMtime(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        return -1;
    }
    return static_cast<int64_t>(st.st_mtime);
}

std::vector<TaskSpec> loadConfig(const std::string& path) {
    const std::string target = path.empty() ? g_opt.streams_file : path;
    std::vector<TaskSpec> specs;
    std::ifstream in(target);
    if (!in.is_open()) {
        std::fprintf(stderr, "[daemon][warn] config open failed: %s\n", target.c_str());
        return specs;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    bool ok = false;
    JsonValue root = JsonParser(buf.str()).parse(ok);
    if (!ok) {
        std::fprintf(stderr, "[daemon][warn] config parse failed: %s\n", target.c_str());
        return specs;
    }
    if (const JsonValue* streams = root.get("streams")) {
        for (const JsonValue& item : streams->arr) {
            TaskSpec s;
            if (const JsonValue* v = item.get("id")) {
                s.id = v->type == JsonValue::Type::Number ? std::to_string(static_cast<int>(v->num)) : v->str;
            }
            s.task = item.getString("task", "detect");
            s.input = item.getString("input");
            s.model = item.getString("model");
            if (const JsonValue* v = item.get("threads")) {
                if (v->type == JsonValue::Type::Number) {
                    s.threads = std::to_string(static_cast<int>(v->num));
                } else {
                    s.threads = v->str;
                }
            }
            if (const JsonValue* v = item.get("port")) {
                if (v->type == JsonValue::Type::Number) {
                    s.port = std::to_string(static_cast<int>(v->num));
                } else {
                    s.port = v->str;
                }
            }
            if (const JsonValue* v = item.get("pace")) {
                s.pace = v->type == JsonValue::Type::Number ? std::to_string(static_cast<int>(v->num)) : v->str;
            }
            if (const JsonValue* v = item.get("npu_core_mask")) {
                s.npu_core_mask =
                    v->type == JsonValue::Type::Number ? std::to_string(static_cast<int>(v->num)) : v->str;
            }
            if (const JsonValue* v = item.get("npu_core_start")) {
                s.npu_core_start =
                    v->type == JsonValue::Type::Number ? std::to_string(static_cast<int>(v->num)) : v->str;
            }
            if (const JsonValue* v = item.get("web_preview_scale")) {
                s.web_preview_scale =
                    v->type == JsonValue::Type::Number ? std::to_string(v->num) : v->str;
            }
            if (const JsonValue* v = item.get("tracking")) {
                s.tracking =
                    v->type == JsonValue::Type::Number ? std::to_string(static_cast<int>(v->num)) : v->str;
            }
            s.aux_model = item.getString("aux_model");
            s.aux_task = item.getString("aux_task", "detect");
            s.output = item.getString("output");
            // output 尾部 `,codec=...` 兼容写法剥离，显式 codec 字段优先
            std::string url_codec;
            s.output = extractCodecFromOutput(s.output, url_codec);
            s.codec = item.getString("codec");
            if (s.codec.empty() && !url_codec.empty()) {
                s.codec = url_codec;
            }
            // 嵌套 events{}/alert{} 对象 → 展开为 event_*/alert_* 透传键
            // （比扁平键可读；alert.snapshot_dir 是告警抓拍目录，周期快照用 periodic_snapshot_dir）
            static const struct {
                const char* obj;
                const char* sub;
                const char* yaml_key;
            } kNestedPassthrough[] = {
                {"events", "line", "event_line"},
                {"events", "region", "event_region"},
                {"events", "classes", "event_classes"},
                {"events", "line_cross", "event_line_cross"},
                {"events", "intrusion", "event_intrusion"},
                {"events", "dwell", "event_dwell"},
                {"events", "dwell_seconds", "event_dwell_seconds"},
                {"events", "absence", "event_absence"},
                {"events", "absence_seconds", "event_absence_seconds"},
                {"alert", "enabled", "alert_enabled"},
                {"alert", "webhook_url", "alert_webhook_url"},
                {"alert", "min_count", "alert_min_count"},
                {"alert", "interval_s", "alert_interval_s"},
                {"alert", "snapshot_dir", "alert_snapshot_dir"},
                {"alert", "dedup_interval_s", "alert_dedup_interval_s"},
                {"alert", "dedup_state_dir", "alert_dedup_state_dir"},
                {"alert", "periodic_snapshot_dir", "snapshot_dir"},
                {"alert", "snapshot_interval_s", "snapshot_interval_s"},
            };
            for (const auto& np : kNestedPassthrough) {
                const JsonValue* obj = item.get(np.obj);
                if (!obj) {
                    continue;
                }
                if (const JsonValue* v = obj->get(np.sub)) {
                    setPassthrough(s, np.yaml_key, passthroughValueToString(*v));
                }
            }
            // 扁平 event_*/alert_* 键与 extra_yaml 原样透传（后写覆盖嵌套对象展开结果）
            for (const auto& kv : item.obj) {
                if (!isPassthroughKey(kv.first)) {
                    continue;
                }
                if (kv.first == "extra_yaml") {
                    if (kv.second.type == JsonValue::Type::String) {
                        s.extra_yaml = kv.second.str;
                    }
                    continue;
                }
                setPassthrough(s, kv.first, passthroughValueToString(kv.second));
            }
            // 命名事件规则数组：rules: [{id,type,line/region,classes,direction,...}, ...]
            if (const JsonValue* rules = item.get("rules")) {
                if (rules->type == JsonValue::Type::Array) {
                    for (const JsonValue& r : rules->arr) {
                        if (r.type != JsonValue::Type::Object) {
                            continue;
                        }
                        std::vector<std::pair<std::string, std::string>> rule_kv;
                        for (const auto& kv : r.obj) {
                            if (!isEventRuleKey(kv.first)) {
                                continue;
                            }
                            const std::string val = passthroughValueToString(kv.second);
                            if (!val.empty()) {
                                rule_kv.emplace_back(kv.first, val);
                            }
                        }
                        if (!rule_kv.empty()) {
                            s.rules.push_back(std::move(rule_kv));
                        }
                    }
                }
            }
            if (!s.input.empty()) {
                specs.push_back(std::move(s));
            }
        }
    }
    // 顶层可选字段：MJPEG 质量（与 configs/streams/stream_multi_web.sample.json 同格式）
    if (const JsonValue* v = root.get("quality")) {
        if (v->type == JsonValue::Type::Number && v->num >= 1 && v->num <= 100) {
            g_opt.quality = static_cast<int>(v->num);
        }
    }
    return specs;
}

bool sameSpec(const TaskSpec& a, const TaskSpec& b) {
    return a.task == b.task && a.input == b.input && a.model == b.model && a.threads == b.threads &&
           a.port == b.port && a.pace == b.pace && a.npu_core_mask == b.npu_core_mask &&
           a.npu_core_start == b.npu_core_start && a.web_preview_scale == b.web_preview_scale &&
           a.output == b.output && a.codec == b.codec && a.tracking == b.tracking &&
           a.aux_model == b.aux_model && a.aux_task == b.aux_task &&
           a.passthrough == b.passthrough && a.extra_yaml == b.extra_yaml && a.rules == b.rules;
}

// 任务身份匹配：显式 id 优先（允许完全相同的 input/task/model 跑多路）；
// 无 id 时回退到 (input, task, model)。
bool taskSpecSameIdentity(const TaskSpec& a, const TaskSpec& b) {
    if (!a.id.empty() || !b.id.empty()) {
        return !a.id.empty() && a.id == b.id;
    }
    return a.input == b.input && a.task == b.task && a.model == b.model;
}

// 任务 id 分配：仅纯数字字符串才作为显式 id（如 id="3"），否则分配自增 id。
// ⚠ 曾用 std::atoi 直接解析非数字 id（如 "b10"）→ 全部塌缩为 1，导致所有任务共用同一
//   daemon_1.yaml/log，子进程读到彼此的 web_preview_port → bind "Address already in use" 崩溃。
int nextTaskId(const std::string& spec_id) {
    if (!spec_id.empty()) {
        bool digits = true;
        for (char c : spec_id) {
            if (c < '0' || c > '9') {
                digits = false;
                break;
            }
        }
        if (digits) {
            const int id = std::max(1, std::atoi(spec_id.c_str()));
            if (id >= g_next_id) {
                g_next_id = id + 1;
            }
            return id;
        }
    }
    return g_next_id++;
}

// ---------------------------------------------------------------------------
// 动态任务持久化（REST 创建/修改的任务写 dynamic_tasks.json，daemon 重启自动恢复）
// ---------------------------------------------------------------------------
// passthrough 平铺为 event_*/alert_* 字符串键（loadConfig 按同名扁平键读回，round-trip 一致）
std::string taskSpecJson(const TaskSpec& s) {
    std::ostringstream j;
    j << "{\"id\":\"" << jsonEscape(s.id) << "\",\"task\":\"" << jsonEscape(s.task)
      << "\",\"input\":\"" << jsonEscape(s.input) << "\",\"model\":\"" << jsonEscape(s.model)
      << "\",\"threads\":\"" << jsonEscape(s.threads) << "\",\"port\":\"" << jsonEscape(s.port)
      << "\",\"pace\":\"" << jsonEscape(s.pace) << "\",\"npu_core_mask\":\"" << jsonEscape(s.npu_core_mask)
      << "\",\"npu_core_start\":\"" << jsonEscape(s.npu_core_start)
      << "\",\"web_preview_scale\":\"" << jsonEscape(s.web_preview_scale)
      << "\",\"output\":\"" << jsonEscape(s.output) << "\",\"codec\":\"" << jsonEscape(s.codec)
      << "\",\"tracking\":\"" << jsonEscape(s.tracking) << "\",\"aux_model\":\"" << jsonEscape(s.aux_model)
      << "\",\"aux_task\":\"" << jsonEscape(s.aux_task) << "\",\"extra_yaml\":\"" << jsonEscape(s.extra_yaml)
      << "\"";
    for (const auto& kv : s.passthrough) {
        j << ",\"" << kv.first << "\":\"" << jsonEscape(kv.second) << "\"";
    }
    j << ",\"rules\":[";
    for (size_t i = 0; i < s.rules.size(); ++i) {
        if (i) {
            j << ",";
        }
        j << "{";
        for (size_t k = 0; k < s.rules[i].size(); ++k) {
            if (k) {
                j << ",";
            }
            j << "\"" << jsonEscape(s.rules[i][k].first) << "\":\""
              << jsonEscape(s.rules[i][k].second) << "\"";
        }
        j << "}";
    }
    j << "]}";
    return j.str();
}

// 调用方需持有 g_tasks_mutex。任务数量少，整体重写（tmp + rename 原子替换）
void saveDynamicTasks() {
    std::ostringstream j;
    j << "{\"streams\":[";
    bool first = true;
    for (const auto& t : g_tasks) {
        if (!t->dynamic) {
            continue;
        }
        if (!first) {
            j << ",";
        }
        first = false;
        j << taskSpecJson(t->spec);
    }
    j << "]}";
    const std::string tmp = g_opt.dynamic_file + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << j.str();
    }
    std::error_code ec;
    std::filesystem::rename(tmp, g_opt.dynamic_file, ec);
    if (ec) {
        std::fprintf(stderr, "[daemon][warn] save dynamic tasks failed: %s\n", ec.message().c_str());
    }
}

void applyConfig(const std::vector<TaskSpec>& specs);

// streams.json + 动态任务合并加载并应用（启动/热切换共用）；返回配置的任务总数。
// reload 不会丢 REST 动态创建的任务（它们在 dynamic_tasks.json 里持久化）
int reloadAndApply(const std::string& override_file) {
    std::vector<TaskSpec> dyn;
    std::error_code fs_ec;
    if (std::filesystem::exists(g_opt.dynamic_file, fs_ec)) {  // 首启不存在属正常，不告警
        dyn = loadConfig(g_opt.dynamic_file);
    }
    std::vector<TaskSpec> specs = loadConfig(override_file);
    specs.insert(specs.end(), dyn.begin(), dyn.end());
    applyConfig(specs);
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        for (const auto& s : dyn) {
            for (auto& t : g_tasks) {
                if (taskSpecSameIdentity(t->spec, s)) {
                    t->dynamic = true;
                    break;
                }
            }
        }
    }
    return static_cast<int>(specs.size());
}

// 热切换：与现有任务对比，新增/重启变更任务、停止已删除任务。
// 任务身份：JSON 显式 id 优先，否则按 (input, task, model) 匹配。
void applyConfig(const std::vector<TaskSpec>& specs) {
    std::lock_guard<std::mutex> lock(g_tasks_mutex);

    // 1) 停止不再配置的任务
    for (auto& t : g_tasks) {
        bool still_wanted = false;
        for (const auto& s : specs) {
            if (taskSpecSameIdentity(t->spec, s)) {
                still_wanted = true;
                break;
            }
        }
        if (!still_wanted && t->wanted.load()) {
            std::printf("[daemon] hot-swap: stopping removed task %d (input=%s)\n", t->id,
                        t->spec.input.c_str());
            stopTask(t.get());
        }
    }

    // 2) 新增或重启变更任务
    for (const auto& s : specs) {
        Task* match = nullptr;
        for (const auto& t : g_tasks) {
            if (taskSpecSameIdentity(t->spec, s)) {
                match = t.get();
                break;
            }
        }
        if (match && sameSpec(match->spec, s)) {
            // 已存在且未变化
        } else if (match) {
            // 配置变化 → 热切换（重启）
            std::printf("[daemon] hot-swap: restarting task %d (input=%s) with new config\n", match->id,
                        match->spec.input.c_str());
            stopTask(match);
            match->spec = s;
            match->wanted.store(true);
            match->restarts = 0;
            match->give_up = false;
            match->pending_restart = false;
            match->restart_window.clear();
            spawnTask(match);
        } else {
            // 新增任务
            auto task = std::make_unique<Task>();
            task->id = nextTaskId(s.id);
            task->spec = s;
            task->port = s.port.empty() ? 0 : std::atoi(s.port.c_str());
            task->yaml_path = g_opt.dump_dir + "/daemon_" + std::to_string(task->id) + ".yaml";
            task->log_path = g_opt.dump_dir + "/daemon_" + std::to_string(task->id) + ".log";
            if (!spawnTask(task.get())) {
                std::fprintf(stderr, "[daemon] spawn failed for task %d\n", task->id);
            }
            g_tasks.push_back(std::move(task));
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP 服务
// ---------------------------------------------------------------------------
std::string taskToJson(const Task& t) {
    const bool running = t.pid > 0;
    const int64_t uptime = running ? static_cast<int64_t>(std::time(nullptr)) - t.start_time_s : 0;
    std::ostringstream j;
    j << "{\"id\":" << t.id
      << ",\"spec_id\":\"" << jsonEscape(t.spec.id)
      << "\",\"task\":\"" << jsonEscape(t.spec.task)
      << "\",\"input\":\"" << jsonEscape(t.spec.input)
      << "\",\"model\":\"" << jsonEscape(t.spec.model)
      << "\",\"threads\":\"" << jsonEscape(t.spec.threads)
      << "\",\"npu_core_mask\":\"" << jsonEscape(t.spec.npu_core_mask)
      << "\",\"npu_core_start\":\"" << jsonEscape(t.spec.npu_core_start)
      << "\",\"pace\":\"" << jsonEscape(t.spec.pace)
      << "\",\"codec\":\"" << jsonEscape(t.spec.codec)
      << "\",\"tracking\":\"" << jsonEscape(t.spec.tracking)
      << "\",\"aux_model\":\"" << jsonEscape(t.spec.aux_model)
      << "\",\"aux_task\":\"" << jsonEscape(t.spec.aux_task)
      << "\",\"extra_yaml\":\"" << jsonEscape(t.spec.extra_yaml)
      << "\",\"passthrough\":{";
    for (size_t i = 0; i < t.spec.passthrough.size(); ++i) {
        if (i) {
            j << ",";
        }
        j << "\"" << jsonEscape(t.spec.passthrough[i].first) << "\":\""
          << jsonEscape(t.spec.passthrough[i].second) << "\"";
    }
    j << "},\"rules\":[";
    for (size_t i = 0; i < t.spec.rules.size(); ++i) {
        if (i) {
            j << ",";
        }
        j << "{";
        for (size_t k = 0; k < t.spec.rules[i].size(); ++k) {
            if (k) {
                j << ",";
            }
            j << "\"" << jsonEscape(t.spec.rules[i][k].first) << "\":\""
              << jsonEscape(t.spec.rules[i][k].second) << "\"";
        }
        j << "}";
    }
    j << "]"
      << ",\"port\":" << t.port
      << ",\"est_mem_mb\":" << t.est_mem_mb
      << ",\"pid\":" << (running ? t.pid : -1)
      << ",\"status\":\"" << (running ? "running" : "stopped")
      << "\",\"uptime_s\":" << uptime
      << ",\"restarts\":" << t.restarts
      << ",\"webrtc\":\"" << jsonEscape(deriveWebrtcUrl(t.spec.output, t.spec.codec))
      << "\",\"preview\":\"http://<board-ip>:" << t.port << "/\"}";
    return j.str();
}

// ---------------------------------------------------------------------------
// 板载负载监控 + 日志尾部读取（监控网页用）
// ---------------------------------------------------------------------------
