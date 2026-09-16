// REST 路由与 HTTP 连接处理（自 daemon 主模块拆出）：handleRequest/handleClient/acceptLoop。
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

#include "core/build_info.h"
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

using mini_json::JsonValue;
#include "daemon/events_store.h"
#include "daemon/llm_review.h"
#include "daemon/mini_json.h"
#include "utils/json_escape.h"
#include "daemon/daemon_internal.h"

using mini_json::JsonValue;
#include "daemon/mini_json.h"
#include "utils/json_escape.h"

// 并行代理各子进程 /status.json（条带分片到 ≤8 线程；单子进程假死只损失它自己的
// child 字段，httpGetBody 1.5s 超时兜底）。调用方需持 g_tasks_mutex。
static std::vector<std::string> fetchChildStatusBodies() {
    std::vector<std::string> child_bodies(g_tasks.size());
    std::vector<size_t> running_idx;
    for (size_t i = 0; i < g_tasks.size(); ++i) {
        if (g_tasks[i]->pid > 0) {
            running_idx.push_back(i);
        }
    }
    const size_t kWorkers = std::min<size_t>(running_idx.size(), 8);
    std::vector<std::thread> fetchers;
    for (size_t k = 0; k < kWorkers; ++k) {
        fetchers.emplace_back([&child_bodies, &running_idx, k, kWorkers]() {
            for (size_t m = k; m < running_idx.size(); m += kWorkers) {
                const size_t idx = running_idx[m];
                child_bodies[idx] = httpGetBody("127.0.0.1", g_tasks[idx]->port, "/status.json");
            }
        });
    }
    for (auto& f : fetchers) {
        f.join();
    }
    return child_bodies;
}

void handleRequest(int client_fd, const std::string& method, const std::string& raw_path,
                   const std::string& body, const std::string& header_token) {
    // 拆分路径与 query
    const size_t qmark = raw_path.find('?');
    const std::string path = urlDecode(raw_path.substr(0, qmark));
    const std::string query = qmark == std::string::npos ? "" : raw_path.substr(qmark + 1);

    // CORS 预检
    if (method == "OPTIONS") {
        sendResponse(client_fd, "200 OK", "text/plain", "");
        return;
    }

    // 写操作鉴权：配置 api_token 后非 GET 请求需携带 token（GET 只读放行，监控页不受影响）。
    // 来源三选一：Authorization: Bearer <t> / X-API-Token: <t> / query ?token=
    // （/internal/event 亦属写操作：子进程 alert_webhook_url 末尾追加 ?token= 即可）
    if (!g_opt.api_token.empty() && method != "GET") {
        const auto qkv = parseQuery(query);
        const auto qt = qkv.find("token");
        const bool authed =
            (qt != qkv.end() && qt->second == g_opt.api_token) || header_token == g_opt.api_token;
        if (!authed) {
            sendJson(client_fd, "{\"error\":\"unauthorized: token required for write operations\"}",
                     "401 Unauthorized");
            return;
        }
    }

    // 监控网页
    if (method == "GET" && (path == "/" || path == "/monitor")) {
        sendResponse(client_fd, "200 OK", "text/html; charset=utf-8", monitorHtml());
        return;
    }

    // 事件时间线页（D4/M10）
    if (method == "GET" && path == "/timeline") {
        sendResponse(client_fd, "200 OK", "text/html; charset=utf-8", timelineHtml());
        return;
    }

    if (method == "GET" && path == "/api/health") {
        std::ostringstream j;
        j << "{\"ok\":true,\"tasks\":" << g_tasks.size() << ",\"api_port\":" << g_opt.api_port << "}";
        sendJson(client_fd, j.str());
        return;
    }

    // 事件流（G4.3 SSE）：text/event-stream 增量推送，事件/复检结论实时可见
    if (method == "GET" && (path == "/api/events/stream" || path == "/api/events/stream.sse")) {
        if (!g_events) {
            sendJson(client_fd, "{\"error\":\"events disabled\"}", "503 Service Unavailable");
            return;
        }
        static const std::string kSseHead =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if (!sendAll(client_fd, kSseHead.data(), kSseHead.size())) {
            return;
        }
        long last_version = -1;
        long last_id = 0;
        long last_review_id = 0;
        while (!g_shutdown.load()) {
            const long v = g_events->waitVersion(last_version, 15000);
            if (v == last_version) {
                static const std::string kKeep = ": keepalive\n\n";
                if (!sendAll(client_fd, kKeep.data(), kKeep.size())) {
                    return;  // 客户端断开
                }
                continue;
            }
            last_version = v;
            for (const auto& it : g_events->query("", "", last_id, 100)) {
                last_id = std::max(last_id, it.id);
                std::ostringstream data;
                data << "{\"id\":" << it.id << ",\"recv_ts\":" << it.recv_ts_ms
                     << ",\"body\":" << mini_json::dump(it.body);
                if (it.has_review) {
                    data << ",\"review\":{\"verdict\":\"" << it.review.verdict
                         << "\",\"source\":\"" << it.review.source
                         << "\",\"confidence\":" << it.review.confidence
                         << ",\"reason\":\"" << jsonEscape(it.review.reason) << "\"}";
                }
                data << "}";
                std::ostringstream ev;
                ev << "id: " << it.id << "\nevent: " << (it.has_review ? "review" : "event")
                   << "\ndata: " << data.str() << "\n\n";
                if (!sendAll(client_fd, ev.str().data(), ev.str().size())) {
                    return;  // 客户端断开
                }
            }
            // 复检结论挂在旧事件上（无新 item）时单独推送 review 帧
            const long lr = g_events->lastReviewEventId();
            if (lr > last_review_id) {
                last_review_id = lr;
                EventStore::Item it;
                if (g_events->get(lr, &it)) {
                    std::ostringstream ev;
                    ev << "id: " << it.id << "\nevent: review\ndata: {\"id\":" << it.id
                       << ",\"body\":" << mini_json::dump(it.body)
                       << ",\"review\":{\"verdict\":\"" << it.review.verdict
                       << "\",\"source\":\"" << it.review.source
                       << "\",\"confidence\":" << it.review.confidence
                       << ",\"reason\":\"" << jsonEscape(it.review.reason) << "\"}}\n\n";
                    if (!sendAll(client_fd, ev.str().data(), ev.str().size())) {
                        return;
                    }
                }
            }
        }
        return;
    }
    if (method == "GET" && (path == "/metrics" || path == "/api/metrics")) {
        const std::string board = boardLoadJson();
        std::vector<std::string> child_bodies;
        std::vector<PrometheusTaskSample> samples;
        {
            std::lock_guard<std::mutex> lock(g_tasks_mutex);
            child_bodies = fetchChildStatusBodies();
            samples.reserve(g_tasks.size());
            for (const auto& t : g_tasks) {
                PrometheusTaskSample ps;
                ps.id = t->id;
                ps.task = t->spec.task;
                ps.state = t->pid > 0 ? "running" : "stopped";
                ps.uptime_s = t->pid > 0
                                  ? static_cast<int64_t>(std::time(nullptr)) - t->start_time_s
                                  : 0;
                ps.child_ok = false;
                samples.push_back(ps);
            }
        }
        // 解析子进程 status.json（在锁外，解析失败仅影响该路 fps/计数指标）
        for (size_t i = 0; i < samples.size() && i < child_bodies.size(); ++i) {
            if (child_bodies[i].empty()) {
                continue;
            }
            bool ok = false;
            const mini_json::JsonValue st = mini_json::JsonParser(child_bodies[i]).parse(ok);
            if (!ok || st.type != mini_json::JsonValue::Type::Object) {
                continue;
            }
            const mini_json::JsonValue* v = st.get("publish_fps");
            if (v && v->type == mini_json::JsonValue::Type::Number) {
                samples[i].publish_fps = v->num;
                samples[i].child_ok = true;
            }
            if ((v = st.get("published_frames")) && v->type == mini_json::JsonValue::Type::Number) {
                samples[i].published_frames = static_cast<long>(v->num);
            }
            if ((v = st.get("dropped_frames")) && v->type == mini_json::JsonValue::Type::Number) {
                samples[i].dropped_frames = static_cast<long>(v->num);
            }
            if ((v = st.get("input_reconnect_count")) && v->type == mini_json::JsonValue::Type::Number) {
                samples[i].input_reconnects = static_cast<long>(v->num);
            }
        }
        const std::string body = formatPrometheusMetrics(board, samples);
        sendResponse(client_fd, "200 OK", "text/plain; version=0.0.4; charset=utf-8", body);
        return;
    }
    if (method == "GET" && path == "/api/summary") {
        const std::string board = boardLoadJson();
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        std::ostringstream j;
        j << "{\"git\":\"" << buildGitCommit() << "\",\"board\":" << board << ",\"tasks\":[";
        // 并行代理各子进程 /status.json（条带分片到 ≤8 线程；单子进程假死只损失
        // 它自己的 child 字段，httpGetBody 1.5s 超时兜底，不再拖死整个面板）
        std::vector<std::string> child_bodies = fetchChildStatusBodies();
        for (size_t i = 0; i < g_tasks.size(); ++i) {
            if (i) {
                j << ",";
            }
            const Task& t = *g_tasks[i];
            const bool running = t.pid > 0;
            const int64_t uptime = running ? static_cast<int64_t>(std::time(nullptr)) - t.start_time_s : 0;
            j << "{\"id\":" << t.id
              << ",\"task\":\"" << jsonEscape(t.spec.task)
              << "\",\"input\":\"" << jsonEscape(t.spec.input)
              << "\",\"model\":\"" << jsonEscape(t.spec.model)
              << "\",\"codec\":\"" << jsonEscape(t.spec.codec)
              << "\",\"port\":" << t.port
              << ",\"pid\":" << (running ? t.pid : -1)
              << ",\"status\":\"" << (running ? "running" : "stopped")
              << "\",\"uptime_s\":" << uptime
              << ",\"restarts\":" << t.restarts
              << ",\"webrtc\":\"" << jsonEscape(deriveWebrtcUrl(t.spec.output, t.spec.codec)) << "\"";
            if (running) {
                const std::string& child = child_bodies[i];
                if (!child.empty()) {
                    j << ",\"child\":" << child;
                }
            }
            j << "}";
        }
        j << "]}";
        sendJson(client_fd, j.str());
        return;
    }

    if (method == "GET" && path == "/api/tasks") {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        std::ostringstream j;
        j << "[";
        for (size_t i = 0; i < g_tasks.size(); ++i) {
            if (i) j << ",";
            j << taskToJson(*g_tasks[i]);
        }
        j << "]";
        sendJson(client_fd, j.str());
        return;
    }

    // /api/tasks/<id>...
    const std::string prefix = "/api/tasks/";
    if (path.rfind(prefix, 0) == 0) {
        const std::string rest = path.substr(prefix.size());
        const size_t slash = rest.find('/');
        const std::string id_str = slash == std::string::npos ? rest : rest.substr(0, slash);
        const std::string sub = slash == std::string::npos ? "" : rest.substr(slash + 1);
        const int id = std::atoi(id_str.c_str());

        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        Task* task = nullptr;
        for (const auto& t : g_tasks) {
            if (t->id == id) {
                task = t.get();
                break;
            }
        }
        if (!task) {
            sendJson(client_fd, "{\"error\":\"task not found\"}", "404 Not Found");
            return;
        }

        if (method == "GET" && sub.empty()) {
            sendJson(client_fd, taskToJson(*task));
            return;
        }
        if (method == "DELETE" && sub.empty()) {
            const std::string before = taskToJson(*task);
            stopTask(task);
            task->wanted.store(false);
            if (task->dynamic) {
                // 动态任务删除 = 从持久化文件移除（daemon 重启后不再恢复）
                task->dynamic = false;
                saveDynamicTasks();
            }
            std::ostringstream j;
            j << "{\"stopped\":" << before << "}";
            sendJson(client_fd, j.str());
            return;
        }
        if (method == "POST" && sub == "restart") {
            stopTask(task);
            task->wanted.store(true);
            task->restarts = 0;
            task->give_up = false;
            task->pending_restart = false;
            task->restart_window.clear();
            const bool ok = spawnTask(task);
            std::ostringstream j;
            j << "{\"ok\":" << (ok ? "true" : "false") << ",\"task\":" << taskToJson(*task) << "}";
            sendJson(client_fd, j.str(), ok ? "200 OK" : "500 Internal Server Error");
            return;
        }
        // 修改任务参数（task/input/model/threads/port/npu_core_mask/...），改完自动重启生效
        if (method == "POST" && sub == "update") {
            const auto kv = parseQuery(query);
            const auto get = [&](const std::string& key, const std::string& def = "") {
                const auto it = kv.find(key);
                return it == kv.end() ? def : it->second;
            };
            TaskSpec& spec = task->spec;
            const std::string new_task = get("task", spec.task);
            if (new_task != spec.task && kv.count("model") == 0) {
                spec.model.clear();  // 任务类型变化且未显式指定模型 → spawnTask 按新类型选默认模型
            }
            spec.task = new_task;
            spec.input = get("input", spec.input);
            spec.model = get("model", spec.model);
            spec.threads = get("threads", spec.threads);
            if (!get("port").empty()) {
                spec.port = get("port");
                task->port = std::atoi(spec.port.c_str());
            }
            if (kv.count("npu_core_mask")) spec.npu_core_mask = get("npu_core_mask");
            if (kv.count("npu_core_start")) spec.npu_core_start = get("npu_core_start");
            if (kv.count("pace")) spec.pace = get("pace");
            if (kv.count("web_preview_scale")) spec.web_preview_scale = get("web_preview_scale");
            if (kv.count("output")) spec.output = get("output");
            if (kv.count("codec")) spec.codec = get("codec");
            if (kv.count("tracking")) spec.tracking = get("tracking");
            if (kv.count("aux_model")) spec.aux_model = get("aux_model");
            if (kv.count("aux_task")) spec.aux_task = get("aux_task");
            // 事件/告警透传（仅 query 里出现的键生效；空值=删除该透传键）
            for (const auto& pkv : kv) {
                if (pkv.first == "extra_yaml") {
                    spec.extra_yaml = pkv.second;
                    continue;
                }
                if (isPassthroughKey(pkv.first)) {
                    setPassthrough(spec, pkv.first, pkv.second);
                }
            }
            std::string url_codec;
            spec.output = extractCodecFromOutput(spec.output, url_codec);
            if (spec.codec.empty() && !url_codec.empty()) {
                spec.codec = url_codec;
            }
            std::fprintf(stderr, "[daemon] update task %d: task=%s input=%s model=%s threads=%s pace=%s codec=%s\n",
                         task->id, spec.task.c_str(), spec.input.c_str(), spec.model.c_str(),
                         spec.threads.c_str(), spec.pace.c_str(), spec.codec.c_str());
            if (spec.input.empty()) {
                sendJson(client_fd, "{\"error\":\"input required\"}", "400 Bad Request");
                return;
            }
            stopTask(task);
            task->wanted.store(true);
            task->restarts = 0;
            task->give_up = false;
            task->pending_restart = false;
            task->restart_window.clear();
            const bool ok = spawnTask(task);
            if (ok && task->dynamic) {
                saveDynamicTasks();  // 动态任务的修改持久化（含透传字段）
            }
            std::ostringstream j;
            j << "{\"ok\":" << (ok ? "true" : "false") << ",\"task\":" << taskToJson(*task) << "}";
            sendJson(client_fd, j.str(), ok ? "200 OK" : "500 Internal Server Error");
            return;
        }
        // 命名事件规则查询/替换（可视化编辑器用）：
        //   GET  /api/tasks/<id>/rules         → {"rules":[{id,type,line/region,...},...]}
        //   POST /api/tasks/<id>/rules         body {"rules":[...]} 全量替换并重启任务
        // 注意：streams.json 托管的任务，REST 改的 rules 在下次配置热加载时会被文件覆盖
        if (method == "GET" && sub == "rules") {
            std::ostringstream j;
            j << "{\"rules\":[";
            for (size_t i = 0; i < task->spec.rules.size(); ++i) {
                if (i) {
                    j << ",";
                }
                j << "{";
                for (size_t k = 0; k < task->spec.rules[i].size(); ++k) {
                    if (k) {
                        j << ",";
                    }
                    j << "\"" << jsonEscape(task->spec.rules[i][k].first) << "\":\""
                      << jsonEscape(task->spec.rules[i][k].second) << "\"";
                }
                j << "}";
            }
            j << "]}";
            sendJson(client_fd, j.str());
            return;
        }
        if (method == "POST" && sub == "rules") {
            bool ok = false;
            mini_json::JsonValue jv = mini_json::JsonParser(body).parse(ok);
            const mini_json::JsonValue* arr = ok ? jv.get("rules") : nullptr;
            if (!arr || arr->type != JsonValue::Type::Array) {
                sendJson(client_fd,
                         "{\"error\":\"body must be JSON object with rules array\"}",
                         "400 Bad Request");
                return;
            }
            std::vector<std::vector<std::pair<std::string, std::string>>> new_rules;
            for (const JsonValue& r : arr->arr) {
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
                    new_rules.push_back(std::move(rule_kv));
                }
            }
            task->spec.rules = std::move(new_rules);
            stopTask(task);
            task->wanted.store(true);
            task->give_up = false;
            task->pending_restart = false;
            task->restart_window.clear();
            const bool spawned = spawnTask(task);
            if (task->dynamic) {
                saveDynamicTasks();
            }
            std::ostringstream j;
            j << "{\"ok\":" << (spawned ? "true" : "false") << ",\"rules\":"
              << task->spec.rules.size() << "}";
            sendJson(client_fd, j.str(), spawned ? "200 OK" : "500 Internal Server Error");
            return;
        }
        if (method == "GET" && sub == "status.json") {
            const std::string body = httpGetBody("127.0.0.1", task->port, "/status.json");
            if (body.empty()) {
                sendJson(client_fd, "{\"error\":\"child not responding\"}", "503 Service Unavailable");
            } else {
                sendResponse(client_fd, "200 OK", "application/json; charset=utf-8", body);
            }
            return;
        }
        if (method == "GET" && sub == "log") {
            const auto kv = parseQuery(query);
            int tail_n = 200;
            const auto it = kv.find("tail");
            if (it != kv.end() && std::atoi(it->second.c_str()) > 0) {
                tail_n = std::atoi(it->second.c_str());
            }
            const std::string body = tailFile(task->log_path, tail_n);
            if (body.empty()) {
                sendResponse(client_fd, "404 Not Found", "text/plain; charset=utf-8", "log not available\n");
            } else {
                sendResponse(client_fd, "200 OK", "text/plain; charset=utf-8", body);
            }
            return;
        }
        sendJson(client_fd, "{\"error\":\"unknown sub-route\"}", "404 Not Found");
        return;
    }

    if (method == "POST" && path == "/api/tasks") {
        const auto kv = parseQuery(query);
        if (kv.find("input") == kv.end() || kv.at("input").empty()) {
            sendJson(client_fd, "{\"error\":\"input required\"}", "400 Bad Request");
            return;
        }
        TaskSpec spec;
        const auto get = [&](const std::string& key, const std::string& def = "") {
            const auto it = kv.find(key);
            return it == kv.end() ? def : it->second;
        };
        spec.id = get("id");
        spec.task = get("task", "detect");
        spec.input = get("input");
        spec.model = get("model");
        spec.threads = get("threads", "3");
        spec.port = get("port");
        spec.pace = get("pace", "0");
        spec.npu_core_mask = get("npu_core_mask");
        spec.npu_core_start = get("npu_core_start");
        spec.web_preview_scale = get("web_preview_scale");
        spec.output = get("output");
        std::string url_codec;
        spec.output = extractCodecFromOutput(spec.output, url_codec);
        spec.codec = get("codec");
        if (spec.codec.empty() && !url_codec.empty()) {
            spec.codec = url_codec;
        }
        spec.tracking = get("tracking");
        spec.aux_model = get("aux_model");
        spec.aux_task = get("aux_task", "detect");
        // 事件/告警透传（query 参数键名与子进程 yaml 一致；空值=清除该键）
        for (const auto& pkv : kv) {
            if (pkv.first == "extra_yaml") {
                spec.extra_yaml = pkv.second;
                continue;
            }
            if (isPassthroughKey(pkv.first)) {
                setPassthrough(spec, pkv.first, pkv.second);
            }
        }

        auto task = std::make_unique<Task>();
        task->id = nextTaskId(spec.id);
        task->spec = spec;
        // 动态任务固化数字 id 作为身份（重启恢复/热切换匹配依赖），并持久化
        task->dynamic = true;
        task->spec.id = std::to_string(task->id);
        task->port = spec.port.empty() ? 0 : std::atoi(spec.port.c_str());
        task->yaml_path = g_opt.dump_dir + "/daemon_" + std::to_string(task->id) + ".yaml";
        task->log_path = g_opt.dump_dir + "/daemon_" + std::to_string(task->id) + ".log";
        const bool ok = spawnTask(task.get());
        std::ostringstream j;
        if (ok) {
            j << "{\"ok\":true,\"task\":" << taskToJson(*task) << "}";
        } else {
            j << "{\"ok\":false,\"error\":\"spawn failed\"}";
        }
        g_tasks.push_back(std::move(task));
        saveDynamicTasks();
        sendJson(client_fd, j.str(), ok ? "200 OK" : "500 Internal Server Error");
        return;
    }

    // -----------------------------------------------------------------------
    // 事件闭环（T4.4 + LLM 复检）：子进程 alert_webhook_url 指向本入口即接入
    //   POST /internal/event    事件上报 → 落盘/内存 → （可选）复检入队 或 转发业务 webhook
    //   GET  /api/events        事件查询（task/type/since/limit 过滤，含复检结论）
    //   GET  /api/events/<id>   单事件详情
    //   POST /api/events/<id>/review  人工复核标注 {"verdict":"true|false|uncertain","reason":".."}
    // -----------------------------------------------------------------------
    if (method == "POST" && path == "/internal/event") {
        if (body.empty() || body.front() != '{') {
            sendJson(client_fd, "{\"error\":\"json object body required\"}", "400 Bad Request");
            return;
        }
        const long id = g_events ? g_events->append(body) : -1;
        if (id < 0) {
            sendJson(client_fd, "{\"error\":\"invalid event json\"}", "400 Bad Request");
            return;
        }
        bool queued = false;
        bool forwarded = false;
        if (g_reviewer && g_reviewer->submit(id, body)) {
            queued = true;  // 复检完成后由复检器转发（body 增补 llm 结论）
        } else if (g_forward_enabled) {
            // 复检未启用或队满：fail-open 原样转发，不阻断告警链路
            forwarded = !g_forward_url_full_.empty()
                            ? httpPostUrl(g_forward_url_full_, body, "application/json", 5000)
                            : httpPostJson(g_forward_host, g_forward_port, g_forward_path, body, 5000);
        }
        std::ostringstream j;
        j << "{\"ok\":true,\"id\":" << id << ",\"queued_review\":" << (queued ? "true" : "false")
          << ",\"forwarded\":" << (forwarded ? "true" : "false") << "}";
        sendJson(client_fd, j.str());
        return;
    }

    if (method == "GET" && path == "/api/events") {
        const auto kv = parseQuery(query);
        const auto get = [&](const std::string& key, const std::string& def = "") {
            const auto it = kv.find(key);
            return it == kv.end() ? def : it->second;
        };
        const std::string task = get("task");
        const std::string type = get("type");  // line_cross/intrusion/dwell/absence/near_distance/count...
        const long since = std::atol(get("since").c_str());
        int limit = std::atoi(get("limit", "100").c_str());
        if (limit <= 0 || limit > 1000) {
            limit = 100;
        }
        std::vector<EventStore::Item> items;
        if (g_events) {
            items = g_events->query(task, type, since, limit);
        }
        std::ostringstream j;
        j << "{\"count\":" << items.size() << ",\"events\":[";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) {
                j << ",";
            }
            j << "{\"id\":" << items[i].id << ",\"recv_ts\":" << items[i].recv_ts_ms
              << ",\"body\":" << mini_json::dump(items[i].body);
            if (items[i].has_review) {
                j << ",\"review\":{\"verdict\":\"" << items[i].review.verdict << "\",\"source\":\""
                  << items[i].review.source << "\",\"confidence\":" << items[i].review.confidence
                  << ",\"reason\":\"";
                mini_json::appendEscaped(j, items[i].review.reason);
                j << "\"}";
            }
            j << "}";
        }
        j << "]}";
        sendJson(client_fd, j.str());
        return;
    }

    const std::string ev_prefix = "/api/events/";
    if (path.rfind(ev_prefix, 0) == 0) {
        const std::string rest = path.substr(ev_prefix.size());
        const size_t slash = rest.find('/');
        const std::string id_str = slash == std::string::npos ? rest : rest.substr(0, slash);
        const std::string sub = slash == std::string::npos ? "" : rest.substr(slash + 1);
        const long ev_id = std::atol(id_str.c_str());

        if (method == "GET" && sub.empty()) {
            EventStore::Item item;
            if (!g_events || !g_events->get(ev_id, &item)) {
                sendJson(client_fd, "{\"error\":\"event not found\"}", "404 Not Found");
                return;
            }
            std::ostringstream j;
            j << "{\"id\":" << item.id << ",\"recv_ts\":" << item.recv_ts_ms
              << ",\"body\":" << mini_json::dump(item.body);
            if (item.has_review) {
                j << ",\"review\":{\"verdict\":\"" << item.review.verdict << "\",\"source\":\""
                  << item.review.source << "\",\"confidence\":" << item.review.confidence
                  << ",\"reason\":\"";
                mini_json::appendEscaped(j, item.review.reason);
                j << "\"}";
            }
            j << "}";
            sendJson(client_fd, j.str());
            return;
        }
        if (method == "POST" && sub == "review") {
            bool ok = false;
            mini_json::JsonValue jv = mini_json::JsonParser(body).parse(ok);
            const std::string verdict = ok ? jv.getString("verdict") : std::string();
            const std::string reason = ok ? jv.getString("reason") : std::string();
            if (verdict != "true" && verdict != "false" && verdict != "uncertain") {
                sendJson(client_fd, "{\"error\":\"verdict must be true|false|uncertain\"}",
                         "400 Bad Request");
                return;
            }
            EventStore::Review review;
            review.verdict = verdict;
            review.reason = reason;
            review.source = "manual";
            review.confidence = 1.0;
            const bool attached = g_events && g_events->attachReview(ev_id, review);
            if (attached && g_flywheel) {
                EventStore::Item item;
                if (g_events->get(ev_id, &item)) {
                    g_flywheel->onReview(ev_id, mini_json::dump(item.body), review);
                }
            }
            sendJson(client_fd, attached ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"event not found\"}",
                     attached ? "200 OK" : "404 Not Found");
            return;
        }
        sendJson(client_fd, "{\"error\":\"unknown sub-route\"}", "404 Not Found");
        return;
    }

    if (method == "POST" && path == "/api/reload") {
        const auto kv = parseQuery(query);
        const auto it = kv.find("file");
        const std::string file = (it != kv.end() && !it->second.empty()) ? it->second : g_opt.streams_file;
        const int configured = reloadAndApply(file);
        std::ostringstream j;
        j << "{\"ok\":true,\"configured_tasks\":" << configured << ",\"file\":\"" << jsonEscape(file) << "\"}";
        sendJson(client_fd, j.str());
        return;
    }

    sendJson(client_fd, "{\"error\":\"not found\"}", "404 Not Found");
}

void handleClient(int client_fd) {
    // 读完整请求：请求头（定位 \r\n\r\n）+ 按 Content-Length 读 body。
    // /internal/event 与人工复核都依赖 body；SO_RCVTIMEO 防慢速攻击挂住线程
    timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    const size_t kHeaderCap = 32 * 1024;
    const size_t kBodyCap = 4 * 1024 * 1024;
    std::string raw;
    size_t header_end = std::string::npos;
    char buf[4096];
    while (raw.size() < kHeaderCap) {
        const ssize_t n = ::recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        raw.append(buf, static_cast<size_t>(n));
        header_end = raw.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            break;
        }
    }
    if (header_end == std::string::npos) {
        ::close(client_fd);
        return;
    }

    // 请求行
    const size_t line_end = raw.find('\n');
    if (line_end == std::string::npos || line_end > header_end) {
        ::close(client_fd);
        return;
    }
    std::string first_line = raw.substr(0, line_end);
    if (!first_line.empty() && first_line.back() == '\r') {
        first_line.pop_back();
    }
    std::istringstream iss(first_line);
    std::string method, path;
    iss >> method >> path;
    if (method.empty() || path.empty()) {
        ::close(client_fd);
        return;
    }

    // Content-Length（大小写不敏感）+ 鉴权 token（X-API-Token / Authorization: Bearer）
    size_t content_length = 0;
    std::string header_token;
    {
        std::string lower_head = raw.substr(0, header_end);
        std::transform(lower_head.begin(), lower_head.end(), lower_head.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string marker = "content-length:";
        const size_t p = lower_head.find(marker);
        if (p != std::string::npos) {
            content_length = static_cast<size_t>(std::atol(raw.c_str() + p + marker.size()));
        }
        const auto extractHeader = [&](const std::string& key, size_t key_len) {
            const size_t hp = lower_head.find(key);
            if (hp == std::string::npos) {
                return std::string();
            }
            std::string v = raw.substr(hp + key_len);
            const size_t eol = v.find("\r\n");
            if (eol != std::string::npos) {
                v.resize(eol);
            }
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
                v.erase(0, 1);
            }
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) {
                v.pop_back();
            }
            return v;
        };
        header_token = extractHeader("x-api-token:", 12);
        if (header_token.empty()) {
            header_token = extractHeader("authorization: bearer", 21);
        }
    }
    if (content_length > kBodyCap) {
        sendResponse(client_fd, "413 Payload Too Large", "application/json", "{\"error\":\"body too large\"}");
        ::close(client_fd);
        return;
    }
    std::string body = raw.substr(header_end + 4);
    while (body.size() < content_length) {
        const ssize_t n = ::recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        body.append(buf, static_cast<size_t>(n));
    }
    if (body.size() > content_length) {
        body.resize(content_length);
    }

    handleRequest(client_fd, method, path, body, header_token);
    ::close(client_fd);
}

void acceptLoop() {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::perror("[daemon] socket");
        return;
    }
    int opt = 1;
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(g_opt.api_port));
    if (g_opt.bind.empty() || g_opt.bind == "0.0.0.0") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else {
        ::inet_pton(AF_INET, g_opt.bind.c_str(), &addr.sin_addr);
    }
    if (::bind(server_fd, (const sockaddr*)&addr, sizeof(addr)) != 0 || ::listen(server_fd, 16) != 0) {
        std::perror("[daemon] bind/listen");
        ::close(server_fd);
        g_shutdown.store(true);  // 端口被占等致命错误：让监督循环退出清理
        return;
    }
    std::printf("[daemon] REST API listening on %s:%d\n", g_opt.bind.c_str(), g_opt.api_port);
    // 固定 worker 池 + 有界连接队列：替代"一连接一线程"，防面板轮询叠加多客户端时线程膨胀
    {
        std::deque<int> conns;
        std::mutex conns_mutex;
        std::condition_variable conns_cv;
        constexpr int kWorkers = 12;
        constexpr size_t kQueueCap = 64;
        std::vector<std::thread> workers;
        for (int i = 0; i < kWorkers; ++i) {
            workers.emplace_back([&]() {
                for (;;) {
                    int fd = -1;
                    {
                        std::unique_lock<std::mutex> lock(conns_mutex);
                        conns_cv.wait(lock, [&] { return g_shutdown.load() || !conns.empty(); });
                        if (g_shutdown.load() && conns.empty()) {
                            return;
                        }
                        fd = conns.front();
                        conns.pop_front();
                    }
                    handleClient(fd);  // fd 由 handleClient 负责关闭
                }
            });
        }
        while (!g_shutdown.load()) {
            sockaddr_in client{};
            socklen_t len = sizeof(client);
            const int client_fd = ::accept(server_fd, (sockaddr*)&client, &len);
            if (client_fd < 0) {
                if (g_shutdown.load()) {
                    break;
                }
                continue;
            }
            bool pushed = false;
            {
                std::lock_guard<std::mutex> lock(conns_mutex);
                if (conns.size() < kQueueCap) {
                    conns.push_back(client_fd);
                    pushed = true;
                }
            }
            if (pushed) {
                conns_cv.notify_one();
            } else {
                ::close(client_fd);  // 过载保护：队列满直接丢弃
            }
        }
        ::close(server_fd);
        {
            std::lock_guard<std::mutex> lock(conns_mutex);
        }
        conns_cv.notify_all();
        for (auto& w : workers) {
            w.join();
        }
        std::lock_guard<std::mutex> lock(conns_mutex);
        for (const int fd : conns) {
            ::close(fd);  // 未处理完的连接随退出丢弃
        }
    }
}

// ---------------------------------------------------------------------------
// 信号与监督
// ---------------------------------------------------------------------------
