#include "daemon/daemon_internal.h"

#include "daemon/mini_json.h"

#include <cstdio>
#include <sstream>

// Prometheus 文本 exposition（/metrics）：
//   - 板载负载取自 boardLoadJson()（与 /api/summary 同源）
//   - 每路任务由调用方聚合（state/uptime 来自监督状态；publish_fps 等来自子进程
//     /status.json 代理，取不到时 child_ok=false，仅输出 task_info 与计数缺省）
// 纯函数、无 socket 依赖，可单测。

namespace {

// Prometheus 指标名仅允许 [a-zA-Z0-9_:]，label 值转义 \ " \n
std::string sanitizeMetricName(std::string name) {
    for (char& c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == ':';
        if (!ok) {
            c = '_';
        }
    }
    return name;
}

std::string escapeLabelValue(std::string v) {
    std::string out;
    out.reserve(v.size());
    for (const char c : v) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out;
}

}  // namespace

std::string formatPrometheusMetrics(const std::string& board_json,
                                    const std::vector<PrometheusTaskSample>& tasks) {
    bool ok = false;
    const mini_json::JsonValue board = mini_json::JsonParser(board_json).parse(ok);
    auto num = [&board, &ok](const char* key) -> double {
        const mini_json::JsonValue* v = ok ? board.get(key) : nullptr;
        return (v && v->type == mini_json::JsonValue::Type::Number) ? v->num : 0.0;
    };

    std::ostringstream o;
    o << "# HELP rkpipe_board_cpu_percent Board CPU usage percent.\n"
      << "# TYPE rkpipe_board_cpu_percent gauge\n"
      << "rkpipe_board_cpu_percent " << num("cpu_usage") << "\n";
    o << "# HELP rkpipe_board_temp_c Board temperature (celsius, max of sensors).\n"
      << "# TYPE rkpipe_board_temp_c gauge\n"
      << "rkpipe_board_temp_c " << num("temp_c") << "\n";

    if (const mini_json::JsonValue* npu = ok ? board.get("npu_load") : nullptr) {
        if (npu->type == mini_json::JsonValue::Type::Array) {
            o << "# HELP rkpipe_board_npu_load_percent NPU core load percent.\n"
              << "# TYPE rkpipe_board_npu_load_percent gauge\n";
            for (size_t i = 0; i < npu->arr.size(); ++i) {
                if (npu->arr[i].type == mini_json::JsonValue::Type::Number) {
                    o << "rkpipe_board_npu_load_percent{core=\"" << i << "\"} " << npu->arr[i].num << "\n";
                }
            }
        }
    }

    o << "# HELP rkpipe_board_mem_used_percent Board memory used percent.\n"
      << "# TYPE rkpipe_board_mem_used_percent gauge\n"
      << "rkpipe_board_mem_used_percent " << num("mem_used_pct") << "\n"
      << "# HELP rkpipe_board_mem_avail_mb Board memory available (MB).\n"
      << "# TYPE rkpipe_board_mem_avail_mb gauge\n"
      << "rkpipe_board_mem_avail_mb " << num("mem_avail_mb") << "\n"
      << "# HELP rkpipe_board_disk_free_mb Disk free (MB, metrics/snapshot dump dir).\n"
      << "# TYPE rkpipe_board_disk_free_mb gauge\n"
      << "rkpipe_board_disk_free_mb " << num("disk_free_mb") << "\n";

    o << "# HELP rkpipe_task_info Task metadata (value always 1).\n"
      << "# TYPE rkpipe_task_info gauge\n";
    for (const PrometheusTaskSample& t : tasks) {
        o << "rkpipe_task_info{task_id=\"" << t.id << "\",task=\"" << escapeLabelValue(sanitizeMetricName(t.task))
          << "\",state=\"" << escapeLabelValue(t.state) << "\"} 1\n";
    }
    o << "# HELP rkpipe_task_uptime_seconds Task uptime in seconds (running only).\n"
      << "# TYPE rkpipe_task_uptime_seconds gauge\n";
    for (const PrometheusTaskSample& t : tasks) {
        if (t.child_ok) {
            o << "rkpipe_task_uptime_seconds{task_id=\"" << t.id << "\"} " << t.uptime_s << "\n";
        }
    }
    o << "# HELP rkpipe_task_publish_fps Child web preview publish fps.\n"
      << "# TYPE rkpipe_task_publish_fps gauge\n";
    for (const PrometheusTaskSample& t : tasks) {
        if (t.child_ok) {
            o << "rkpipe_task_publish_fps{task_id=\"" << t.id << "\"} " << t.publish_fps << "\n";
        }
    }
    o << "# HELP rkpipe_task_published_frames_total Child published frames.\n"
      << "# TYPE rkpipe_task_published_frames_total counter\n";
    for (const PrometheusTaskSample& t : tasks) {
        if (t.child_ok) {
            o << "rkpipe_task_published_frames_total{task_id=\"" << t.id << "\"} " << t.published_frames << "\n";
        }
    }
    o << "# HELP rkpipe_task_dropped_frames_total Child dropped frames.\n"
      << "# TYPE rkpipe_task_dropped_frames_total counter\n";
    for (const PrometheusTaskSample& t : tasks) {
        if (t.child_ok) {
            o << "rkpipe_task_dropped_frames_total{task_id=\"" << t.id << "\"} " << t.dropped_frames << "\n";
        }
    }
    o << "# HELP rkpipe_task_input_reconnect_count Input stream reconnects.\n"
      << "# TYPE rkpipe_task_input_reconnect_count counter\n";
    for (const PrometheusTaskSample& t : tasks) {
        if (t.child_ok) {
            o << "rkpipe_task_input_reconnect_count{task_id=\"" << t.id << "\"} " << t.input_reconnects << "\n";
        }
    }
    return o.str();
}
