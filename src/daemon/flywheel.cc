#include "daemon/flywheel.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "daemon/mini_json.h"
#include "utils/json_escape.h"

namespace {

std::string flywheelTimestampName() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count();
    return std::to_string(ms);
}

}  // namespace

FlywheelRecord buildFlywheelRecord(long event_id,
                                   const std::string& body_json,
                                   const EventStore::Review& review) {
    FlywheelRecord rec;
    if (review.verdict != "false") {
        return rec;  // 只归档误报
    }
    bool ok = false;
    const mini_json::JsonValue body = mini_json::JsonParser(body_json).parse(ok);
    if (!ok || body.type != mini_json::JsonValue::Type::Object) {
        return rec;
    }
    const mini_json::JsonValue* snap = body.get("snapshot");
    if (!snap || snap->type != mini_json::JsonValue::Type::String || snap->str.empty()) {
        return rec;  // 无现场图，无法作为视觉训练样本
    }

    const std::string task = body.getString("task").empty() ? "detect" : body.getString("task");
    std::string type = body.getString("rule_type");
    if (type.empty()) {
        type = body.getString("event");
    }
    if (type.empty()) {
        type = "alert";
    }
    rec.ok = true;
    rec.rel_dir = task + "/" + type;
    rec.filename = std::to_string(event_id) + "_" + flywheelTimestampName() + ".jpg";
    rec.snapshot_src = snap->str;
    rec.manifest_json = "{\"id\":" + std::to_string(event_id) +
                        ",\"ts\":" + std::to_string(review.ts_ms) +
                        ",\"task\":\"" + jsonEscape(task) +
                        "\",\"type\":\"" + jsonEscape(type) +
                        "\",\"source\":\"" + jsonEscape(review.source) +
                        "\",\"detail\":\"" + jsonEscape(body.getString("detail")) +
                        "\",\"image\":\"" + jsonEscape(rec.rel_dir + "/" + rec.filename) +
                        "\",\"original_snapshot\":\"" + jsonEscape(rec.snapshot_src) + "\"}";
    return rec;
}

EventFlywheel::EventFlywheel(std::string root) : root_(std::move(root)) {
    if (!root_.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(root_, ec);
        std::fprintf(stderr, "[daemon][flywheel] enabled: %s\n", root_.c_str());
    }
}

void EventFlywheel::onReview(long event_id, const std::string& body_json,
                             const EventStore::Review& review) const {
    if (!enabled()) {
        return;
    }
    const FlywheelRecord rec = buildFlywheelRecord(event_id, body_json, review);
    if (!rec.ok) {
        return;
    }
    std::error_code ec;
    const std::filesystem::path dst_dir = std::filesystem::path(root_) / rec.rel_dir;
    std::filesystem::create_directories(dst_dir, ec);
    const std::filesystem::path dst = dst_dir / rec.filename;
    std::filesystem::copy_file(rec.snapshot_src, dst, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        std::fprintf(stderr, "[daemon][flywheel][warn] copy snapshot failed: %s (%s)\n",
                     rec.snapshot_src.c_str(), ec.message().c_str());
        return;
    }
    std::ofstream manifest(root_ + "/manifest.jsonl", std::ios::app);
    if (manifest.is_open()) {
        manifest << rec.manifest_json << "\n";
    }
}
