#include "daemon/events_store.h"

#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>

namespace {

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

EventStore::EventStore(const std::string& file_path, size_t mem_capacity, size_t file_max_bytes)
    : file_path_(file_path), mem_capacity_(mem_capacity ? mem_capacity : 2048),
      file_max_bytes_(file_max_bytes ? file_max_bytes : 32ull << 20) {
    std::lock_guard<std::mutex> lock(mutex_);
    loadTailLocked();
}

void EventStore::loadTailLocked() {
    if (file_path_.empty()) {
        return;
    }
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        return;
    }
    // 只回放末尾 mem_capacity_ 行（用临时 deque 存行，避免整文件驻留）
    std::deque<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            lines.push_back(line);
            if (lines.size() > mem_capacity_) {
                lines.pop_front();
            }
        }
    }
    for (const auto& l : lines) {
        bool ok = false;
        mini_json::JsonValue v = mini_json::JsonParser(l).parse(ok);
        if (!ok || v.type != mini_json::JsonValue::Type::Object) {
            continue;
        }
        const mini_json::JsonValue* kind = v.get("kind");
        if (kind && kind->type == mini_json::JsonValue::Type::String && kind->str == "review") {
            const long event_id = static_cast<long>(v.getNumber("event_id", 0));
            for (auto it = items_.rbegin(); it != items_.rend(); ++it) {
                if (it->id == event_id) {
                    it->has_review = true;
                    it->review.verdict = v.getString("verdict");
                    it->review.reason = v.getString("reason");
                    it->review.confidence = v.getNumber("confidence", 0.0);
                    it->review.source = v.getString("source");
                    it->review.ts_ms = static_cast<int64_t>(v.getNumber("ts", 0.0));
                    break;
                }
            }
            continue;
        }
        const mini_json::JsonValue* idv = v.get("id");
        const mini_json::JsonValue* body = v.get("body");
        if (!idv || idv->type != mini_json::JsonValue::Type::Number || !body ||
            body->type != mini_json::JsonValue::Type::Object) {
            continue;
        }
        Item item;
        item.id = static_cast<long>(idv->num);
        item.recv_ts_ms = static_cast<int64_t>(v.getNumber("recv_ts", 0.0));
        item.body = *body;
        items_.push_back(std::move(item));
        if (items_.size() > mem_capacity_) {
            items_.pop_front();
        }
        if (item.id >= next_id_) {
            next_id_ = item.id + 1;
        }
    }
}

void EventStore::appendLineLocked(const std::string& line) {
    if (file_path_.empty()) {
        return;
    }
    struct stat st {};
    if (::stat(file_path_.c_str(), &st) == 0 &&
        static_cast<size_t>(st.st_size) + line.size() + 1 > file_max_bytes_) {
        // 超限轮转：events.jsonl → events.jsonl.1（覆盖旧代）
        const std::string rotated = file_path_ + ".1";
        ::rename(file_path_.c_str(), rotated.c_str());
    }
    std::ofstream out(file_path_, std::ios::binary | std::ios::app);
    if (out.is_open()) {
        out << line << "\n";
    }
}

long EventStore::append(const std::string& body) {
    // body 须为 JSON 对象：复检/查询/转发都依赖结构化解析
    bool ok = false;
    mini_json::JsonValue parsed = mini_json::JsonParser(body).parse(ok);
    if (!ok || parsed.type != mini_json::JsonValue::Type::Object) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Item item;
    item.id = next_id_++;
    item.recv_ts_ms = nowMs();
    item.body = std::move(parsed);

    std::ostringstream line;
    line << "{\"id\":" << item.id << ",\"recv_ts\":" << item.recv_ts_ms << ",\"body\":" << body << "}";
    appendLineLocked(line.str());

    items_.push_back(std::move(item));
    if (items_.size() > mem_capacity_) {
        items_.pop_front();
    }
    ++version_;
    version_cv_.notify_all();
    return item.id;
}

bool EventStore::attachReview(long event_id, const Review& review) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool found = false;
    for (auto it = items_.rbegin(); it != items_.rend(); ++it) {
        if (it->id == event_id) {
            it->has_review = true;
            it->review = review;
            found = true;
            break;
        }
    }
    std::ostringstream line;
    line << "{\"kind\":\"review\",\"event_id\":" << event_id << ",\"source\":\"" << review.source
         << "\",\"verdict\":\"" << review.verdict << "\",\"confidence\":" << review.confidence
         << ",\"ts\":" << (review.ts_ms ? review.ts_ms : nowMs()) << ",\"reason\":\"";
    for (const char c : review.reason) {
        if (c == '"' || c == '\\') {
            line << '\\';
        }
        if (static_cast<unsigned char>(c) >= 0x20) {
            line << c;
        }
    }
    line << "\"}";
    appendLineLocked(line.str());
    last_review_event_id_ = event_id;
    ++version_;
    version_cv_.notify_all();
    return found;
}

bool EventStore::matchType(const mini_json::JsonValue& body, const std::string& type) {
    if (type.empty()) {
        return true;
    }
    // 事件类型：body.event（rule_events/near_distance/count）；rule_events 细分看 body.rule_type
    const mini_json::JsonValue* ev = body.get("event");
    if (ev && ev->type == mini_json::JsonValue::Type::String) {
        if (ev->str == type) {
            return true;
        }
        if (ev->str == "rule_events") {
            const mini_json::JsonValue* rt = body.get("rule_type");
            return rt && rt->type == mini_json::JsonValue::Type::String && rt->str == type;
        }
    }
    return false;
}

std::vector<EventStore::Item> EventStore::query(const std::string& task, const std::string& type,
                                                long since, int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Item> matched;
    for (const auto& it : items_) {
        if (it.id <= since) {
            continue;
        }
        if (!task.empty()) {
            const mini_json::JsonValue* t = it.body.get("task");
            if (!t || t->type != mini_json::JsonValue::Type::String || t->str != task) {
                continue;
            }
        }
        if (!matchType(it.body, type)) {
            continue;
        }
        matched.push_back(it);
    }
    if (limit > 0 && matched.size() > static_cast<size_t>(limit)) {
        // 保留最近的 limit 条，按 id 升序返回
        matched.erase(matched.begin(), matched.end() - limit);
    }
    return matched;
}

bool EventStore::get(long id, Item* out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = items_.rbegin(); it != items_.rend(); ++it) {
        if (it->id == id) {
            if (out) {
                *out = *it;
            }
            return true;
        }
    }
    return false;
}

size_t EventStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

long EventStore::lastReviewEventId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_review_event_id_;
}

long EventStore::version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return version_;
}

long EventStore::waitVersion(long known, int timeout_ms) const {
    std::unique_lock<std::mutex> lock(mutex_);
    version_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [&] { return version_ != known; });
    return version_;
}
