#pragma once

// rk_pipe_daemon 共享的极简 JSON 解析（仅覆盖配置/事件所需子集：object/array/string/number/bool）。
// 注意：\uXXXX 转义支持 UTF-8 编码与代理对组合（0xD800-0xDFFF），中文/emoji 转义可正确解码。

#include <cmath>
#include <cstdio>
#include <ostream>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace mini_json {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object } type =JsonValue::Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* get(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }
    std::string getString(const std::string& key, const std::string& def = "") const {
        const JsonValue* v = get(key);
        return (v && v->type == JsonValue::Type::String) ? v->str : def;
    }
    double getNumber(const std::string& key, double def = 0.0) const {
        const JsonValue* v = get(key);
        return (v && v->type ==JsonValue::Type::Number) ? v->num : def;
    }
};

// JSON 转义（控制字符剥离，UTF-8 字面量原样透传）
inline void escapeTo(const std::string& value, std::string* out) {
    for (const char ch : value) {
        switch (ch) {
            case '\\': *out += "\\\\"; break;
            case '"': *out += "\\\""; break;
            case '\n': *out += "\\n"; break;
            case '\r': *out += "\\r"; break;
            case '\t': *out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) >= 0x20) {
                    *out += ch;
                }
                break;
        }
    }
}

// 流式输出转义文本（ostringstream 场景；std::string 版见 escapeTo）
inline void appendEscaped(std::ostream& out, const std::string& value) {
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) >= 0x20) {
                    out << ch;
                }
                break;
        }
    }
}

// 序列化为紧凑 JSON（/api/events 回显事件 body 等场景）
inline std::string dump(const JsonValue& v) {
    std::string out;
    switch (v.type) {
        case JsonValue::Type::Null: out += "null"; break;
        case JsonValue::Type::Bool: out += v.b ? "true" : "false"; break;
        case JsonValue::Type::Number: {
            char buf[32];
            const double d = v.num;
            // 整数按整型输出（防 %g 有效位数截断大整数 ts），浮点 17 位保 round-trip
            if (d == static_cast<double>(static_cast<long long>(d)) && d < 9.0e15 && d > -9.0e15) {
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
            } else {
                std::snprintf(buf, sizeof(buf), "%.17g", d);
            }
            out += buf;
            break;
        }
        case JsonValue::Type::String:
            out += '"';
            escapeTo(v.str, &out);
            out += '"';
            break;
        case JsonValue::Type::Array: {
            out += '[';
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out += ',';
                out += dump(v.arr[i]);
            }
            out += ']';
            break;
        }
        case JsonValue::Type::Object: {
            out += '{';
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out += ',';
                out += '"';
                escapeTo(v.obj[i].first, &out);
                out += "\":";
                out += dump(v.obj[i].second);
            }
            out += '}';
            break;
        }
    }
    return out;
}


// 码点 → UTF-8（\uXXXX 解码用）
inline void appendUtf8Cp(unsigned cp, std::string* out) {
    if (cp < 0x80) {
        *out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        *out += static_cast<char>(0xC0 | (cp >> 6));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *out += static_cast<char>(0xE0 | (cp >> 12));
        *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        *out += static_cast<char>(0xF0 | (cp >> 18));
        *out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text), pos_(0) {}

    JsonValue parse(bool& ok) {
        depth_ = 0;
        skipWs();
        JsonValue v = parseValue(ok);
        skipWs();
        return v;
    }

private:
    // 递归深度上限：嵌套超限判非法（防深嵌套栈溢出 DoS——本解析器直接吃
    // 外部输入：/internal/event、streams.json、LLM 回复）。真实配置/事件嵌套 <10。
    static constexpr int kMaxDepth = 64;

    struct DepthGuard {
        int& depth;
        bool exceeded;
        explicit DepthGuard(int& d) : depth(d), exceeded(++d > kMaxDepth) {}
        ~DepthGuard() { --depth; }
    };

    const std::string& text_;
    size_t pos_;
    int depth_ = 0;

    void skipWs() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
                                       text_[pos_] == '\r')) {
            ++pos_;
        }
    }
    bool consume(char c) {
        skipWs();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    JsonValue parseValue(bool& ok) {
        DepthGuard guard{depth_};
        if (guard.exceeded) {
            ok = false;
            return {};
        }
        skipWs();
        if (pos_ >= text_.size()) {
            ok = false;
            return {};
        }
        const char c = text_[pos_];
        if (c == '{') {
            return parseObject(ok);
        }
        if (c == '[') {
            return parseArray(ok);
        }
        if (c == '"') {
            return parseString(ok);
        }
        if (c == 't' || c == 'f') {
            JsonValue v;
            v.type =JsonValue::Type::Bool;
            if (text_.compare(pos_, 4, "true") == 0) {
                v.b = true;
                pos_ += 4;
                ok = true;
            } else if (text_.compare(pos_, 5, "false") == 0) {
                v.b = false;
                pos_ += 5;
                ok = true;
            } else {
                ok = false;
            }
            return v;
        }
        if (c == 'n') {
            JsonValue v;
            if (text_.compare(pos_, 4, "null") == 0) {
                pos_ += 4;
                ok = true;
            } else {
                ok = false;
            }
            return v;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            JsonValue v;
            v.type =JsonValue::Type::Number;
            const size_t start = pos_;
            if (c == '-') {
                ++pos_;
            }
            while (pos_ < text_.size() &&
                   ((text_[pos_] >= '0' && text_[pos_] <= '9') || text_[pos_] == '.' || text_[pos_] == 'e' ||
                    text_[pos_] == 'E' || text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            v.num = std::atof(text_.substr(start, pos_ - start).c_str());
            ok = true;
            return v;
        }
        ok = false;
        return {};
    }

    JsonValue parseObject(bool& ok) {
        JsonValue v;
        v.type =JsonValue::Type::Object;
        ++pos_;  // '{'
        skipWs();
        if (consume('}')) {
            ok = true;
            return v;
        }
        for (;;) {
            skipWs();
            JsonValue key = parseString(ok);
            if (!ok) {
                return {};
            }
            if (!consume(':')) {
                ok = false;
                return {};
            }
            JsonValue val = parseValue(ok);
            if (!ok) {
                return {};
            }
            v.obj.emplace_back(key.str, std::move(val));
            if (consume('}')) {
                ok = true;
                return v;
            }
            if (!consume(',')) {
                ok = false;
                return {};
            }
        }
    }

    JsonValue parseArray(bool& ok) {
        JsonValue v;
        v.type =JsonValue::Type::Array;
        ++pos_;  // '['
        skipWs();
        if (consume(']')) {
            ok = true;
            return v;
        }
        for (;;) {
            v.arr.push_back(parseValue(ok));
            if (!ok) {
                return {};
            }
            if (consume(']')) {
                ok = true;
                return v;
            }
            if (!consume(',')) {
                ok = false;
                return {};
            }
        }
    }

    JsonValue parseString(bool& ok) {
        JsonValue v;
        v.type =JsonValue::Type::String;
        ++pos_;  // '"'
        std::string out;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                ok = true;
                v.str = out;
                return v;
            }
            if (c == '\\' && pos_ < text_.size()) {
                const char e = text_[pos_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        if (pos_ + 4 <= text_.size()) {
                            unsigned cp =
                                static_cast<unsigned>(std::strtoul(text_.substr(pos_, 4).c_str(), nullptr, 16));
                            pos_ += 4;
                            // 代理对：\uD800-\uDBFF + \uDC00-\uDFFF 组合为一个码点
                            if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= text_.size() &&
                                text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                                const unsigned lo = static_cast<unsigned>(
                                    std::strtoul(text_.substr(pos_ + 2, 4).c_str(), nullptr, 16));
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    pos_ += 6;
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                }
                            }
                            appendUtf8Cp(cp, &out);
                        }
                        break;
                    }
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        ok = false;
        return {};
    }
};

}  // namespace mini_json
