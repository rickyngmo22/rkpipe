#pragma once

// JSON 字符串转义（daemon / web 预览 / 告警 / 输出路由 / LLM 模块共用）。
// 语义：转义 \\ " \n \r \t；其余控制字符(<0x20)直接丢弃（裸控制符是非法 JSON），
//       中文等多字节 UTF-8 内容原样透传。

#include <string>

inline std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) >= 0x20) {
                    out.push_back(ch);
                }
                break;
        }
    }
    return out;
}
