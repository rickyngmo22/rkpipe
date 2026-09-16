#include "io/http_request.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

std::string urlDecode(const std::string& s) {
    auto hexVal = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const int hi = hexVal(s[i + 1]);
            const int lo = hexVal(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        } else if (s[i] == '+') {
            out.push_back(' ');
            continue;
        }
        out.push_back(s[i]);
    }
    return out;
}

std::map<std::string, std::string> parseQuery(const std::string& query) {
    std::map<std::string, std::string> kv;
    size_t start = 0;
    while (start <= query.size()) {
        const size_t amp = query.find('&', start);
        const std::string pair = query.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
        const size_t eq = pair.find('=');
        if (eq != std::string::npos) {
            kv[urlDecode(pair.substr(0, eq))] = urlDecode(pair.substr(eq + 1));
        }
        if (amp == std::string::npos) {
            break;
        }
        start = amp + 1;
    }
    return kv;
}

std::string httpRequestMethod(const std::string& request) {
    const std::size_t space = request.find(' ');
    return space == std::string::npos ? "GET" : request.substr(0, space);
}

std::string httpRequestPath(const std::string& request) {
    const std::size_t line_end = request.find("\r\n");
    const std::string line = request.substr(0, line_end);
    const std::size_t method_end = line.find(' ');
    if (method_end == std::string::npos) {
        return "/";
    }
    const std::size_t path_end = line.find(' ', method_end + 1);
    if (path_end == std::string::npos) {
        return "/";
    }
    std::string path = line.substr(method_end + 1, path_end - method_end - 1);
    // 剥离 query string（允许 /stream.mjpg?t=123 这种强制刷新 URL）
    const std::size_t qmark = path.find('?');
    if (qmark != std::string::npos) {
        path = path.substr(0, qmark);
    }
    return path;
}

std::string httpRequestBody(const std::string& request) {
    const std::size_t header_end = request.find("\r\n\r\n");
    return header_end == std::string::npos ? "" : request.substr(header_end + 4);
}

long httpContentLength(const std::string& request) {
    const std::size_t header_end = request.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return -1;
    }
    std::string lower_headers = request.substr(0, header_end);
    for (char& ch : lower_headers) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const std::size_t key = lower_headers.find("content-length:");
    if (key == std::string::npos) {
        return -1;
    }
    const std::size_t value_begin = request.find_first_not_of(" \t", key + std::strlen("content-length:"));
    if (value_begin == std::string::npos) {
        return -1;
    }
    const std::size_t value_end = request.find_first_not_of("0123456789", value_begin);
    return std::strtol(request.substr(value_begin, value_end - value_begin).c_str(), nullptr, 10);
}
