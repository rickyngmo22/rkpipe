#pragma once

// 极简 HTTP POST 客户端（alert webhook / daemon 事件转发共用）。
// 仅支持 http:// 明文（https 走 llm/ 的 libcurl provider 路线）；
// host 兼容域名与 IPv4/IPv6 地址（getaddrinfo 解析）。
// deadline 覆盖 DNS/connect/send 全程：黑洞地址在 timeout_ms 内放弃。

#include <string>

// 返回 true = 请求完整发出（对端是否 2xx 看 status_code）。
// status_code 非空时回带对端状态行码（未读到响应为 0）。
bool httpPost(const std::string& host, int port, const std::string& path,
              const std::string& body, const std::string& content_type,
              int timeout_ms, long* status_code = nullptr);

// URL 形式入口（自动分派）：
//   http://  → 内置 socket 实现
//   https:// → libcurl 实现（构建未链 libcurl 时返回 false 并打印告警）
bool httpPostUrl(const std::string& url, const std::string& body,
                 const std::string& content_type, int timeout_ms,
                 long* status_code = nullptr);
