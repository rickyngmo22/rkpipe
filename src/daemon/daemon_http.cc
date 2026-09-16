// HTTP 收发原语（自 daemon 主模块拆出）：服务端响应与轻量客户端（代理子进程状态/转发事件）。
#include <arpa/inet.h>
#include <fcntl.h>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#include "daemon/daemon_internal.h"
#include "utils/http_post.h"

bool sendAll(int fd, const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, p + sent, size - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

void sendResponse(int fd, const std::string& status, const std::string& content_type, const std::string& body) {
    std::ostringstream head;
    head << "HTTP/1.1 " << status << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Cache-Control: no-store\r\n"  // 防浏览器缓存旧页面 JS（监控页/API 均需实时）
         << "Connection: close\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
         << "Access-Control-Allow-Headers: *\r\n\r\n";
    sendAll(fd, head.str().data(), head.str().size());
    sendAll(fd, body.data(), body.size());
}

void sendJson(int fd, const std::string& body, const std::string& status) {
    sendResponse(fd, status, "application/json; charset=utf-8", body);
}

// 解析 http://host[:port]/path；仅支持 http（本机代理/内网网关场景）
bool parseHttpUrl(const std::string& url, std::string* host, int* port, std::string* path) {
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) {
        return false;
    }
    const std::string rest = url.substr(prefix.size());
    const size_t slash = rest.find('/');
    *path = slash == std::string::npos ? "/" : rest.substr(slash);
    const std::string host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
    const size_t colon = host_port.rfind(':');
    if (colon != std::string::npos) {
        *host = host_port.substr(0, colon);
        *port = std::atoi(host_port.substr(colon + 1).c_str());
    } else {
        *host = host_port;
        *port = 80;
    }
    return !host->empty() && *port > 0;
}

// 建 TCP 连接（非阻塞 connect + poll 超时，防对端假死时无限阻塞）
int connectTimeout(const std::string& host, int port, int timeout_ms) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    const int cr = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    bool connected = cr == 0;
    if (cr != 0 && errno == EINPROGRESS) {
        pollfd pfd{fd, POLLOUT, 0};
        if (::poll(&pfd, 1, timeout_ms) > 0) {
            int so_err = 0;
            socklen_t len = sizeof(so_err);
            ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len);
            connected = so_err == 0;
        }
    }
    if (!connected) {
        ::close(fd);
        return -1;
    }
    ::fcntl(fd, F_SETFL, flags);  // 恢复阻塞模式，收发靠 SO_SNDTIMEO/SO_RCVTIMEO
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return fd;
}

// 极简 HTTP GET 客户端：返回 body（失败/超时返回空）。代理子进程 /status.json 用，
// 子进程假死时 1.5s 内放弃，不拖垮 /api/summary
std::string httpGetBody(const std::string& host, int port, const std::string& path, int timeout_ms) {
    const int fd = connectTimeout(host, port, timeout_ms);
    if (fd < 0) {
        return "";
    }
    std::string req = "GET " + path + " HTTP/1.0\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
    std::string raw;
    char buf[4096];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
        raw.append(buf, static_cast<size_t>(n));
        if (raw.size() > (4ull << 20)) {
            break;
        }
    }
    ::close(fd);
    const size_t split = raw.find("\r\n\r\n");
    return split == std::string::npos ? "" : raw.substr(split + 4);
}

// JSON POST：事件转发业务 webhook 用。仅校验发送完整即视为成功（细节见 utils/http_post）
bool httpPostJson(const std::string& host, int port, const std::string& path, const std::string& body,
                  int timeout_ms) {
    return httpPost(host, port, path, body, "application/json", timeout_ms);
}