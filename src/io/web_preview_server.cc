#include "../../include/io/web_preview_server.h"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <turbojpeg.h>

#include "core/runtime_affinity.h"

namespace {

constexpr const char* kBoundary = "frame";

// 兼容旧调试开关：RK_PIPE_REPLAY_PACE=1 时仍按原视频帧率预览
bool replayPaceEnvEnabled() {
    const char* v = std::getenv("RK_PIPE_REPLAY_PACE");
    return v && *v && std::string(v) != "0";
}

// #region debug-point D:debug-report
struct DebugEndpoint {
    std::string host = "127.0.0.1";
    int port = 7777;
    std::string path = "/event";
    std::string session = "web-preview-stall";
    bool configured = false;
};

DebugEndpoint loadDebugEndpoint() {
    DebugEndpoint endpoint;

    // 调试上报为可选能力，默认关闭（避免硬编码路径与无谓的 socket 连接）：
    //   RK_PIPE_DEBUG_SERVER_URL=http://host:port/path  开启并指定上报端点
    //   RK_PIPE_DEBUG_SESSION_ID=xxx                   覆盖会话标识
    //   旧调试文件仍可通过 RK_PIPE_DEBUG_ENV_FILE=<path> 指定（不再硬编码）。
    const char* url_env = std::getenv("RK_PIPE_DEBUG_SERVER_URL");
    const char* env_file = std::getenv("RK_PIPE_DEBUG_ENV_FILE");
    if (!url_env || url_env[0] == '\0') {
        if (!env_file || env_file[0] == '\0') {
            return endpoint;
        }
        std::ifstream env(env_file);
        std::string line;
        while (std::getline(env, line)) {
            if (line.rfind("DEBUG_SERVER_URL=", 0) == 0) {
                const std::string url = line.substr(std::strlen("DEBUG_SERVER_URL="));
                const std::string prefix = "http://";
                if (url.rfind(prefix, 0) == 0) {
                    std::string host_port = url.substr(prefix.size());
                    std::size_t slash = host_port.find('/');
                    endpoint.path = slash == std::string::npos ? "/event" : host_port.substr(slash);
                    host_port = slash == std::string::npos ? host_port : host_port.substr(0, slash);
                    std::size_t colon = host_port.rfind(':');
                    if (colon != std::string::npos) {
                        endpoint.host = host_port.substr(0, colon);
                        endpoint.port = std::atoi(host_port.substr(colon + 1).c_str());
                    }
                }
                endpoint.configured = true;
            } else if (line.rfind("DEBUG_SESSION_ID=", 0) == 0) {
                endpoint.session = line.substr(std::strlen("DEBUG_SESSION_ID="));
            }
        }
        return endpoint;
    }

    const std::string url(url_env);
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) == 0) {
        std::string host_port = url.substr(prefix.size());
        std::size_t slash = host_port.find('/');
        endpoint.path = slash == std::string::npos ? "/event" : host_port.substr(slash);
        host_port = slash == std::string::npos ? host_port : host_port.substr(0, slash);
        std::size_t colon = host_port.rfind(':');
        if (colon != std::string::npos) {
            endpoint.host = host_port.substr(0, colon);
            endpoint.port = std::atoi(host_port.substr(colon + 1).c_str());
        }
        endpoint.configured = true;
    }
    const char* session = std::getenv("RK_PIPE_DEBUG_SESSION_ID");
    if (session && session[0] != '\0') {
        endpoint.session = session;
    }
    return endpoint;
}

const DebugEndpoint& debugEndpoint() {
    static const DebugEndpoint endpoint = loadDebugEndpoint();
    return endpoint;
}

std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

void reportDebugEvent(const char* hypothesis_id,
                      const char* location,
                      const std::string& msg,
                      const std::string& data_json) {
    const DebugEndpoint& endpoint = debugEndpoint();
    if (!endpoint.configured) {
        return;
    }
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (::inet_pton(AF_INET, endpoint.host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return;
    }
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return;
    }
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    std::ostringstream body;
    body << "{\"sessionId\":\"" << jsonEscape(endpoint.session)
         << "\",\"runId\":\"pre-fix\""
         << ",\"hypothesisId\":\"" << jsonEscape(hypothesis_id)
         << "\",\"location\":\"" << jsonEscape(location)
         << "\",\"msg\":\"" << jsonEscape(msg)
         << "\",\"data\":" << data_json
         << ",\"ts\":" << now_ms << "}";
    const std::string body_str = body.str();
    std::ostringstream request;
    request << "POST " << endpoint.path << " HTTP/1.1\r\n"
            << "Host: " << endpoint.host << ":" << endpoint.port << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body_str.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << body_str;
    const std::string request_str = request.str();
    ::send(fd, request_str.data(), request_str.size(), MSG_NOSIGNAL);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
}
// #endregion

std::string makeHtmlPage(const std::string& title,
                         const std::vector<int>& extra_stream_ports,
                         const std::vector<std::string>& webrtc_urls) {
    // 窗格媒体元素：有 WebRTC URL 用 iframe（经 MediaMTX 观看推流），否则 MJPEG <img>
    auto pane_media = [&](std::size_t pane_index, const std::string& img_tag) {
        const bool has_wrtc = webrtc_urls.size() > pane_index && !webrtc_urls[pane_index].empty();
        if (!has_wrtc) {
            return img_tag;
        }
        return std::string("<iframe class=\"webrtc-stream\" src=\"") + webrtc_urls[pane_index] +
               "\" data-wrtc=\"1\" allow=\"autoplay; encrypted-media\" "
               "style=\"width:100%;aspect-ratio:16/9;border:0;background:#000;border-radius:8px;display:block;\" "
               "allowfullscreen></iframe>";
    };

    std::ostringstream html;
    html << "<!doctype html><html><head><meta charset=\"utf-8\">"
         << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
         << "<title>" << title << "</title>"
         << "<style>"
         << "body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
         << ".wrap{padding:12px;max-width:1600px;margin:0 auto;}"
         << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;}"
         << ".card{background:#1a1a1a;padding:10px;border-radius:8px;}"
         << ".label{margin:0 0 8px 0;opacity:.85;font-size:14px;}"
         << ".stat{margin:6px 0 0 0;font-size:12px;opacity:.6;font-family:monospace;}"
         << "img{width:100%;height:auto;background:#000;border-radius:8px;display:block;}"
         << "p{opacity:.8;}"
         << "</style></head><body><div class=\"wrap\">"
         << "<h2>" << title << "</h2>"
         << "<p>实时预览流 / Real-time preview</p>"
         << "<div class=\"grid\">"
         << "<div class=\"card\"><p class=\"label\">stream_1"
         << (webrtc_urls.size() > 0 && !webrtc_urls[0].empty() ? " (WebRTC)" : "")
         << "</p>"
         << pane_media(0, "<img src=\"/stream.mjpg\" alt=\"rkpipe_preview_stream_1\">")
         << "<p class=\"stat\" id=\"stm_1\">--</p></div>";
    for (std::size_t i = 0; i < extra_stream_ports.size(); ++i) {
        const int port = extra_stream_ports[i];
        if (port <= 0) {
            continue;
        }
        const bool has_wrtc = webrtc_urls.size() > (i + 1) && !webrtc_urls[i + 1].empty();
        html << "<div class=\"card\"><p class=\"label\">stream_" << (i + 2)
             << (has_wrtc ? " (WebRTC)" : "") << "</p>"
             << pane_media(i + 1,
                           "<img class=\"remote-stream\" data-port=\"" + std::to_string(port) +
                               "\" alt=\"rkpipe_preview_stream_" + std::to_string(i + 2) + "\">")
             << "<p class=\"stat\" id=\"stm_" << port << "\">--</p></div>";
    }
    html << "</div>"
         << "<script>"
         << "const host=window.location.hostname||'127.0.0.1';"
         << "document.querySelectorAll('img.remote-stream').forEach((img)=>{"
         << "const port=img.dataset.port;"
         << "img.src=`http://${host}:${port}/stream.mjpg`;"
         << "});"
         << "function fmtStat(d){"
         << "let s=(d.source_fps>0?'Video '+d.source_fps+' fps \\u00b7 ':'')"
         << "+'Stream '+d.publish_fps.toFixed(1)+' fps';"
         << "if(d.encode_ms>0)s+=' \\u00b7 enc '+d.encode_ms.toFixed(1)+'ms';"
         << "if(d.dropped_frames>0)s+=' \\u00b7 drop '+d.dropped_frames;"
         << "if(d.clients>0)s+=' \\u00b7 '+d.clients+' cli';"
         << "return s;} "
         << "function refreshStats(){"
         << "fetch('/status.json').then(r=>r.json()).then(d=>{"
         << "const el=document.getElementById('stm_1');if(el)el.textContent=fmtStat(d);"
         << "}).catch(()=>{});"
         << "document.querySelectorAll('[data-port]').forEach((el)=>{"
         << "const port=el.dataset.port;"
         << "fetch(`http://${host}:${port}/status.json`).then(r=>r.json()).then(d=>{"
         << "const id=document.getElementById('stm_'+port);if(id)id.textContent=fmtStat(d);"
         << "}).catch(()=>{});"
         << "});}"
         << "refreshStats();setInterval(refreshStats,1000);"
         << "</script>"
         << "</div></body></html>";
    return html.str();
}

std::string httpHeader(const std::string& status,
                       const std::string& content_type,
                       std::size_t content_length) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n"
        << "Connection: close\r\n"
        << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
        << "Pragma: no-cache\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << content_length << "\r\n\r\n";
    return oss.str();
}

std::string streamHeader() {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Connection: close\r\n"
        << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
        << "Pragma: no-cache\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n\r\n";
    return oss.str();
}

std::string readRequest(int fd) {
    std::string request;
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 16384) {
        const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        request.append(buffer, static_cast<std::size_t>(n));
    }
    return request;
}

std::string requestPath(const std::string& request) {
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

}  // namespace

WebPreviewServer::~WebPreviewServer() {
    stop();
}

bool WebPreviewServer::start(const Config& config) {
    stop();
    config_ = config;

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::perror("[WebPreview] socket");
        return false;
    }

    int opt = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));
    if (config_.bind_address.empty() || config_.bind_address == "0.0.0.0") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, config_.bind_address.c_str(), &addr.sin_addr) != 1) {
        std::fprintf(stderr, "[WebPreview] invalid bind address: %s\n", config_.bind_address.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::bind(server_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("[WebPreview] bind");
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::listen(server_fd_, 8) != 0) {
        std::perror("[WebPreview] listen");
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    current_publish_fps_.store(0.0);
    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        fps_window_start_ = std::chrono::steady_clock::now();
        fps_window_count_ = 0;
    }
    replay_pace_enabled_ = config_.replay_pace || replayPaceEnvEnabled();
    replay_started_ = false;
    replay_frame_index_ = 0;
    replay_start_time_ = std::chrono::steady_clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_frames_.clear();
    }
    // turbojpeg 压缩器在 encodeLoop 线程复用；初始化失败时回退 cv::imencode
    if (!tj_compressor_) {
        tj_compressor_ = tjInitCompress();
    }
    accept_thread_ = std::thread(&WebPreviewServer::acceptLoop, this);
    encode_thread_ = std::thread(&WebPreviewServer::encodeLoop, this);
    std::printf("[WebPreview] listening on %s (mode=%s fps=%.2f pace=%d)\n", url().c_str(),
                config_.preview_mode.c_str(), config_.replay_fps, replay_pace_enabled_ ? 1 : 0);
    // #region debug-point D:start
    reportDebugEvent("D",
                     "web_preview_server.cc:start",
                     "[DEBUG] WebPreview server started",
                     std::string("{\"bind_address\":\"") + jsonEscape(config_.bind_address) +
                         "\",\"port\":" + std::to_string(config_.port) +
                         ",\"jpeg_quality\":" + std::to_string(config_.jpeg_quality) + "}");
    // #endregion
    return true;
}

void WebPreviewServer::stop() {
    if (!running_.load()) {
        return;
    }

    if (config_.preview_mode == "replay" && config_.eof_linger_ms > 0 && replay_started_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.eof_linger_ms));
    }

    if (!running_.exchange(false)) {
        return;
    }

    frame_cv_.notify_all();
    queue_cv_.notify_all();

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
    if (tj_compressor_) {
        tjDestroy(tj_compressor_);
        tj_compressor_ = nullptr;
    }

    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }

    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : client_fds_) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
        client_fds_.clear();
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

std::string WebPreviewServer::url() const {
    return std::string("http://") + (config_.bind_address.empty() ? "0.0.0.0" : config_.bind_address) +
           ":" + std::to_string(config_.port) + "/";
}

int WebPreviewServer::clientCount() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return static_cast<int>(client_fds_.size());
}

std::string WebPreviewServer::makeStatusJson() const {
    std::uint64_t input_dropped = 0;
    std::uint64_t input_reconnects = 0;
    if (input_stat_source_) {
        auto stats = input_stat_source_();
        input_dropped = stats.first;
        input_reconnects = stats.second;
    }
    std::ostringstream json;
    json << "{\n"
         << "  \"title\": \"" << jsonEscape(config_.title) << "\",\n"
         << "  \"task\": \"" << jsonEscape(config_.task) << "\",\n"
         << "  \"input\": \"" << jsonEscape(config_.input_path) << "\",\n"
         << "  \"port\": " << config_.port << ",\n"
         << "  \"source_fps\": " << config_.source_fps << ",\n"
         << "  \"frame_width\": " << config_.frame_width << ",\n"
         << "  \"frame_height\": " << config_.frame_height << ",\n"
         << "  \"running\": " << (running_.load() ? "true" : "false") << ",\n"
         << "  \"preview_mode\": \"" << jsonEscape(config_.preview_mode) << "\",\n"
         << "  \"publish_fps\": " << publishFPS() << ",\n"
         << "  \"published_frames\": " << publishedFrames() << ",\n"
         << "  \"dropped_frames\": " << droppedFrames() << ",\n"
         << "  \"encode_ms\": " << encodeMs() << ",\n"
         << "  \"clients\": " << clientCount() << ",\n"
         << "  \"input_dropped_frames\": " << input_dropped << ",\n"
         << "  \"input_reconnect_count\": " << input_reconnects << "\n"
         << "}\n";
    return json.str();
}

void WebPreviewServer::publishFrame(cv::Mat frame, bool pre_scaled) {
    if (!running_.load() || frame.empty()) {
        return;
    }

    const bool paced = replay_pace_enabled_ && config_.preview_mode == "replay" && config_.replay_fps > 0.0;

    // 原帧率播放（web_preview_replay_pace=1）：在生产端按源帧率节流，通过
    // backpressure 让解码/NPU 流水线整体按源帧率推进，避免"显示帧率对但内容跳帧快进"。
    if (paced) {
        if (!replay_started_) {
            replay_started_ = true;
            replay_frame_index_ = 0;
            replay_start_time_ = std::chrono::steady_clock::now();
        }
        const double frame_interval_us = 1000000.0 / config_.replay_fps;
        const auto desired_offset = std::chrono::microseconds(
            static_cast<long long>(std::llround(static_cast<double>(replay_frame_index_) * frame_interval_us)));
        const auto desired_time = replay_start_time_ + desired_offset;
        const auto now = std::chrono::steady_clock::now();
        if (desired_time > now) {
            std::this_thread::sleep_until(desired_time);
        }
        ++replay_frame_index_;
    }

    // 异步编码：输出线程只做格式转换 + 入队，JPEG 编码放到 encodeLoop 独立线程，
    // 避免 cv::imencode 阻塞输出线程（当前主要输出开销）。帧以 cv::Mat 引用计数
    // 保活，入队即持有数据，编码完成后由队列释放。
    if (frame.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
        frame = std::move(bgr);
    } else if (frame.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        frame = std::move(bgr);
    } else if (frame.channels() != 3) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (paced) {
            // 原帧率模式：编码队列满时阻塞生产端（背压到上游流水线），
            // 保证帧不丢弃、内容不跳帧（丢最旧帧会导致浏览器看到"快进"）。
            queue_cv_.wait(lock, [&]() { return !running_.load() || pending_frames_.size() < kMaxPendingFrames; });
            if (!running_.load()) {
                return;
            }
        } else if (pending_frames_.size() >= kMaxPendingFrames) {
            // 非节流（实时预览）模式：队满丢最旧帧，保持低延迟预览并限制内存占用，不阻塞输出线程
            pending_frames_.pop_front();
            dropped_frames_.fetch_add(1);
        }
        PendingFrame pending;
        pending.bgr = std::move(frame);
        pending.pre_scaled = pre_scaled;
        pending_frames_.push_back(std::move(pending));
        published_frames_.fetch_add(1);
    }
    queue_cv_.notify_one();
}

void WebPreviewServer::encodeLoop() {
    pinCurrentThreadFromEnv("RK_PIPE_PIN_ENCODE");
    for (;;) {
        PendingFrame pending;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&]() { return !running_.load() || !pending_frames_.empty(); });
            if (!running_.load()) {
                pending_frames_.clear();
                return;
            }
            pending = std::move(pending_frames_.front());
            pending_frames_.pop_front();
        }
        queue_cv_.notify_one();  // 唤醒因原帧率模式排队满而阻塞的生产端

        // 无 MJPEG 客户端时跳过 JPEG 编码（窗格用 WebRTC iframe / 无人观看时），
        // 释放 CPU 给推理；有人请求 /stream.mjpg 时自动恢复编码。
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            if (client_fds_.empty()) {
                continue;
            }
        }
        encodeAndPublish(std::move(pending.bgr), pending.pre_scaled);
    }
}

void WebPreviewServer::encodeAndPublish(cv::Mat bgr, bool pre_scaled) {
    // 编码前降采样（preview_scale<1.0）：减小 JPEG 像素量，降低编码耗时与预览带宽。
    // 若帧已在生产端（RGA 硬件）完成降采样（pre_scaled），则不再二次缩放。
    if (!pre_scaled && config_.preview_scale > 0.1 && config_.preview_scale < 1.0) {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(), config_.preview_scale, config_.preview_scale, cv::INTER_AREA);
        bgr = std::move(resized);
    }

    const auto encode_begin = std::chrono::steady_clock::now();
    std::vector<unsigned char> jpeg;
    // 优先 turbojpeg（NEON SIMD，比 cv::imencode 快 1.5~2.5x），失败回退 OpenCV。
    if (tj_compressor_) {
        unsigned char* jpeg_buf = nullptr;
        unsigned long jpeg_size = 0;
        if (tjCompress2(tj_compressor_, bgr.data, bgr.cols, bgr.step, bgr.rows,
                        TJPF_BGR, &jpeg_buf, &jpeg_size, TJSAMP_420,
                        config_.jpeg_quality, TJFLAG_FASTDCT) == 0) {
            jpeg.assign(jpeg_buf, jpeg_buf + jpeg_size);
        }
        tjFree(jpeg_buf);
    }
    if (jpeg.empty()) {
        std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
        if (!cv::imencode(".jpg", bgr, jpeg, params)) {
            return;
        }
    }
    const auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - encode_begin)
                               .count();
    encode_ms_.store(static_cast<double>(encode_ms));

    std::size_t jpeg_size = 0;
    std::uint64_t frame_version = 0;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_jpeg_ = std::move(jpeg);
        ++frame_version_;
        jpeg_size = latest_jpeg_.size();
        frame_version = frame_version_;
    }
    // #region debug-point D:publish
    static std::atomic<int> publish_count{0};
    const int count = ++publish_count;
    if ((count % 30) == 1) {
        int active_clients = 0;
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            active_clients = static_cast<int>(client_fds_.size());
        }
        reportDebugEvent("D",
                         "web_preview_server.cc:publish",
                         "[DEBUG] WebPreview published JPEG",
                         std::string("{\"publish_count\":") + std::to_string(count) +
                             ",\"frame_version\":" + std::to_string(frame_version) +
                             ",\"encode_ms\":" + std::to_string(encode_ms) +
                             ",\"jpeg_bytes\":" + std::to_string(jpeg_size) +
                             ",\"active_clients\":" + std::to_string(active_clients) + "}");
    }
    // #endregion
    frame_cv_.notify_all();

    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        ++fps_window_count_;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_window_start_).count();
        if (elapsed_ms >= 1000) {
            current_publish_fps_.store(fps_window_count_ * 1000.0 / static_cast<double>(elapsed_ms));
            fps_window_start_ = now;
            fps_window_count_ = 0;
        }
    }
}

void WebPreviewServer::acceptLoop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load()) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            std::perror("[WebPreview] accept");
            continue;
        }
        std::thread(&WebPreviewServer::handleClient, this, client_fd).detach();
    }
}

void WebPreviewServer::handleClient(int client_fd) {
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (client_fds_.size() >= kMaxClients) {
            ::close(client_fd);
            return;
        }
        client_fds_.insert(client_fd);
    }

    const std::string request = readRequest(client_fd);
    const std::string path = requestPath(request);
    // #region debug-point E:client-path
    reportDebugEvent("E",
                     "web_preview_server.cc:handle-client",
                     "[DEBUG] WebPreview client request",
                     std::string("{\"path\":\"") + jsonEscape(path) +
                         "\",\"client_fd\":" + std::to_string(client_fd) + "}");
    // #endregion

    if (path == "/" || path == "/index.html") {
        sendTextResponse(client_fd,
                         "200 OK",
                         "text/html; charset=utf-8",
                         makeHtmlPage(config_.title, config_.extra_stream_ports, config_.webrtc_urls));
        closeClient(client_fd);
        return;
    }

    if (path == "/healthz") {
        sendTextResponse(client_fd, "200 OK", "text/plain; charset=utf-8", "ok\n");
        closeClient(client_fd);
        return;
    }

    if (path == "/status.json") {
        sendTextResponse(client_fd, "200 OK", "application/json; charset=utf-8", makeStatusJson());
        closeClient(client_fd);
        return;
    }

    if (path == "/snapshot.jpg") {
        std::vector<unsigned char> jpeg;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            jpeg = latest_jpeg_;
        }
        if (jpeg.empty()) {
            sendTextResponse(client_fd, "503 Service Unavailable", "text/plain; charset=utf-8", "frame not ready\n");
        } else {
            sendBinaryResponse(client_fd, "200 OK", "image/jpeg", jpeg);
        }
        closeClient(client_fd);
        return;
    }

    if (path == "/stream.mjpg") {
        streamClient(client_fd);
        return;
    }

    sendTextResponse(client_fd, "404 Not Found", "text/plain; charset=utf-8", "not found\n");
    closeClient(client_fd);
}

void WebPreviewServer::streamClient(int client_fd) {
    const std::string header = streamHeader();
    if (!sendAll(client_fd, header.data(), header.size())) {
        closeClient(client_fd);
        return;
    }

    std::uint64_t last_version = 0;
    int sent_frames = 0;
    while (running_.load()) {
        std::vector<unsigned char> jpeg;
        std::uint64_t version = 0;

        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [&]() { return !running_.load() || frame_version_ > last_version; });
            if (!running_.load()) {
                break;
            }
            jpeg = latest_jpeg_;
            version = frame_version_;
        }

        if (jpeg.empty()) {
            continue;
        }

        std::ostringstream part;
        part << "--" << kBoundary << "\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n\r\n";
        const std::string part_header = part.str();
        if (!sendAll(client_fd, part_header.data(), part_header.size()) ||
            !sendAll(client_fd, jpeg.data(), jpeg.size()) ||
            !sendAll(client_fd, "\r\n", 2)) {
            break;
        }
        last_version = version;
        ++sent_frames;
        // #region debug-point E:stream-progress
        if ((sent_frames % 60) == 1) {
            reportDebugEvent("E",
                             "web_preview_server.cc:stream-client",
                             "[DEBUG] WebPreview streamed frame",
                             std::string("{\"client_fd\":") + std::to_string(client_fd) +
                                 ",\"sent_frames\":" + std::to_string(sent_frames) +
                                 ",\"frame_version\":" + std::to_string(version) +
                                 ",\"jpeg_bytes\":" + std::to_string(jpeg.size()) + "}");
        }
        // #endregion
    }
    // #region debug-point E:stream-close
    reportDebugEvent("E",
                     "web_preview_server.cc:stream-close",
                     "[DEBUG] WebPreview stream closed",
                     std::string("{\"client_fd\":") + std::to_string(client_fd) +
                         ",\"sent_frames\":" + std::to_string(sent_frames) +
                         ",\"last_version\":" + std::to_string(last_version) + "}");
    // #endregion

    closeClient(client_fd);
}

void WebPreviewServer::closeClient(int client_fd) {
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        client_fds_.erase(client_fd);
    }
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
}

bool WebPreviewServer::sendAll(int fd, const void* data, std::size_t size) {
    const char* ptr = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, ptr + sent, size - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

bool WebPreviewServer::sendTextResponse(int fd,
                                        const std::string& status,
                                        const std::string& content_type,
                                        const std::string& body) {
    const std::string header = httpHeader(status, content_type, body.size());
    return sendAll(fd, header.data(), header.size()) && sendAll(fd, body.data(), body.size());
}

bool WebPreviewServer::sendBinaryResponse(int fd,
                                          const std::string& status,
                                          const std::string& content_type,
                                          const std::vector<unsigned char>& body) {
    const std::string header = httpHeader(status, content_type, body.size());
    return sendAll(fd, header.data(), header.size()) &&
           (body.empty() || sendAll(fd, body.data(), body.size()));
}
