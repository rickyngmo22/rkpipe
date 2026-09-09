#include "../../include/io/output_runtime.h"

#include "io/result_sink.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/stat.h>
#include <string_view>
#include <sys/socket.h>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

struct DisplayLaunchEnv {
    bool available = false;
    std::string shell_prefix;
    std::string description;
};

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

bool fileExists(const std::string& path) {
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0;
}

DisplayLaunchEnv detectDisplayLaunchEnv() {
    DisplayLaunchEnv env;

    const char* display = std::getenv("DISPLAY");
    if (display && *display) {
        env.available = true;
        env.description = std::string("DISPLAY=") + display;
        return env;
    }

    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    const char* xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (wayland_display && *wayland_display) {
        env.available = true;
        env.description = std::string("WAYLAND_DISPLAY=") + wayland_display;
        return env;
    }

    const uid_t uid = ::getuid();
    const std::string runtime_dir =
        (xdg_runtime_dir && *xdg_runtime_dir) ? std::string(xdg_runtime_dir)
                                              : std::string("/run/user/") + std::to_string(uid);

    const std::string wayland_socket = runtime_dir + "/wayland-0";
    if (fileExists(wayland_socket)) {
        env.available = true;
        env.shell_prefix =
            "XDG_RUNTIME_DIR=" + shellQuote(runtime_dir) + " WAYLAND_DISPLAY=wayland-0 ";
        env.description = "WAYLAND_DISPLAY=wayland-0 (auto)";
        return env;
    }

    if (fileExists("/tmp/.X11-unix/X0")) {
        env.available = true;
        env.shell_prefix = "DISPLAY=:0 ";
        env.description = "DISPLAY=:0 (auto)";
        return env;
    }

    return env;
}

bool ffplayAvailable() {
    const char* path = std::getenv("PATH");
    if (!path || !*path) {
        return false;
    }

    std::string path_value(path);
    std::size_t start = 0;
    while (start <= path_value.size()) {
        const std::size_t end = path_value.find(':', start);
        const std::string dir = path_value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const std::string candidate = (dir.empty() ? "." : dir) + "/ffplay";
        if (::access(candidate.c_str(), X_OK) == 0) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

bool parseUdpPort(const std::string& url, int* port) {
    if (!port) {
        return false;
    }
    *port = 0;

    constexpr std::string_view kUdpPrefix("udp://");
    if (url.compare(0, kUdpPrefix.size(), kUdpPrefix) != 0) {
        return false;
    }

    const std::size_t host_begin = kUdpPrefix.size();
    const std::size_t port_sep = url.find(':', host_begin);
    if (port_sep == std::string::npos) {
        return false;
    }
    const std::size_t port_end = url.find('?', port_sep + 1);
    const std::string port_text = url.substr(port_sep + 1, port_end == std::string::npos ? std::string::npos : port_end - port_sep - 1);
    if (port_text.empty()) {
        return false;
    }

    char* parse_end = nullptr;
    const long parsed = std::strtol(port_text.c_str(), &parse_end, 10);
    if (!parse_end || *parse_end != '\0' || parsed <= 0 || parsed > 65535) {
        return false;
    }

    *port = static_cast<int>(parsed);
    return true;
}

bool isUdpPortAvailable(const std::string& url) {
    int port = 0;
    if (!parseUdpPort(url, &port)) {
        return true;
    }

    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return true;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    const int bind_result = ::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    const int bind_errno = errno;
    ::close(fd);

    if (bind_result == 0) {
        return true;
    }
    if (bind_errno == EADDRINUSE) {
        return false;
    }
    return true;
}

}  // namespace

bool shouldEnableOutput(const AppConfig& options) {
    // 逐帧结果 JSONL 汇初始化(RK_PIPE_RESULT_JSONL,见 io/result_sink.h):
    // 闭源核心启动阶段会以完整配置回调本函数,这是当前核心下开源侧唯一携带配置的
    // 启动缝;下个核心 Release 起由核心直接初始化结果汇,本接线点退役。
    configureFrameResultSink(options);
    if (options.push_local) {
        return true;
    }
    if (options.web_preview) {
        return true;
    }

    std::string backend = options.output_backend;
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (backend == "opencv") {
        return options.gui;
    }
    return !options.output_video_path.empty();
}

void launchLocalPreview(const AppConfig& options) {
    if (!options.push_local) {
        return;
    }

    if (options.gui) {
        std::printf("Local preview handled by in-process OpenCV display because gui=1.\n");
        return;
    }

    const char* disable_ffplay = std::getenv("RK_PIPE_DISABLE_FFPLAY");
    if (disable_ffplay && std::strcmp(disable_ffplay, "1") == 0) {
        std::printf("Local streaming launch skipped by RK_PIPE_DISABLE_FFPLAY.\n");
        return;
    }

    const DisplayLaunchEnv display_env = detectDisplayLaunchEnv();
    if (!display_env.available) {
        std::printf("Local preview skipped: no DISPLAY or WAYLAND_DISPLAY in current shell.\n");
        return;
    }

    if (!ffplayAvailable()) {
        std::printf("Local preview skipped: ffplay not found in PATH.\n");
        return;
    }

    if (!isUdpPortAvailable(options.output_video_path)) {
        std::printf("Local preview skipped: UDP port is already in use for %s\n", options.output_video_path.c_str());
        std::printf("Hint: close existing ffplay or change output_video_path.\n");
        return;
    }

    std::printf("Local streaming enabled. Output: %s\n", options.output_video_path.c_str());
    std::printf("Local preview display env: %s\n", display_env.description.c_str());
    std::printf("Launching ffplay in 0.2 seconds with software H.264 decode... (Log: ffplay_log.txt)\n");

    std::thread([options, display_env]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::string cmd =
            display_env.shell_prefix +
            "ffplay -fflags nobuffer -flags low_delay -framedrop -vcodec h264 "
            "-vf format=yuv420p -i \"" +
            options.output_video_path +
            "\" > ffplay_log.txt 2>&1";
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::printf("ffplay exited with error code: %d. Check ffplay_log.txt for details.\n", ret);
        }
    }).detach();
}
