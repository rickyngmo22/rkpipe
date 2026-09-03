/*
 * rkpipe CLI —— 通过 C ABI(rkpipe/rkpipe.h)驱动完整流水线的命令行工具。
 *
 * 同时承担 ABI 验收测试的职责: 本程序只使用 rkpipe.h 声明的接口,
 * 与闭源核心的任何内部头无关;若本程序能编译链接并跑通,则 ABI 契约成立。
 *
 * 用法:
 *   rkpipe_cli <config.yaml> [--quiet]
 */
#include <signal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rkpipe/rkpipe.h"

static volatile sig_atomic_t g_interrupted = 0;

static void on_signal(int) { g_interrupted = 1; }

static const char* event_name(rkpipe_event_t ev) {
    switch (ev) {
        case RK_PIPE_EVENT_STARTED:   return "STARTED";
        case RK_PIPE_EVENT_RESULT:    return "RESULT";
        case RK_PIPE_EVENT_COMPLETED: return "COMPLETED";
        case RK_PIPE_EVENT_ERROR:     return "ERROR";
    }
    return "?";
}

static void on_event(rkpipe_event_t ev, const char* payload_json, void* user_ctx) {
    int quiet = *(int*)user_ctx;
    if (!quiet) {
        printf("[rkpipe] %s %s\n", event_name(ev), payload_json ? payload_json : "{}");
    }
    /* 注意: 回调在库内部线程触发,此处不要调用 rkpipe_* 接口(见 README 线程契约) */
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config.yaml> [--quiet]\n", argv[0]);
        return 2;
    }
    int quiet = (argc > 2 && strcmp(argv[2], "--quiet") == 0);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    rkpipe_handle h = rkpipe_create(argv[1]);
    if (!h) {
        fprintf(stderr, "rkpipe_create failed: config load/init error (%s)\n", argv[1]);
        return 1;
    }

    rkpipe_set_event_cb(h, on_event, &quiet);

    int rc = rkpipe_start(h);
    if (rc != RK_PIPE_OK) {
        fprintf(stderr, "rkpipe_start failed: %d\n", rc);
        rkpipe_destroy(h);
        return 1;
    }

    /* 输入驱动: 本地文件/有限流自然结束;实时流持续运行,Ctrl-C 退出进程 */
    while (!g_interrupted) {
        rc = rkpipe_wait(h, 200);
        if (rc == RK_PIPE_OK) break;
        if (rc != RK_PIPE_ERR_TIMEOUT) break;
    }

    char* status = rkpipe_get_status(h);
    if (status) {
        printf("[rkpipe] final status: %s\n", status);
        rkpipe_free(status);
    }
    int exit_code = rkpipe_get_exit_code(h);
    rkpipe_destroy(h);
    return exit_code == RK_PIPE_OK ? 0 : 1;
}
