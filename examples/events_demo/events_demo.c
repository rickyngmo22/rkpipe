/*
 * 最小事件回调示例(纯 C): create -> set_event_cb -> start -> wait -> destroy。
 *
 * 演示 rkpipe C ABI 的完整生命周期。本地视频等有限输入会自然结束,
 * 实时流(RTSP)会持续运行直到 Ctrl-C 终止进程。
 */
#include <signal.h>
#include <stdio.h>

#include "rkpipe/rkpipe.h"

static void on_event(rkpipe_event_t ev, const char* payload_json, void* user_ctx) {
    (void)user_ctx;
    const char* name = "?";
    switch (ev) {
        case RK_PIPE_EVENT_STARTED:   name = "STARTED";   break;
        case RK_PIPE_EVENT_RESULT:    name = "RESULT";    break;
        case RK_PIPE_EVENT_COMPLETED: name = "COMPLETED"; break;
        case RK_PIPE_EVENT_ERROR:     name = "ERROR";     break;
    }
    printf("[events_demo] %s: %s\n", name, payload_json ? payload_json : "{}");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config.yaml>\n", argv[0]);
        fprintf(stderr, "  配置字段语义见 examples/configs 与 README\n");
        return 2;
    }

    rkpipe_handle h = rkpipe_create(argv[1]);
    if (!h) {
        fprintf(stderr, "create failed: 无法加载配置或初始化失败\n");
        return 1;
    }
    rkpipe_set_event_cb(h, on_event, NULL);
    rkpipe_start(h);
    rkpipe_wait(h, -1);  /* 无限等待自然结束 */

    char* status = rkpipe_get_status(h);
    printf("[events_demo] status: %s\n", status ? status : "{}");
    rkpipe_free(status);

    rkpipe_destroy(h);
    return 0;
}
