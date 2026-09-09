/*
 * rk_pipe SDK 公共 API（v0.2）
 *
 * 这是 rk_pipe 对外暴露的唯一接口，纯 C ABI、稳定契约。
 * 内部实现（流水线编排 / RKNN 推理 / 模型后处理 / RGA 零拷贝等）对调用方完全不可见：
 *   - 所有内部类型（AppConfig、TaskResult、image_buffer_t、rknn_app_context_t ...）
 *     均不通过本头文件泄漏；
 *   - 库内符号默认隐藏（-fvisibility=hidden），仅本头声明的函数可见。
 *
 * 配置即契约：rkpipe_create 接收 YAML 配置路径，字段语义与 rk_pipe 命令行配置一致
 * （task/mode/model_path/input_path/thread_count/output_video_path/web_preview_*...）。
 */
#ifndef RKPIPE_RKPIPE_H
#define RKPIPE_RKPIPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RK_PIPE_API __declspec(dllexport)
#else
#define RK_PIPE_API __attribute__((visibility("default")))
#endif

/* 不透明句柄：一个已加载配置的流水线实例 */
typedef struct rkpipe* rkpipe_handle;

typedef enum {
    RK_PIPE_OK = 0,
    RK_PIPE_ERR_INVALID_ARG = -1,  /* 参数非法（空指针/非法句柄） */
    RK_PIPE_ERR_LOAD_CONFIG = -2,  /* YAML 配置加载失败 */
    RK_PIPE_ERR_INIT_RUNTIME = -3, /* 运行时初始化失败（模型/labels/后处理） */
    RK_PIPE_ERR_OPEN_INPUT = -4,   /* 输入源打开失败 */
    RK_PIPE_ERR_STATE = -5,        /* 状态非法（重复 start 等） */
    RK_PIPE_ERR_TIMEOUT = -6,      /* 等待超时 */
    RK_PIPE_ERR_INTERNAL = -7      /* 流水线内部错误 */
} rkpipe_status_t;

/* 事件类型；payload_json 为对应 JSON 文本 */
typedef enum {
    RK_PIPE_EVENT_STARTED = 0,   /* 流水线已启动：{"task":"detect","mode":"pipeline"} */
    RK_PIPE_EVENT_RESULT = 1,    /* 逐帧结构化结果：payload 为 schema v1 JSON（字段定义见
                                    docs/event_payload.md）。当前版本预留不触发：获取逐帧结果
                                    请先用环境变量 RK_PIPE_RESULT_JSONL=<file> 落盘同格式
                                    JSONL；事件下发需配套核心库 Release（闭源门面逐帧调用
                                    开源序列化器 buildFrameResultJson 后回调） */
    RK_PIPE_EVENT_COMPLETED = 2, /* 自然结束：{"exit_code":0} */
    RK_PIPE_EVENT_ERROR = 3      /* 运行错误：{"error":"..."} */
} rkpipe_event_t;

typedef void (*rkpipe_event_cb)(rkpipe_event_t ev, const char* payload_json, void* user_ctx);

/* 库版本 */
RK_PIPE_API const char* rkpipe_version(void);

/*
 * 由 YAML 配置创建流水线句柄（只加载配置，不启动）。
 * 失败返回 NULL。
 */
RK_PIPE_API rkpipe_handle rkpipe_create(const char* config_yaml_path);

/* 销毁句柄。若流水线仍在运行会先等待其结束。 */
RK_PIPE_API void rkpipe_destroy(rkpipe_handle h);

/* 注册事件回调（建议在 start 前调用；可随时更新）。 */
RK_PIPE_API int rkpipe_set_event_cb(rkpipe_handle h, rkpipe_event_cb cb, void* user_ctx);

/*
 * 启动流水线（非阻塞，在后台线程运行）。
 * 重复 start 返回 RK_PIPE_ERR_STATE。
 */
RK_PIPE_API int rkpipe_start(rkpipe_handle h);

/*
 * 请求停止并等待流水线线程退出。
 * 注意 v0.2：流水线为"输入驱动"，本地文件/有限流会自然结束；
 * 实时流（RTSP 等）会持续运行，timeout_ms<0 时阻塞至外部终止输入或进程退出。
 * 返回 RK_PIPE_OK（已结束）或 RK_PIPE_ERR_TIMEOUT。
 */
RK_PIPE_API int rkpipe_stop(rkpipe_handle h, int timeout_ms);

/*
 * 等待流水线自然结束（适合本地文件/有限流）。
 * timeout_ms<0 无限等待。返回 RK_PIPE_OK（已结束）或 RK_PIPE_ERR_TIMEOUT。
 */
RK_PIPE_API int rkpipe_wait(rkpipe_handle h, int timeout_ms);

/* 状态 JSON 快照（调用方用 rkpipe_free 释放）：
 * {"task":"...","mode":"...","state":0|1|2,"exit_code":N} */
RK_PIPE_API char* rkpipe_get_status(rkpipe_handle h);

/* 流水线退出码（RK_PIPE_OK=0 或负的错误码；未运行/运行中返回 0） */
RK_PIPE_API int rkpipe_get_exit_code(rkpipe_handle h);

/* 释放 rkpipe_get_status 返回的内存 */
RK_PIPE_API void rkpipe_free(void* p);

#ifdef __cplusplus
}
#endif

#endif /* RKPIPE_RKPIPE_H */
