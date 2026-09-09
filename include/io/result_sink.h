#pragma once

// 逐帧结果 JSONL 文件汇:每帧把 buildFrameResultJson(schema v1,docs/event_payload.md)
// 的输出按行写入文件——与 RK_PIPE_EVENT_RESULT 事件 payload 同格式,是当前核心库下
// 获取逐帧结构化结果的通道(事件下发需配套核心库 Release)。
//
// 配置:环境变量 RK_PIPE_RESULT_JSONL=<文件路径>(空/未设=关闭;每次运行 truncate 重开)。
//   说明:AppConfig 与闭源核心按成员布局直接耦合,公开仓库不能新增配置字段,故过渡期
//   走环境变量;YAML 键 result_jsonl_path 待配套核心 Release 协同启用,届时本函数改为
//   同时读配置字段。sequential 模式暂不生效(其开源钩子拿不到 TaskResult)。
//
// 线程与生命周期契约:
//   - configureFrameResultSink 在闭源核心启动阶段被间接调用:核心以完整配置回调开源
//     shouldEnableOutput(AppConfig)(当前版本下开源侧唯一携带配置的启动缝),在那里
//     转调本函数;下个核心 Release 起由核心直接初始化结果汇,该接线点可退役。
//   - writeFrameResult 由输出线程逐帧调用(单线程有序);内部互斥只保护与 configure/
//     shutdown 的并发,序列化在锁外完成。
//   - 进程正常退出时静态对象析构自动 flush;意外终止可能丢尾部缓冲(逐行容错读取,
//     同 --dump-detections 语义)。
//
// 纯逻辑(stdio/环境变量),进 CI 纯逻辑源集单测覆盖。

struct PipelineFrame;
class AppConfig;

// 幂等初始化:环境变量未设/为空时无副作用;已初始化时重复调用不重开文件。
// 打开失败只告警一次并保持禁用,不在输出线程反复刷错误。
void configureFrameResultSink(const AppConfig& options);

// 逐帧写出:未配置/已禁用/本帧无结果时为 no-op(先判空再序列化,零额外开销)。
void writeFrameResult(const PipelineFrame& frame);

// flush 并关闭,状态复位(之后可重新 configure,文件按 truncate 重新打开)。
void shutdownFrameResultSink();
