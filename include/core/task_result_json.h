#pragma once

// 逐帧任务结果 → JSON(schema v1)序列化纯函数,schema 权威文档见 docs/event_payload.md。
//
// 两条传输通道使用同一 schema:
//   1. 环境变量 RK_PIPE_RESULT_JSONL 指定的 JSONL 文件汇(io/result_sink.h)——当前核心库即可用;
//   2. RK_PIPE_EVENT_RESULT 事件 payload——需配套核心库 Release,闭源门面逐帧调用
//      buildFrameResultJson 后经 rkpipe_event_cb 下发。
//
// 纯逻辑、无硬件依赖,进 CI 纯逻辑源集单测覆盖。版本策略:v1 内字段只增不改,
// 破坏性变更升 kResultPayloadSchemaVersion 并同步 docs/event_payload.md。

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/pipeline_frame.h"

// 当前 payload schema 版本(对应 docs/event_payload.md 的 v1)
constexpr int kResultPayloadSchemaVersion = 1;

/*
 * 把一帧的完整结果(TaskResult 变体 + 帧信封)序列化为单行 JSON(不带换行符)。
 *
 * detect_track_ids 为 detect 任务的逐目标跟踪 ID(与 detect 结果顺序对齐,可为空):
 *   - JSONL 汇阶段(v1)恒为 nullptr,detect 结果不带 track_id——跟踪 ID 只存在于
 *     renderPipelineTrackingOutput 的返回值中,当前核心库不会把它传到本函数;
 *   - 下个核心 Release 的 RESULT 事件缝由闭源门面传入,届时 detect 结果带 track_id。
 * pose/obb/seg 的跟踪 ID 在结果结构体内,不受此参数影响。
 *
 * frame.hasResult=false 或 variant 为空 → type:"none" 信封(字段定义见 schema 文档)。
 */
std::string buildFrameResultJson(const PipelineFrame& frame,
                                 const std::vector<int>* detect_track_ids = nullptr);

/*
 * seg 掩膜(框内裁剪二值 CV_8UC1,>0 为前景)→ 行主序 RLE 计数数组字符串 "[c0,c1,...]"。
 * 计数数组从 0 值游程开始(首像素为前景时首项为 0),与 tools/eval eval_coco.py
 * decode_box_rle 解码端闭合;掩膜空/类型不符 → "[]"。
 */
std::string encodeBoxMaskRle(const cv::Mat& binary_mask);

// JSON 字符串逃逸:引号/反斜杠/换行等控制字符;<0x20 的其余控制字符转 \u00XX;
// UTF-8 字节原样透传。
std::string jsonEscape(const std::string& value);
