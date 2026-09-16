#pragma once

// TaskResult → JSON 序列化（llm 分析 prompt 与 /api/llm/status 共用）。
// 纯函数、无硬件依赖，CI 构建可测。
// 注意：闭源侧暂无 core/task_result_json(schema v1)，此为 llm 模块自带序列化；
// 后续 schema 统一时切换到统一序列化器并保持字段兼容。

#include <string>
#include <vector>

#include "core/task_result.h"

namespace llm {

// 序列化一帧任务结果为紧凑 JSON 对象：
//   {"type":"detect","count":N,"items":[{"cls_id":0,"name":"person","conf":0.87,"box":[l,t,r,b]}...]}
// class_names 非空且 cls_id 在范围内时输出 name 字段，否则回退 coco_cls_to_name / class_<id>。
std::string taskResultToJson(const TaskResult& result, const std::vector<std::string>& class_names);

}  // namespace llm
