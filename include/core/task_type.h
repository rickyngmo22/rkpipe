#pragma once

// 任务类型契约：由 YAML 配置的 task 字段解析得出。
// 解析实现（getTaskType，字符串 -> 枚举）位于内部运行时，开源侧只依赖枚举本身。

enum class TaskType {
    Detect,
    Pose,
    OBB,
    Seg,
    Depth
};
