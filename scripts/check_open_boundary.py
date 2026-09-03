#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_open_boundary.py — 开源边界自检。

在开源仓库内运行,验证两件事:
  1. 开源代码没有引用闭源核心的任何符号/头(以已知闭源组件名清单为准);
  2. 开源代码中不存在密钥/内网地址等敏感内容。

CI(见 .github/workflows/ci.yml)与发布流水线都会运行本脚本;
失败即说明边界被破坏,禁止发布。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCAN_DIRS = ["include", "src", "examples", "tests", "tools"]
SCAN_EXT = (".h", ".cc", ".cpp", ".c", ".py", ".sh", ".yaml", ".yml", ".txt", ".md")

# 闭源组件标识: 这些名字只应出现在文档的"闭源说明"章节,不应出现在代码/头文件
CLOSED_MARKERS = [
    "ProcessingPipeline",      # 调度器类
    "initializeAppRuntime",    # 核心入口
    "runConfiguredApp",        # 核心入口
    "openInputSourceOrReport", # 核心入口(门面内部)
    "AppRunner",               # 核心编排
    "SystemRuntime",           # 核心编排
    "TaskFrameProcessor",      # 核心编排
    "thread_local_memory_pool_priv",
    "daemon_main",             # REST 守护进程入口(仅内部仓库)
    "app_runner.h", "app_runtime.h", "system_runtime.h",
    "processing_pipeline.h", "task_frame_processor.h",
    "mempool.h", "threadpool.h",
    "runtime_support.h", "runtime_overlay.h", "runtime_stats.h",
    "runtime_debug.h", "alert_runtime.h", "detector_runtime.h",
]

# 敏感内容: 私钥材料 / 凭据 / 常见内网网段字面量
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "TLS/SSH 私钥"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"'][^\"']{4,}"), "疑似凭据"),
    (re.compile(r"rtsp://[^\s\"']*:[^\s@\"']+@"), "RTSP 带凭据 URL"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "内网 IP 字面量"),
]

failures = []

for d in SCAN_DIRS:
    base = os.path.join(ROOT, d)
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for name in files:
            if not name.endswith(SCAN_EXT):
                continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, ROOT)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for marker in CLOSED_MARKERS:
                if marker in text:
                    line_no = text[:text.index(marker)].count("\n") + 1
                    failures.append(f"[闭源引用] {rel}:{line_no} 出现闭源标识 '{marker}'")
            for pat, desc in SECRET_PATTERNS:
                for m in pat.finditer(text):
                    line_no = text[:m.start()].count("\n") + 1
                    failures.append(f"[敏感内容] {rel}:{line_no} {desc}")

if failures:
    print("边界自检失败:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("边界自检通过: 无闭源符号引用,无敏感内容")
