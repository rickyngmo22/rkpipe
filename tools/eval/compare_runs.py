#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_runs.py —— 跨任务历史对比表（论文风格的模型对比汇总）。

扫多个 rknn_eval 输出目录，读 summary_*.json（单模型摘要）汇总成一张对比表：
同任务不同模型（INT8/FP16/不同架构）的精度 + 速度并排，量化损失自动计算。

用法：
  python3 compare_runs.py runs/detect runs/pose runs/seg runs/obb [--out compare.md]
  python3 compare_runs.py web_runs/*           # Web 任务产物同样支持
"""

import glob
import json
import os
import sys


def load_runs(dirs):
    rows = []
    for d in dirs:
        for sp in sorted(glob.glob(os.path.join(d, "summary_*.json"))):
            tag = os.path.splitext(os.path.basename(sp))[0].replace("summary_", "")
            if tag.startswith("conf"):  # conf 扫描产物不进对比表
                continue
            try:
                s = json.load(open(sp))
            except (OSError, ValueError):
                continue
            if not isinstance(s, dict) or "kind" not in s:
                continue
            mp = os.path.join(d, "metrics_%s.json" % tag)
            fps = 0.0
            if os.path.exists(mp):
                try:
                    fps = json.load(open(mp))["rk_pipe"]["pipeline"]["avg_fps"]
                except (OSError, ValueError, KeyError):
                    pass
            rows.append({"dir": d, "tag": tag, "kind": s.get("kind", "bbox"), "s": s, "fps": fps})
    return rows


def metric(row, key):
    v = row["s"].get(key)
    return None if v is None else float(v) * 100.0


def fmt(v):
    return "%.1f" % v if v is not None else "-"


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if not dirs:
        print(__doc__)
        return 1

    rows = load_runs(dirs)
    if not rows:
        print("未找到可用的 summary_*.json")
        return 1

    cols_by_kind = {
        "bbox": ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_max"],
        "segm": ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_max"],
        "kpts": ["AP", "AP50", "AP75", "AP_medium", "AP_large", "AR_20"],
    }

    md = ["# 评测历史对比\n"]
    cur = None
    for r in rows:
        if r["kind"] != cur:
            cur = r["kind"]
            cols = [c for c in cols_by_kind.get(cur, ["AP", "AP50"]) if any(c in x["s"] for x in rows if x["kind"] == cur)]
            md.append("\n## 任务类型: %s\n" % cur)
            md.append("| 目录 | 模型 | " + " | ".join(cols) + " | FPS |")
            md.append("|---|---|" + "---|" * (len(cols) + 1))
        md.append("| %s | %s | %s | %.1f |" %
                  (r["dir"], r["tag"],
                   " | ".join(fmt(metric(r, c)) for c in cols), r["fps"]))

    # 量化损失：同 kind 内逐对（首个 vs 其余）
    md.append("\n## 量化损失（各类型首个模型为基准）\n")
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    for kind, rs in by_kind.items():
        base, others = rs[0], rs[1:]
        if not others:
            continue
        m0 = "mAP50_95" if kind == "obb" else "AP"
        md.append("\n### %s（基准 %s/%s）\n" % (kind, base["dir"], base["tag"]))
        md.append("| 模型 | Δ主指标 | Δ速度 |")
        md.append("|---|---|---|")
        b0 = metric(base, m0)
        for o in others:
            d0 = metric(o, m0) - b0 if (metric(o, m0) is not None and b0 is not None) else None
            speed = o["fps"] / base["fps"] if base["fps"] > 0 else 0
            md.append("| %s/%s | %s | %.2fx |" % (o["dir"], o["tag"], fmt(d0), speed))

    text = "\n".join(md) + "\n"
    print(text)
    if out:
        open(out, "w").write(text)
        print("已写入", out)
    return 0


def time_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    sys.exit(main())
