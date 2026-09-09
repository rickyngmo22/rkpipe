#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_eval.py —— PC 端远程评测驱动（数据集在 PC，推理在 RK3588 板端）。

流程（设计文档 §2）：转换/校验(PC) → rsync 推图 → ssh 板端 rknn_eval --dump-only
→ scp 拉 dump/metrics → PC 端 rknn_eval --reuse-dump 评测出报告。

用法：
  python3 run_eval.py --spec datasets/my.yaml --board user@<board-ip> \
      --model board:<板上路径>/y26n.rknn \
      --task detect --ann datasets/my.coco.json \
      [--compare fp16=board:<板上路径>/y26n_fp16.rknn] \
      [--out-dir runs/my] [--preview] [--keep-remote]

说明：
  --model board:<板上路径>   模型已在板上（推荐，scp 一次反复评测）
  --model <本地文件>         先 scp 上板（到 /tmp/rk_eval/<job>/models/）
  spec.board_images 板上已有图片目录时可跳过推图

链路健壮性（PC↔板）：
  ssh ControlMaster 连接复用（整个 job 一条多路复用连接）+ 保活探测 +
  首连自动收 host key；rsync --partial/--timeout 断点续传；推图/拉取失败自动重试；
  整集推图只传图片扩展名（不把标注/labels 一起推上去）。

阶段状态机（--state-file，设计文档 §2，Web 轮询展示）：
  preparing → pushing(推图进度 pct / 分批 batch i/n) → running(模型 i/n)
  → pulling(拉 dump) → evaluating → done / error(含原因)。中断后人工重跑：
  rsync 增量只补差异，已完成部分不重传。
"""

import argparse
import glob
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time
import uuid

try:
    import yaml
except ImportError:
    yaml = None

REMOTE_WORK = "/tmp/rk_eval"
DEFAULT_REMOTE_BIN = "~/rkpipe/build/rknn_eval"
DEFAULT_REMOTE_CWD = "~/rkpipe"  # label 等默认值是相对路径（app_runtime.cc）
IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")

_STATE_FILE = None       # --state-file：阶段状态机落库（Web 每秒轮询）
_STATE_INFO = {}         # 当前阶段的明细（pct/batch/model…），换阶段整体替换
_T0 = time.time()


def set_state(state, info=None):
    """写阶段状态（tmp+rename 原子替换）；未指定 --state-file 时为空操作。"""
    if not _STATE_FILE:
        return
    if info is not None:
        _STATE_INFO.clear()
        _STATE_INFO.update(info)
    rec = {"state": state, "elapsed_s": round(time.time() - _T0, 1),
           "ts": time.strftime("%F %T")}
    rec.update(_STATE_INFO)
    tmp = _STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, _STATE_FILE)
    except OSError:
        pass


def sh(cmd, capture=False, stream_log=None, on_line=None):
    """本地执行；stream_log 流式转发 stdout 并落盘；on_line 逐行回调
    （rsync --info=progress2 用 \\r 刷新同一行，此处按 \\r/\\n 都切行）。"""
    print("+", cmd)
    if capture and stream_log is None and on_line is None:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    logf = open(stream_log, "a") if stream_log else None
    try:
        if on_line is None:
            for line in p.stdout:
                print("  |", line.rstrip())
                if logf:
                    logf.write(line)
        else:
            buf = ""
            while True:
                ch = p.stdout.read(1)
                if not ch:
                    if buf:
                        if logf:
                            logf.write(buf + "\n")
                        on_line(buf)
                    break
                if ch in "\r\n":
                    if buf:
                        if logf:
                            logf.write(buf + "\n")
                        on_line(buf)
                    buf = ""
                else:
                    buf += ch
    finally:
        if logf:
            logf.close()
        p.wait()
    return p.returncode


def sh_retry(cmd, tries=3, what="", on_line=None):
    """链路类命令（rsync/scp）瞬断重试；配 --partial 续传而非重传。"""
    for i in range(1, tries + 1):
        rc = sh(cmd, on_line=on_line)
        if rc == 0:
            return 0
        if i < tries:
            print("[run_eval] %s 失败(rc=%d)，%ds 后重试（%d/%d）"
                  % (what, rc, 2 * i, i, tries - 1))
            time.sleep(2 * i)
    return rc


class BoardLink:
    """PC↔板 ssh 通道：ControlMaster 首连建复用套接字，之后 ssh/scp/rsync
    全部零握手（一次 job 几十条 ssh 命令提速明显）；保活探测断连；
    StrictHostKeyChecking=accept-new 免去首次连接的人工确认。"""

    def __init__(self, board, job_id):
        self.board = board
        self.opts = ["-o", "ControlMaster=auto",
                     "-o", "ControlPath=/tmp/rkssh_%s_%%r@%%h:%%p" % job_id,
                     "-o", "ControlPersist=600",
                     "-o", "ServerAliveInterval=15",
                     "-o", "ServerAliveCountMax=4",
                     "-o", "StrictHostKeyChecking=accept-new",
                     "-o", "ConnectTimeout=10"]
        self.opt_s = " ".join(shlex.quote(o) for o in self.opts)

    def ssh(self, cmd):
        return "ssh %s %s %s" % (self.opt_s, shlex.quote(self.board), shlex.quote(cmd))

    def scp(self, src, dst):
        return "scp %s %s %s" % (self.opt_s, shlex.quote(src), shlex.quote(dst))

    def rsync(self, args):
        """args 为已构造好的参数词（含源/目的），统一走复用连接。"""
        return "rsync %s -e %s" % (" ".join(args), shlex.quote("ssh " + self.opt_s))


def main():
    global _STATE_FILE, _T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="数据集 spec yaml（同 convert_dataset.py）")
    ap.add_argument("--board", required=True, help="ssh 目标，如 user@<board-ip>")
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, help="board:<板上路径> 或本地文件")
    ap.add_argument("--compare", action="append", default=[],
                    help="tag=board:path 或 tag=<本地文件> 或 tag=dump:<已拉回dump>")
    ap.add_argument("--ann", help="标注 JSON（PC 路径；yolo/voc 且未预转换时可不填自动转换）")
    ap.add_argument("--gt", help="obb：切片级 GT jsonl（PC 评测用，板端推理不需要）")
    ap.add_argument("--label", default=None, help="模型 label 文件（远程执行会转绝对路径）")
    ap.add_argument("--obj-num", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--preview", action="store_true", help="板端开 MJPEG 实时预览")
    ap.add_argument("--preview-port", type=int, default=8090)
    ap.add_argument("--remote-bin", default=DEFAULT_REMOTE_BIN)
    ap.add_argument("--remote-cwd", default=DEFAULT_REMOTE_CWD,
                    help="板端命令的执行目录（rknn_eval 的 label 等默认值为相对路径）")
    ap.add_argument("--keep-remote", action="store_true", help="保留板上工作目录")
    ap.add_argument("--push-batch", type=int, default=0,
                    help="分批推图：每批 N 张，批完即删板图（0=整集一次推；大集防涨满板端存储）")
    ap.add_argument("--skip-push", action="store_true", help="跳过推图（已推过）")
    ap.add_argument("--state-file", default=None,
                    help="阶段状态机 JSON 落盘路径（Web 控制台轮询展示进度）")
    args = ap.parse_args()

    if yaml is None:
        raise SystemExit("需要 pyyaml（pip install pyyaml）")
    if not args.spec:
        raise SystemExit("需要 --spec（数据集描述，见设计文档 §3.1）")

    _STATE_FILE = args.state_file
    _T0 = time.time()
    try:
        return run(args)
    except SystemExit as e:
        set_state("error", info={"error": str(e)})
        raise
    except Exception as e:  # 未知异常同样落库，Web 显示失败原因
        set_state("error", info={"error": "%s: %s" % (type(e).__name__, e)})
        raise


def run(args):
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    link = BoardLink(args.board, uuid.uuid4().hex[:6])

    job = "%s_%s" % (spec.get("name", "ds"), uuid.uuid4().hex[:6])
    out_dir = args.out_dir or os.path.join("runs", job)
    os.makedirs(out_dir, exist_ok=True)
    remote_job = posixpath.join(REMOTE_WORK, job)
    os.makedirs(os.path.join(out_dir, "remote"), exist_ok=True)
    set_state("preparing", {"note": "数据集转换/校验（PC 端）"})

    # ---- 1. 数据集准备（PC 端转换/校验）----
    is_obb = args.task == "obb"
    ann, gt_jsonl = args.ann, None
    if is_obb:
        # obb：GT 为 DOTA jsonl。spec.format=yolo-obb 时先转换；dota 原生用户直接
        # 在板端用 --dota-labels 切片（见 rknn_eval），此处只处理"图已是小图"的场景
        if spec.get("format") == "yolo-obb":
            gt_jsonl = os.path.join(out_dir, "dataset.gt.jsonl")
            cmd = ["python3", os.path.join(os.path.dirname(__file__), "convert_dataset.py"),
                   "--spec", args.spec, "--out", gt_jsonl]
            if sh(" ".join(shlex.quote(c) for c in cmd)) != 0 or not os.path.exists(gt_jsonl):
                raise SystemExit("yolo-obb 转换失败")
        elif args.gt:
            gt_jsonl = args.gt
        # gt_jsonl 为空时 PC 端评测需显式 --gt（大图切片 GT），否则第 7 步报错
    elif spec.get("format") == "coco":
        ann = ann or spec.get("labels") or spec.get("ann")
        if not ann or not os.path.exists(ann):
            raise SystemExit("coco 数据集需 --ann 或 spec.labels 指向标注 JSON")
    else:
        conv_out = os.path.join(out_dir, "dataset.coco.json")
        cmd = ["python3", os.path.join(os.path.dirname(__file__), "convert_dataset.py"),
               "--spec", args.spec, "--out", conv_out]
        if sh(" ".join(shlex.quote(c) for c in cmd)) != 0 or not os.path.exists(conv_out):
            raise SystemExit("数据集转换失败")
        ann = ann or conv_out

    # ---- 2. 推图（rsync 增量断点续传；--push-batch 分批模式边推边删）----
    board_images = spec.get("board_images")
    remote_images = posixpath.join(remote_job, "images")
    batched = args.push_batch > 0 and not board_images and not args.skip_push
    if board_images:
        print("[run_eval] 使用板上已有图片:", board_images)
        remote_images = board_images
        set_state("pushing", {"pct": 100, "note": "board_images 板上已有，跳过推图"})
    elif args.skip_push:
        print("[run_eval] --skip-push：假设图已在", remote_images)
        set_state("pushing", {"pct": 100, "note": "--skip-push 跳过推图"})

    img_files = []
    if batched:
        for f in sorted(os.listdir(spec["images"])):
            if f.lower().endswith(tuple("." + e for e in IMG_EXTS)):
                img_files.append(f)
        n_batches = (len(img_files) + args.push_batch - 1) // args.push_batch
        print("[run_eval] 分批推图: %d 张 / 每批 %d = %d 批" %
              (len(img_files), args.push_batch, n_batches))

    _pct_band = {"v": -1}

    def push_progress(line):
        """rsync --info=progress2 的 x% → 状态落库（5% 粒度节流）。"""
        m = re.search(r"(\d+)%", line)
        if m:
            pct = int(m.group(1))
            if pct // 5 > _pct_band["v"]:
                _pct_band["v"] = pct // 5
                set_state("pushing", {"pct": pct, "note": "rsync 增量推图"})

    def push_batch(bi):
        """rsync 第 bi 批（--files-from 清单；接收端目录需预先存在）。"""
        lst = os.path.join(out_dir, "remote", "batch_%d.txt" % bi)
        with open(lst, "w") as f:
            f.write("\n".join(img_files[bi * args.push_batch:(bi + 1) * args.push_batch]))
        set_state("pushing", {"batch": bi + 1, "batches": n_batches, "pct": 0})
        if sh(link.ssh("mkdir -p %s" % shlex.quote(remote_images))) != 0:
            raise SystemExit("ssh 建目录失败")
        if sh_retry(link.rsync(["-az", "--partial", "--timeout=180",
                                "--files-from=" + shlex.quote(lst),
                                shlex.quote(spec["images"]) + "/",
                                shlex.quote("%s:%s/" % (args.board, remote_images))]),
                    what="rsync 推批 %d" % bi) != 0:
            raise SystemExit("rsync 推批 %d 失败" % bi)

    if not board_images and not args.skip_push and not batched:
        t0 = time.time()
        set_state("pushing", {"pct": 0, "note": "rsync 增量推图"})
        if sh(link.ssh("mkdir -p %s" % shlex.quote(remote_images))) != 0:
            raise SystemExit("ssh 建目录失败")
        # 只传图片扩展名：标注/labels/说明文件不占板端带宽与存储
        args_r = ["-az", "--info=progress2", "--partial", "--timeout=180",
                  "--include=*/"]
        args_r += ["--include=*.%s" % e for e in IMG_EXTS]
        args_r += ["--exclude=*",
                   shlex.quote(spec["images"]) + "/",
                   shlex.quote("%s:%s/" % (args.board, remote_images))]
        if sh_retry(link.rsync(args_r), what="rsync 推图",
                    on_line=push_progress) != 0:
            raise SystemExit("rsync 推图失败")
        set_state("pushing", {"pct": 100, "note": "推图完成"})
        print("[run_eval] 推图完成 %.1fs" % (time.time() - t0))

    # ---- 3. 模型上传（board: 前缀 = 已在板上；分批模式缓存避免重复 scp）----
    models = [(os.path.splitext(os.path.basename(args.model))[0], args.model)]
    for c in args.compare:
        tag, _, m = c.partition("=")
        models.append((tag or "cmp", m))
    _remote_model_cache = {}

    def resolve_model(m):
        if m.startswith("board:"):
            return m[6:]
        if m in _remote_model_cache:
            return _remote_model_cache[m]
        local = os.path.abspath(m)
        if not os.path.exists(local):
            raise SystemExit("模型不存在: %s" % m)
        remote = posixpath.join(remote_job, "models", os.path.basename(m))
        if sh(link.ssh("mkdir -p %s" % shlex.quote(posixpath.dirname(remote)))) != 0:
            raise SystemExit("ssh 失败")
        if sh_retry(link.scp(local, "%s:%s" % (args.board, remote)),
                    what="scp 模型 %s" % m) != 0:
            raise SystemExit("scp 模型失败: %s" % m)
        _remote_model_cache[m] = remote
        return remote

    # ---- 4+5. 板端推理 + 拉 dump（整集：逐模型跑完统一拉；分批：批内推-跑-拉-删）----
    def infer_and_pull(tag, rm, images_dir, out_sub, pull_tag, mi=0):
        """单个模型在 images_dir 上推理，dump/metrics 拉回本地并命名为 pull_tag。"""
        set_state("running", {"model": tag, "model_idx": mi + 1, "models": len(models)})
        cmd_args = ["--task", args.task, "--images", images_dir,
                    "--conf", str(args.conf), "--threads", str(args.threads),
                    "--dump-only", "--out-dir", out_sub,
                    "--model", rm, "--name", tag]
        if args.preview:
            cmd_args += ["--preview", "--preview-port", str(args.preview_port)]
        if args.obj_num:
            cmd_args += ["--obj-num", str(args.obj_num)]
        if args.label:
            # ssh 非交互 shell 的 cwd 不是项目目录，相对路径必须转绝对
            cmd_args += ["--label", os.path.abspath(args.label)]
        # ssh 非交互 shell 的 cwd 是家目录，rknn_eval 的 label 等默认相对路径
        # 会失效——固定 cd 到板端仓库目录再执行
        full = "cd %s && %s %s" % (shlex.quote(args.remote_cwd), args.remote_bin,
                                   " ".join(shlex.quote(a) for a in cmd_args))
        if sh(link.ssh(full),
              stream_log=os.path.join(out_dir, "remote", "infer_%s.log" % pull_tag)) != 0:
            raise SystemExit("板端推理失败: %s（详见 remote/infer_%s.log）" % (tag, pull_tag))
        set_state("pulling", {"model": tag})
        rc = sh_retry(link.scp("%s:%s" % (args.board,
                                          posixpath.join(out_sub, "dump_%s.jsonl" % tag)),
                               out_dir + "/")
                      + " && " +
                      link.scp("%s:%s" % (args.board,
                                          posixpath.join(out_sub, "metrics_%s.json" % tag)),
                               out_dir + "/"),
                      what="拉取 dump %s" % tag)
        if rc != 0:
            raise SystemExit("拉取 dump 失败: %s" % tag)
        # 重命名（分批场景保留批号）
        if pull_tag != tag:
            for ext in (".jsonl", ".metrics.json"):
                src = os.path.join(out_dir, "dump_%s.jsonl" % tag) if ext == ".jsonl" \
                    else os.path.join(out_dir, "metrics_%s.json" % tag)
                dst = os.path.join(out_dir, "dump_%s%s" % (pull_tag, ".jsonl")) if ext == ".jsonl" \
                    else os.path.join(out_dir, "metrics_%s.json" % pull_tag)
                if os.path.exists(src):
                    os.replace(src, dst)

    if batched:
        for bi in range(n_batches):
            push_batch(bi)
            print("\n[run_eval] ===== 批 %d/%d（%d 张）=====" %
                  (bi + 1, n_batches, min(args.push_batch, len(img_files) - bi * args.push_batch)))
            for mi, (tag, m) in enumerate(models):
                rm = resolve_model(m)
                infer_and_pull(tag, rm, remote_images,
                               posixpath.join(remote_job, "batch_%d" % bi, "out"),
                               "%s_b%02d" % (tag, bi), mi)
            if not args.keep_remote:
                sh(link.ssh("rm -rf %s %s" %
                            (shlex.quote(remote_images),
                             shlex.quote(posixpath.join(remote_job, "batch_%d" % bi)))))
        # 合并各批 dump / 汇总 metrics（FPS=Σ帧/Σ耗时，含每批模型加载开销）
        for mi, (tag, m) in enumerate(models):
            parts = sorted(glob.glob(os.path.join(out_dir, "dump_%s_b*.jsonl" % tag)))
            with open(os.path.join(out_dir, "dump_%s.jsonl" % tag), "w") as w:
                for p in parts:
                    w.write(open(p).read())
            frames = elapsed = 0.0
            for p in sorted(glob.glob(os.path.join(out_dir, "metrics_%s_b*.json" % tag))):
                try:
                    mjs = json.load(open(p))["rk_pipe"]["pipeline"]
                    frames += mjs.get("processed_frames", 0)
                    elapsed += mjs.get("elapsed_seconds", 0)
                except (OSError, ValueError, KeyError):
                    pass
            if frames and elapsed:
                agg = {"rk_pipe": {"pipeline": {
                    "avg_fps": frames / elapsed, "processed_frames": int(frames),
                    "elapsed_seconds": elapsed},
                    "note": "分批模式汇总（含每批模型加载开销，FPS 略保守）"}}
                json.dump(agg, open(os.path.join(out_dir, "metrics_%s.json" % tag), "w"), indent=1)
    else:
        for i, (tag, m) in enumerate(models):
            rm = resolve_model(m)
            print("\n[run_eval] ===== 板端推理 %d/%d: %s =====" % (i + 1, len(models), tag))
            infer_and_pull(tag, rm, remote_images, posixpath.join(remote_job, "out"), tag, i)
        print("[run_eval] dump/metrics 已拉回", out_dir)

    # ---- 6. 清理板端（保模型目录外的工作文件）----
    if not args.keep_remote:
        sh(link.ssh("rm -rf %s" % shlex.quote(remote_job)))

    # ---- 7. PC 端评测 + 报告（复用本地 rknn_eval --reuse-dump）----
    set_state("evaluating", {"note": "PC 端评测 + 报告渲染"})
    eval_bin = os.environ.get("RK_EVAL_BIN", "build/rknn_eval")
    if not os.path.exists(eval_bin):
        eval_bin = DEFAULT_REMOTE_BIN  # 允许在 boards.yaml 覆盖本机跑（降级路径）
    cmp_args = []
    for tag, m in models[1:]:
        cmp_args += ["--compare", "%s=dump:%s" % (tag,
                     os.path.join(out_dir, "dump_%s.jsonl" % tag))]
    cmd = [eval_bin, "--task", args.task,
           "--conf", str(args.conf),
           "--reuse-dump", os.path.join(out_dir, "dump_%s.jsonl" % models[0][0]),
           "--name", models[0][0], "--out-dir", out_dir] + cmp_args
    if is_obb:
        if not gt_jsonl:
            raise SystemExit("obb 评测需要 GT：spec.format=yolo-obb 自动转换，"
                             "或 --gt 指向切片级 jsonl（大图请用 split_dota.py）")
        cmd += ["--gt", gt_jsonl]
    else:
        cmd += ["--ann", ann]
    if args.label:
        cmd += ["--label", args.label]
    if args.obj_num:
        cmd += ["--obj-num", str(args.obj_num)]
    print("\n[run_eval] ===== PC 端评测 =====")
    rc = sh(" ".join(shlex.quote(c) for c in cmd))
    set_state("done", {"out_dir": out_dir, "note": "报告 %s/report.md" % out_dir})
    print("\n[run_eval] 完成：报告 %s/report.md" % out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
