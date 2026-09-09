#!/usr/bin/env python3
# 多路部署编排器:一路一进程的官方编排工具(当前核心的多路形态,见 docs/ROADMAP.md #9)。
#
#   python3 scripts/multistream.py run --config a.yaml b.yaml [--binary ./build/console_detector]
#                 [--log-dir ./multistream_logs] [--cores auto|0,1,2]
#                 [--restart-delay 3] [--no-restart] [--dry-run]
#
# 职责:
#   - 按流轮转分配 npu_core_start(用户在 config 里显式配置的流保持不动),派生派生 YAML;
#   - 预检:二进制存在、config 存在、web_preview_port / output_video_path 跨流重复即报错;
#   - 每流独立日志 + 周期状态行;崩溃自动重启(退避 --restart-delay 秒);
#   - SIGTERM/SIGINT 优雅停止全部子进程(先 TERM 后 KILL)。
#
# 环境变量(RK_PIPE_EVENT_RULES / RK_PIPE_WEB_TOKEN / RK_PIPE_RESULT_JSONL 等)原样透传给
# 全部子进程;需要按流区分的(如 RESULT_JSONL)请拆成多次调用或用 per-config 路径。

import argparse
import os
import re
import signal
import subprocess
import sys
import time

TERM_GRACE_SECONDS = 5.0


def log(msg: str) -> None:
    sys.stdout.write(f"[multistream] {msg}\n")
    sys.stdout.flush()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def scalar_match(text: str, key: str):
    m = re.search(rf"^\s*{key}\s*:\s*(\S+)\s*$", text, re.M)
    return m


def strip_quotes(v: str) -> str:
    return v.strip().strip('"').strip("'")


def preflight(binary: str, configs, cores_arg: str) -> list:
    """返回每流的 npu_core_start 分配(已显式配置的为 None=保持不动)。"""
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        log(f"错误: 二进制不存在或不可执行: {binary}")
        sys.exit(1)

    ports = {}
    outputs = {}
    assignments = []
    for idx, cfg in enumerate(configs):
        if not os.path.isfile(cfg):
            log(f"错误: 配置不存在: {cfg}")
            sys.exit(1)
        text = read_text(cfg)

        port_m = scalar_match(text, "web_preview_port")
        if port_m:
            port = int(port_m.group(1))
            if port > 0:
                if port in ports:
                    log(f"错误: web_preview_port={port} 在流 {ports[port]} 与流 {idx}({cfg}) 重复,"
                        "请为每路分配独立端口")
                    sys.exit(1)
                ports[port] = idx

        out_m = scalar_match(text, "output_video_path")
        if out_m:
            out = strip_quotes(out_m.group(1))
            if out and not out.startswith("udp://"):
                if out in outputs:
                    log(f"错误: output_video_path={out} 在流 {outputs[out]} 与流 {idx}({cfg}) 重复")
                    sys.exit(1)
                outputs[out] = idx

        existing = scalar_match(text, "npu_core_start")
        assignments.append(int(existing.group(1)) if existing else None)

    # core 分配:显式配置的流保持;其余按 cores 列表轮转填充空位
    if cores_arg == "auto":
        cores = [0, 1, 2]
    else:
        cores = [int(v) for v in cores_arg.split(",") if v != ""]
        if not cores or any(c < 0 or c > 2 for c in cores):
            log(f"错误: --cores 非法: {cores_arg}(应为 auto 或 0..2 的组合,如 0,1,2)")
            sys.exit(1)
    rotation = iter(cores * (len(configs) // max(1, len(cores)) + 1))
    for idx, assigned in enumerate(assignments):
        if assigned is None:
            assignments[idx] = next(rotation)
    return assignments


def write_derived_config(base_path: str, core: int, work_dir: str, idx: int) -> str:
    """派生 YAML:写入 npu_core_start(已有则改值,无则追加到末尾)。"""
    text = read_text(base_path)
    line = f"npu_core_start: {core}"
    m = scalar_match(text, "npu_core_start")
    if m:
        derived = text[: m.start()] + line + text[m.end():]
    else:
        derived = text.rstrip("\n") + f"\n{line}\n"
    os.makedirs(work_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    derived_path = os.path.join(work_dir, f"stream_{idx}_{base_name}.yaml")
    with open(derived_path, "w", encoding="utf-8") as f:
        f.write(derived)
    return derived_path


def main() -> None:
    parser = argparse.ArgumentParser(description="rkpipe 多路部署编排器(一路一进程)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="前台守护:启动全部流并看护")
    run.add_argument("--config", nargs="+", required=True, help="每路一个 YAML 配置(至少一路)")
    run.add_argument("--binary", default="./build/console_detector", help="rkpipe 可执行文件路径")
    run.add_argument("--log-dir", default="./multistream_logs", help="日志与派生配置目录")
    run.add_argument("--cores", default="auto", help="npu_core_start 分配: auto(0,1,2 轮转) 或如 0,1,2")
    run.add_argument("--restart-delay", type=float, default=3.0, help="崩溃重启退避秒数")
    run.add_argument("--no-restart", action="store_true", help="子进程退出后不重启(只跑一轮)")
    run.add_argument("--dry-run", action="store_true", help="只做预检并打印分配,不启动")
    args = parser.parse_args()

    configs = args.config
    if not configs:
        log("错误: 至少需要一个 --config")
        sys.exit(1)
    assignments = preflight(args.binary, configs, args.cores)

    os.makedirs(args.log_dir, exist_ok=True)
    streams = []
    for idx, (cfg, core) in enumerate(zip(configs, assignments)):
        derived = write_derived_config(cfg, core, args.log_dir, idx)
        log_path = os.path.join(
            args.log_dir,
            f"stream_{idx}_{os.path.splitext(os.path.basename(cfg))[0]}.log")
        streams.append({
            "idx": idx, "config": cfg, "derived": derived, "log": log_path,
            "core": core, "proc": None, "restarts": 0, "started_at": None,
        })

    log(f"预检通过: {len(streams)} 路, 二进制 {args.binary}")
    for s in streams:
        log(f"  流 {s['idx']}: npu_core_start={s['core']} config={s['config']} -> {s['derived']}")
    if args.dry_run:
        return

    stop_flag = {"stop": False}

    def request_stop(signum, frame):
        if not stop_flag["stop"]:
            log(f"收到信号 {signum},开始优雅停止全部流...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    child_env = dict(os.environ)
    child_env["RK_PIPE_MS_STREAM"] = "1"  # 标记由编排器托管(调试用)

    def spawn(stream) -> None:
        logf = open(stream["log"], "ab")
        stream["proc"] = subprocess.Popen(
            [args.binary, stream["derived"]], stdout=logf, stderr=subprocess.STDOUT,
            env=child_env, cwd=os.getcwd())
        stream["started_at"] = time.time()
        logf.close()
        log(f"流 {stream['idx']} 启动 pid={stream['proc'].pid} (core {stream['core']})")

    for s in streams:
        spawn(s)

    while not stop_flag["stop"]:
        time.sleep(0.5)
        now = time.time()
        for s in streams:
            proc = s["proc"]
            if proc is None:
                continue
            code = proc.poll()
            if code is None:
                continue
            uptime = now - s["started_at"] if s["started_at"] else 0.0
            log(f"流 {s['idx']} 退出 code={code} (本次运行 {uptime:.0f}s, 累计重启 {s['restarts']})")
            s["proc"] = None
            if args.no_restart or stop_flag["stop"]:
                continue
            # 退避:刚启动即退的流多等一轮,避免崩溃风暴刷日志
            delay = args.restart_delay * (2.0 if uptime < args.restart_delay else 1.0)
            s["restart_at"] = now + delay
            s["pending_delay"] = delay
        # --no-restart 且全部流已退出:编排使命完成,正常结束
        if args.no_restart and all(s["proc"] is None for s in streams):
            log("全部流已退出(--no-restart),编排器结束")
            break
        # 到期重启
        for s in streams:
            if s["proc"] is None and not args.no_restart and "restart_at" in s:
                if now >= s["restart_at"]:
                    log(f"流 {s['idx']} 重启(退避 {s['pending_delay']:.0f}s 后)")
                    del s["restart_at"]
                    s["restarts"] += 1
                    spawn(s)
        # 周期状态行
        if int(now) % 30 == 0:
            states = ", ".join(
                f"s{s['idx']}={'运行' if s['proc'] and s['proc'].poll() is None else '停止'}"
                f"(重启{s['restarts']})" for s in streams)
            log(f"状态: {states}")
            time.sleep(0.2)  # 同一秒内不重复打印

    # 优雅停止:TERM → 等 grace → KILL
    log("停止全部流...")
    for s in streams:
        if s["proc"] and s["proc"].poll() is None:
            s["proc"].terminate()
    deadline = time.time() + TERM_GRACE_SECONDS
    for s in streams:
        if s["proc"] and s["proc"].poll() is None:
            remaining = max(0.0, deadline - time.time())
            try:
                s["proc"].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"流 {s['idx']} 未在 {TERM_GRACE_SECONDS:.0f}s 内退出,KILL")
                s["proc"].kill()
    log("已全部停止")


if __name__ == "__main__":
    main()
