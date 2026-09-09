#!/bin/bash
# eval_web_service.sh —— 板端评测控制台服务（零依赖，随仓库跑）
#
# 板端开启服务，PC 浏览器直接操作（无需 PC 端 ssh/rsync 环境）：
#   ./tools/eval/eval_web_service.sh start     # 启动（默认 :8081，PORT=9090 可改端口）
#   ./tools/eval/eval_web_service.sh status
#   ./tools/eval/eval_web_service.sh stop
#   ./tools/eval/eval_web_service.sh restart
#
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8081}"
PID_FILE="/tmp/rknn_eval_web_$PORT.pid"
LOG_FILE="$REPO/web_runs/eval_web_$PORT.log"

is_running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-start}" in
  start)
    if is_running; then
      echo "已在运行 (pid $(cat "$PID_FILE"), :$PORT)"
      exit 0
    fi
    mkdir -p "$REPO/web_runs"
    cd "$REPO" || exit 1
    nohup python3 tools/eval/rknn_eval_web.py --port "$PORT" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    if is_running; then
      IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
      echo "已启动 (pid $(cat "$PID_FILE"))"
      echo "  PC 浏览器打开: http://${IP:-<板IP>}:${PORT}"
      echo "  日志: $LOG_FILE"
    else
      echo "启动失败，最后几行日志："; tail -5 "$LOG_FILE"; exit 1
    fi
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "已停止 (:${PORT})"
    else
      rm -f "$PID_FILE"; echo "未运行"
    fi
    ;;
  status)
    if is_running; then
      echo "运行中 (pid $(cat "$PID_FILE"), :$PORT)"
    else
      echo "未运行"; exit 1
    fi
    ;;
  restart)
    "$0" stop; sleep 1; "$0" start
    ;;
  *)
    echo "用法: $0 {start|stop|status|restart}   （端口: PORT=9090 $0 start）"
    exit 1
    ;;
esac
