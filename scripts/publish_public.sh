#!/usr/bin/env bash
# publish_public.sh — 从内部私库导出开源仓库并发布。
#
# 私库(全量备份,含闭源 .a)与本公库的差别仅一处:
#   prebuilt/**/*.a 不进公库 git(按 README 约定随 Release 分发)。
# 其余文件(含 sha256sums.txt、LEGAL/EULA、CI)原样同步。
#
# 用法:
#   scripts/publish_public.sh                 # 导出+本地提交,不推送
#   scripts/publish_public.sh --push          # 导出+提交+推送到公库
#   scripts/publish_public.sh --remote <url>  # 首次使用:指定公库远程地址
#
# 环境变量: RKPIPE_PUBLIC_DIR 覆盖公库本地目录(默认 ../rkpipe-public)

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PUB_DIR="${RKPIPE_PUBLIC_DIR:-$(dirname "$ROOT")/rkpipe-public}"
PUSH=0
REMOTE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --push)   PUSH=1 ;;
        --remote) REMOTE="$2"; shift ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
    shift
done

# 1. 发布前自检: 边界检查必须通过,失败禁止发布
echo "==> 运行开源边界自检..."
python3 "$ROOT/scripts/check_open_boundary.py"

for f in LICENSE NOTICE README.md; do
    [ -f "$ROOT/$f" ] || { echo "缺少 $f,拒绝发布"; exit 1; }
done

# 2. 导出 git 追踪的文件到临时目录,剔除闭源二进制
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$STAGE"
find "$STAGE/prebuilt" -name "*.a" -delete 2>/dev/null || true

# 3. 同步到公库工作区(--delete 保证与私库导出完全一致)
mkdir -p "$PUB_DIR"
rsync -a --delete --exclude=.git "$STAGE"/ "$PUB_DIR"/

# 4. 公库本地仓库初始化(仅首次)
if [ ! -d "$PUB_DIR/.git" ]; then
    git -C "$PUB_DIR" init -q -b main
fi
if [ -n "$REMOTE" ]; then
    git -C "$PUB_DIR" remote remove public 2>/dev/null || true
    git -C "$PUB_DIR" remote add public "$REMOTE"
fi
if [ -z "$(git -C "$PUB_DIR" remote get-url public 2>/dev/null || true)" ]; then
    echo "!! 公库远程未设置,仅完成本地导出。"
    echo "   首次发布: scripts/publish_public.sh --remote git@github.com:<用户>/<仓库名>.git --push"
    exit 0
fi

# 5. 提交并推送(公库历史独立于私库,全新干净历史)
git -C "$PUB_DIR" pull --rebase public main 2>/dev/null || true
git -C "$PUB_DIR" add -A
if ! git -C "$PUB_DIR" diff --cached --quiet; then
    git -C "$PUB_DIR" commit -q -m "publish: sync $(date +%F)"
else
    echo "==> 无变化,无需新提交"
fi

if [ "$PUSH" -eq 1 ]; then
    git -C "$PUB_DIR" push -u public main
    echo "==> 已推送到公库: $(git -C "$PUB_DIR" remote get-url public)"
    echo "==> 提醒: 若核心库有更新,请把 librkpipe_core.a 上传到公库对应 Release,并核对 prebuilt/*/sha256sums.txt"
else
    echo "==> 本地导出完成($PUB_DIR),未推送。确认后加 --push 重跑。"
fi
