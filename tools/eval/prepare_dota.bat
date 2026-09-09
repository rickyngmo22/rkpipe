@echo off
rem ============================================================
rem  prepare_dota.bat — DOTA v1.0 数据集一键预处理（Windows）
rem
rem  用法:  双击后按提示拖入压缩包; 或命令行:
rem         prepare_dota.bat <DOTA zip 或目录> [--limit N]
rem
rem  输出:  当前目录 dota_ready/（val_images + val_labelTxt +
rem         若本机有 python+cv2: patches + patches_gt.jsonl）
rem
rem  需要:  Python 3（可选 opencv-python + numpy，用于本机切片；
rem         没有也能整理目录，切片交给网页 zip 上传在板端做）
rem ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0\..\.."          rem 仓库根

rem ---- 找 python ----
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% --version >nul 2>nul || (
  echo [错误] 未找到 Python。请安装 Python 3 并勾选 Add to PATH：
  echo   https://www.python.org/downloads/
  echo 或仅整理目录后改用网页「上传数据集 zip」让板端自动切片。
  pause
  exit /b 2
)

rem ---- 取输入参数 ----
set INPUT=%~1
if "%INPUT%"=="" (
  echo 请把 DOTA 压缩包(.zip) 或已解压目录 拖到这个窗口后回车：
  set /p INPUT=
)
if "%INPUT%"=="" (
  echo [错误] 未提供输入。用法: prepare_dota.bat ^<zip或目录^>
  pause
  exit /b 2
)
if not exist "%INPUT%" (
  echo [错误] 输入不存在: %INPUT%
  pause
  exit /b 2
)

echo ==== DOTA 预处理: %INPUT%
%PY% tools\eval\prepare_dota.py --input "%INPUT%" %2 %3 %4 %5
if errorlevel 1 (
  echo [失败] 见上方错误信息
  pause
  exit /b 1
)
echo.
echo ==== 完成。按网页评测指南(docs/benchmark_dota_obb.md)第 3 节上传即可。
pause
