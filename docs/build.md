# 构建与模型转换

## 环境依赖(板端 RK3588,Debian/Ubuntu)

| 依赖 | 用途 | 安装 |
|---|---|---|
| OpenCV ≥ 4.5 | 图像处理/视频 IO | 系统包或 `-DOPENCV_ROOT=<prefix>` 指定自编前缀 |
| libturbojpeg | Web 预览 JPEG 编码 | `apt install libturbojpeg0-dev` |
| RGA (librga) | 零拷贝格式转换/缩放 | 系统自带 `rga` 头与库 |
| rockchip-mpp | 视频编解码 | 系统自带 |
| FFmpeg (libav*) | 视频封装推流 | `apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev` |
| RKNN Runtime | NPU 推理 | 本仓库 `3rdparty/rknn/`(librknnrt.so,遵循 Rockchip 许可) |

```bash
cmake -B build -DOPENCV_ROOT=/path/to/opencv-install   # 可省略,默认用系统 OpenCV
cmake --build build -j $(nproc)
```

## 无硬件 CI 构建

```bash
cmake -B build-ci -DRK_PIPE_CI_BUILD=ON
cmake --build build-ci -j $(nproc)
ctest --test-dir build-ci --output-on-failure
```

纯逻辑单测覆盖:YAML/命令行配置解析、事件规则引擎(绊线/入侵/滞留/离岗几何与状态机)、ROI 过滤、后处理纯函数(NMS/解码)、性能统计。

## 模型准备

仓库不分发模型权重。以公开权重为起点:

1. 用 Ultralytics / 官方仓库导出 ONNX(检测头含解码或裸输出均可,见各 `convert_yolo26_*.py` 的注释);
2. 按任务运行根目录转换脚本,例如:
   - `python3 convert_yolo26_fp16.py --onnx yolo26n.onnx --out model/yolo26n.rknn`
   - 量化版本参考 `convert_yolo26_cls_sigmoid_int8.py`(cls 输出 Sigmoid 收窄值域,保住 INT8 分数)
3. 把得到的 `.rknn` 与 labels 文本路径填入示例配置。

各脚本的 `--help` 与头部注释为权威说明;`tools/probes/` 下的探针程序用于在板上核对单模型输出,排查量化掉点问题时很好用。

## 示例素材

仓库自带类别表(`assets/labels/`),示例配置可直接引用。还差两样东西:

**测试视频**(任选其一,零下载方式):

```bash
mkdir -p video
# ffmpeg 生成 30 秒测试图(流水线跑通用;想看真实检测效果请用自带素材/摄像头)
ffmpeg -f lavfi -i testsrc2=duration=30:size=1280x720:rate=30 video/demo.mp4
```

也可以直接用摄像头(`input_path: "/dev/video0"`)、RTSP 地址或任意本地 mp4。

**模型**:按上一节从公开权重转换,产物放到 `model/` 目录(与示例配置中的相对路径对应)。

## 模型精度/性能评测(rknn_eval)

`build/rknn_eval` 一键完成板端评测:配置校验 → C ABI 驱动流水线导出结果 dump → Python 评测(pycocotools / 旋转 IoU)→ 论文风格报告(MD + LaTeX + CSV)→ 可选 CI 断言。支持 detect / pose / seg / obb 四任务,INT8 vs FP16 自动对比与量化损失计算:

```bash
build/rknn_eval --task detect \
  --model model/yolo26n.rknn --compare fp16=model/yolo26n_fp16.rknn \
  --images data/coco/val2017 --ann data/coco/annotations/instances_val2017.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 \
  --threads 4 --out-dir eval_runs/detect --assert-ap50 40
```

常用选项:`--threads-sweep`(线程扫描测速)、`--conf-sweep`(选部署阈值)、`--reuse-dump`(跳过推理重评)、`--vis-dir`(渲染检测结果图)。数据集需自行准备 COCO val2017 或 DOTA v1.0;完整命令与数据准备见 [tools/eval/README.md](../tools/eval/README.md)。

## 消费本仓库(CMake)

```cmake
find_package(rkpipe)          # 暴露 rkpipe::core(预编译核心)
add_executable(myapp main.cc)
target_include_directories(myapp PRIVATE <本仓库>/include)
target_link_libraries(myapp PRIVATE
    "-Wl,--start-group" <本仓库源码编译出的模块库> rkpipe::core "-Wl,--end-group"
    ${OpenCV_LIBS} rknnrt rga turbojpeg rockchip_mpp avformat avcodec avutil swscale)
```

直接构建本仓库的 `rkpipe_cli` / `events_demo` 并抄 `examples/*/CMake` 片段是更快的起步方式。
