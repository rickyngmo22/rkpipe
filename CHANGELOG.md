# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/),版本遵循语义化版本。

## [未发布]

### 移除
- 语义分割(sem)与单目 3D 检测(detect3d)任务：契约类型/检测器/后处理/绘制/评测载荷与单测；后续版本再随需求恢复
- 多任务同路组合（Y5 aux）与依赖它的 depth-dist 测距标注/近距告警：`AppConfig` 的 `aux_model_path`/`aux_task`/`depth_dist_*` 字段、`--aux-model`/`--aux-task`/`--depth-dist-*` 命令行、`PipelineFrame` 的 `auxDepth*` 四字段、JSONL/RESULT 的 `aux` 附加载荷、`utils/depth_distance.h` 纯函数与对应单测；配套核心库已同步重编
- 二阶段任务与级联:rtmpose 两阶段姿态、ocr_det / ocr_rec、二级分类(composite_cls / cls_top1 / ctc_decode / 透视矫正)全部源码、单测与文档,OCR 字典资产与转换脚本
- `events_demo` 示例与 `tools/probes/` 模型输出探针及 `RK_PIPE_BUILD_PROBES` 构建选项
- 边界自检脚本 `scripts/check_open_boundary.py`、发布同步脚本 `scripts/publish_public.sh` 与对应 CI job

### 新增
- OBB 评测指南 `docs/benchmark_dota_obb.md`；**PC 端一键预处理** `tools/eval/prepare_dota.bat/.py`（解压官方包→整理 val_images/labelTxt→本机切片出 patches+gt，上传量约为原图一半；`--limit N` 支持小样试跑）；网页端 DOTA 走 PC 预处理产物（patches + gt jsonl 经文件夹同步上传），实测 3 图 zip → 10 切片 → yolo26n_obb mAP50 65.5 全流程通过(DOTA v1.0 数据准备/自动切片/已切片复用/INT8 vs FP16 对比/网页控制台操作/评测口径)
- **网页评测控制台**（`tools/eval/eval_web_service.sh start`，PC 浏览器直连 `http://<板IP>:8081`，零依赖纯标准库）:数据集上传（zip 或 PC 文件夹增量同步,二次评测只传差异）→ 模型下拉（`model/models.json` 登记）/上传 → 提交评测（同板串行队列,多模型对比,实时日志 + MJPEG 抽帧快照）→ 自动出 mAP 报告（MD/CSV/逐类表）→ 历史任务还原与跨任务对比。**存储安全**：同步/上传前空间预检（不足即拒绝），默认任务成功后用完即删上传缓存（可勾选保留走增量复评），页面显示板端剩余空间，缓存根目录 `RK_EVAL_DATA` 可指到大容量分区。**提交页交互修正与标注放开**：标注文件不再限制扩展名/命名（自定义数据集任意命名均可，按原名上板）；修复网页提交把已选图片文件整体序列化进请求体导致的 400（改为字段白名单）；核心默认 label 路径改为 `assets/labels/`（随公库布局）。任务切换即时过滤模型下拉（页面加载即按任务过滤）；提交失败时弹窗透出服务端原因、HTML 页面禁缓存防旧脚本（`Cache-Control: no-store`）；修复网页提交 400 的根因——`FormData` 走 multipart 而后端按 urlencoded 解析导致全部字段丢失（改用 `URLSearchParams` urlencoded 提交）；修复补传驱动 `join` 的转义错误导致的整页 JS 失效（该错误会让任务过滤/提交拦截/分批上传全部失灵，现已用 esprima 全量语法校验页面脚本）；对比下拉不再被任务切换强制改选（选回"不对比"状态可保持）；任务切换即时过滤模型下拉（页面加载即按任务过滤）；对比模型默认"不对比"时添加按钮禁用（选模型才启用，选回"不对比"自动清掉已添加行）；选完图片文件夹只做清单比对不立刻上传，点「开始评测」时才上传（并发 4，进度可见）。**数据集分批增量上传**：文件夹同步模式按「每批张数」分批——先传 2 批（1 运行 + 1 候选），任务逐批消费（脚本按 `batch_state` 进度 + `.ready` 哨兵等待），板上跑完一批由页面自动补传下一批，板上任意时刻最多 2 批，评测全程无需一次性上传大数据集；**补传不再依赖页面存活**：提交页统一引入补传模块并自动请求屏幕唤醒锁（上传期间防 PC/浏览器休眠）；数据集为**一次性全量上传**（选文件夹 → 清单比对 → 4 并发全量上传 → 提交，进度实时可见；比对请求 30s 超时与失败有明确提示，提交时未完成比对会自动补跑），数据保留在板上缓存（同文件夹再次提交只补差异，任务结束自动清理，可勾选保留）；「每批张数」输入框随分批机制一并移除；修复 iframe 初始化竞态（脚本未就绪即移交导致补传静默丢失、任务卡在等下一批——现重试至驱动就绪才跳转）；修复补传驱动的两个致命问题——iframe 方案在整页跳转时随页面上下文被卸载导致补传停止（改为提交页自身驱动 + 任务页开新标签查看，评测完成自动跳转）；以及批次表误传对象数组导致 `S.batches[idx]` 取空、批 2 永远不打 `.ready`（改按批号索引稀疏数组）——误传 `[{idx,names}]` 对象数组而驱动按下标 `S.batches[idx]` 取文件列表，导致 `S.batches[2]` 恒为 undefined、批 2 永远不打 `.ready`、任务卡死在批间隙（现改为按批号索引的稀疏数组）；每批跑完即删，任务结束清空缓存（默认）。配套 `run_eval.py`(PC ssh/rsync 脚本化远程评测,断点续传+状态机)与 `convert_dataset.py`(YOLO/VOC/labelme/yolo-obb → COCO/DOTA 转换+校验+GT 预览);`rknn_eval` 新增 `--dump-only/--preview/--preview-port/--vis-sample`

### 变更
- 网页数据集上传简化为一次性全量上传（不分批/不断点，选文件夹 → 全量上传 → 提交，4 并发进度可见；同文件夹再次提交按清单只补差异，任务结束自动清理缓存可勾选保留）；移除分批消费/断点续跑/补传驱动的相关机制与界面:`model/` 附带四个 YOLO 版本(yolo26n / yolov8n / yolo11n / yolov5s)的板端模型,覆盖 detect / pose / obb / seg / depth,示例配置开箱即用;来源与许可见 NOTICE
- 二进制收敛为两个:`console_detector`(板端控制台应用,由原 CLI 改名,经 C ABI 驱动完整流水线)与 `rknn_eval`(精度/性能评测)
- 仓库作为独立项目发布:文档不再区分源码层与核心库的分发形态;任务范围收敛到单阶段(detect / pose / obb / seg / depth)
- 修复读帧限速假顶:预编译核心的热降档在未启用(未设 `RK_PIPE_THERMAL_MAX_C`)时,governor 仍按默认 `target_fps=30` 回填读帧 cap,流水线被硬性节流在 ~30fps(与温度/负载/线程数无关);修复后默认满速,600 帧视频源 + yolov8n + 6 worker 实测 158 FPS(修复前 31 FPS),1 worker 53 FPS。需要热降档时显式设置 `RK_PIPE_THERMAL_MAX_C` 即恢复原行为
- 预编译核心库同步重发布(`librkpipe_core.a`,sha256 见 `prebuilt/aarch64/sha256sums.txt`):核心与契约头按精简后的单阶段任务集一一对齐,无占位兼容层。核心内不再包含 rtmpose/OCR/二级分类实现;事件规则链路保留(`RK_PIPE_EVENT_RULES` 命名多规则 + 扁平键唤醒,板端 1800/1800 帧评估验证),新增 `tracking_runtime` 各接口的可选 `engine_tracked` 出参供核心回传本帧跟踪结果(默认 nullptr,源码兼容)

### 修复
- 发布阻断:`.gitignore` 的本地模型目录规则(`model/`)误遮蔽 `include/model/` 与 `src/model/`——9 个契约头(yolov8/yolov26/yolov5/rtmpose/ocr 等)与 5 个模型层实现从未被 git 跟踪,全新 clone 的完整构建在 `detection.cc` 处必失败(CI 只覆盖纯逻辑模式与缺库报错路径,未暴露);本地数据目录规则全部锚定仓库根(`/model/` 等),两处源码目录已纳入跟踪,并以"仅 git 跟踪文件导出"的方式完成完整构建+流水线运行验证
- 多路编排器 `--no-restart` 收尾:全部子进程退出后编排器此前仍在前台空转,现打印"全部流已退出"并以退出码 0 结束(有限输入批量跑的正确收尾;板端双流实测)
- 预处理缓冲跨分配器释放:detection 预处理的 malloc 回退指针此前会被交给线程本地内存池释放(未定义行为);现以 priv_data 哨兵标记分配来源,统一经 `releasePreprocessBuffer` 释放,并补上转换失败路径的泄漏
- `read_lines_from_file` 长行(>1023 字符)导致的堆越界写:改为跨段累积逻辑行 + 数组按需扩容;`read_data_from_file` 补 fseek/ftell/malloc 检查
- Web 预览服务器生命周期:客户端线程从 detach 改为登记 + join(stop 返回后不再有线程引用服务对象,消除 use-after-free);fd 关闭权收敛消除双重 close;accept 改 poll 轮询;客户端 socket 增加收发超时,防慢速连接占满连接槽
- CMake:预编译核心库检查移到所有依赖之前(缺库时首先给出明确指引);补 librknnrt/libchromaprint/rknn_eval 的 install 与 `$ORIGIN/../lib` RPATH;修复安装态 rkpipeConfig.cmake 的归档路径,补 Version 文件与 include/线程依赖;`cmake_minimum_required` 提升至 3.20
- 文档与实际不符:build.md 的转换脚本用法(脚本无命令行参数)、tools/eval/README 的失效路径与虚指文件、faq 中不存在的 CI ABI 回归描述、eval 脚本中的陈旧工具名

### 变更
- 范围收敛到当前核心的可用边界:移除命名规则示例配置(告警/webhook 派发在当前核心缺失,引擎本身保留,激活方式见 `include/core/event_engine.h`);板端本地演示配置不入库。多任务/级联相关文档保留——`docs/multitask.md` 是已交付级联模块(OCR rec / det→cls / RetinaFace 解码)的权威说明
- 内容去本机化:探针(`tools/probes/`)与评测示例配置(`tools/eval/example_detect.conf`)的默认路径从板端部署目录的绝对路径改为仓库相对路径(`model/`、`test_images/`、`assets/labels/`);边界自检新增"本机绝对路径"检查(部署目录与 `/home/<user>` 字面量,扫描扩展补 `.conf`/`.cmake`);`.gitignore` 补数据集/评测输出目录(`data/`、`coco_eval/`、`eval_runs/`)与板端本地演示配置(`/demo_*.yaml`)
- 后处理去重:letterbox 逆映射(`map_box_to_frame`/`map_point_to_frame`,7 处拷贝收敛为 1 份,yolov8/yolov26 关键点的钳位差异作为显式参数保留)、yolo26 张量类型判定与元素读取(`Y26TensorType`/`y26_tensor_at`/`y26_tensor_type_from_attr`,4 份拷贝收敛)、logit 域阈值换算(`y26_conf_to_logit_threshold`,3 份收敛);yolov8_obb 删除与 postprocess_common 逐字重复的排序/sigmoid/量化助手及死代码 CalculateOverlap
- 边界:核心实现的声明(rtmpose/ocr_det/RtmposeDetector/postprocess 部分函数/detect3d_decode)统一加 `[closed-core]` 标注;边界自检新增"契约头声明的符号必须在核心库中有定义"的 nm 同步校验与 sha256 归档校验(公库无归档时优雅跳过)
- CI:actions/checkout 以 commit SHA 固定;unit-tests 增加 gcc+clang 矩阵;新增 concurrency 组取消旧运行
- 测试:tracker/检测器工厂用例此前不属于任何构建目标(死测试),现完整构建也编译测试并按用例注册 ctest(单用例失败不再吞掉其余结果);测试临时文件按 PID 区分
- 边界自检:扫描范围从 5 个目录扩大到 git 全部跟踪文本文件,新增 sha256 归档校验与私有文件跟踪检查
- 移除 web 预览/输出路由中的 debug-point 调试脚手架(约 300 行);`rga_nv12_disabled` 改原子
- tools/probes 探针修正扩展名(.c → .cc,实为 C++),纳入可选构建(`RK_PIPE_BUILD_PROBES=ON`)并补 README;源文件统一 `.cc` 扩展名
- 新增 .clang-format;CI 的 full-build 负向测试不再依赖 runner 预装包

### 新增
- 运行指南 `docs/run_guide.md`:边界自检→纯逻辑单测→板端构建→素材准备→检测流水线→Web 预览→事件/JSONL→模型探针→评测 dry-run→多路编排的 10 步逐项验收清单,每步给出命令与预期结果(全部命令已在 RK3588 板端实测通过)
- 多路部署编排器 `scripts/multistream.py`(纯 stdlib):一路一进程形态的官方看护工具——按流轮转分配 `npu_core_start`(用户显式配置的流保持不动,派生 YAML 写入 `--log-dir`)、预检(二进制/配置存在、web_preview_port 与 output_video_path 跨流重复即拒绝)、每流独立日志、崩溃自动重启(退避可调)、SIGTERM/SIGINT 优雅停止(先 TERM 后 KILL);`--dry-run` 只做预检打印分配
- Web 预览访问令牌鉴权:环境变量 `RK_PIPE_WEB_TOKEN=<令牌>` 非空时,除 `/healthz`(存活探针)外所有路径要求令牌(查询参数 `?token=` 或 `X-Auth-Token` 头,常时比较);预览页/LLM 页内端点自动携带令牌(令牌值经 URL 安全白名单净化防注入),多窗格跨端口访问同样生效。Config 为核心按布局填充的类型不可加字段,经环境变量下发(未配置时行为同旧版)。socket 级单测(401/200/healthz/页面嵌入)+ 板端真实流水线 curl 验证
- 事件规则引擎扩展为命名多规则底座(自私有仓库移植适配,纯逻辑 CI 可测):8 类规则(绊线/入侵/滞留/离岗/聚集/遗留物/跌倒/超速),规则实例携带 id/类别过滤/方向合规(A2B/B2A 逆行过滤)/绊线判定点(center|bottom,行人脚底惯例)/各类去抖阈值;经环境变量 `RK_PIPE_EVENT_RULES=<yaml>` 启用(规则 YAML 自行编写,字段语义见 `include/core/event_engine.h` 头注释),命名规则存在时扁平 event_* 键不再合成、旧部署行为逐帧一致。布局红线适配:EventEngine/RuleEvent 为核心按布局消费的类型,不可改成员——规则状态放进程内侧表、规则身份经 RuleEvent.detail 的 "[id] " 前缀携带、新增 stats()/takeStats()/ruleHint() 为纯增量非虚方法;event_rules 配置键与 RuleEvent.rule_id 字段随核心库 Release 协同(docs/multitask.md §4)
- detect→裁剪→分类级联(M0)模块与辅助任务数据化:二级分类解码 `postprocess/cls_top1.h`(argmax+softmax 双模式)、分类模型封装 `model/composite_cls.h`(拉伸 resize+自反量化+行和≈1 自动判别概率域);级联探针 `tools/probes/det_cls_probe`(板端验证:bus.jpg 公交框→ImageNet cls 654 minibus);结果 JSONL 增加 `aux` 附加载荷——配置 aux_task=depth 时逐帧携带辅助深度元数据(roi/range,板端 1800/1800 帧),是当前核心下首个可数据消费的多任务通道(栅格不外发,通用辅助结果位随核心协同,见 docs/multitask.md §4)
- OCR det→rec 两阶级联模块(M6/M7 自私有仓库移植,板端已验收):CTC 贪心解码(`postprocess/ctc_decode.h`:blank 折叠/字典映射/[B,T,C] 与 [B,C,T] 布局自动识别/概率域与 raw logits 双模式)、rec 模型封装(`model/ocr_rec.h`:letterbox 高对齐补黑、UINT8 直喂、自反量化、softmax 概率自动检测)、透视矫正纯函数(`postprocess/ocr_rec_geometry.h`:质心外扩+透视矫正,CI 27 项断言);字典资产 `assets/dict/`(PP-OCR 6623 行 / LPRNet 68 行,来源见 NOTICE)与转换脚本 `convert_ppocr_rec.py`;级联探针 `tools/probes/ocr_det_rec_probe`(det 经 Detector 工厂驱动推理→矫正→rec→CTC,`--json` 输出 schema v1 "type":"ocr" 行)
- 多任务组合设计文档 `docs/multitask.md`:现状盘点(辅助任务渲染 only)、OCR 级联设计与板端验证方法、下个核心 Release 的布局耦合协同同步清单(OCRDetectTaskResult.lines / CompositeCls+Face 变体 / ocr_rec_* 配置键 / 通用辅助结果位 / RESULT 事件下发)、同帧 N 任务并行契约草案
- M8 RetinaFace 人脸检测解码核心(`postprocess/retinaface_decode.h`):先验框生成(320×320 → 4200 anchors,strides 8/16/32,枚举顺序逐项对照 zoo `rknn_box_priors.h`;公式化生成,640×640 → 16800 亦可)、center-form 框解码(variances [0.1,0.2])、5 点 landmark 解码(variance 0.1)、conf 过滤 + 分数降序截断 + IoU NMS。纯函数无硬件依赖,进 CI 纯逻辑源集并新增 29 项单测(先验框基准值/零回归闭合/方差数学/NMS 行为/病态输入防护)
- 逐帧结构化结果 schema v1(`core/task_result_json.h` 纯函数序列化器 + docs/event_payload.md 权威文档):信封(frame/width/height/source/type)+ 各任务载荷(detect/pose/obb/seg/depth/sem/ocr_det/ocr/detect3d);seg 掩膜按框内裁剪行主序 RLE 编码(与 eval_coco.py 解码端闭合);bbox 统一 `[x,y,w,h]` 原帧像素,与 dump_detections 口径一致;非有限浮点钳为 0 保证 JSON 合法。9 类变体 + RLE + 逃逸共 52 项断言进 CI 纯逻辑单测
- 逐帧结果 JSONL 文件汇(`io/result_sink.h`,环境变量 `RK_PIPE_RESULT_JSONL=<文件路径>`):pipeline 模式逐帧落盘,与 `RK_PIPE_EVENT_RESULT` 事件 payload 同格式——当前核心库即可用的逐帧结构化结果通道。经核心启动缝 `shouldEnableOutput` 初始化、逐帧缝 `finalizePipelineFrameOutput` 写出(下个核心 Release 起由核心直接初始化,接线点退役);当前仅 pipeline 模式生效。实现过程中发现 AppConfig 与核心按成员布局直接耦合,公开仓库不可单独加配置字段(加字段后板端实跑段错误复现并定位),故过渡期配置走环境变量,YAML 键 `result_jsonl_path` 待配套核心协同启用

### 说明
- 板端实证(gdb 断点):公开核心 0.2.0 逐帧调用 EventEngine::update(扁平与命名模式均 1800/1800 帧)但 AlertRuntime::maybeEventAlert 零调用——规则事件→告警/webhook 的派发在当前核心缺失,命名规则与扁平规则的告警激活同随该核心缺口解决;当前可用的命名规则能力为叠加可视化(drawOverlay)与统计/提示接口(stats/takeStats/ruleHint)。已录入 docs/multitask.md §4 协同清单
- 契约头注释级变更:`rkpipe.h` 的 `RK_PIPE_EVENT_RESULT` 注释补充 payload schema 指引(事件本身仍未触发,ABI 无变化;触发需配套核心库 Release,门面逐帧调用序列化器 `buildFrameResultJson`)
- face 任务的流水线接线(任务类型/绘制/示例配置)依赖调度核心的 Face 结果分发支持,待配套新核心随 Release 发布后启用

## [0.2.0] — 首次公开发布

### 新增
- 基础任务:检测(detect)、姿态(pose)、旋转框(obb)、实例分割(seg)、单目深度(depth)
- 模型家族:YOLO26 / YOLOv8 / YOLOv5(detect 与 seg),YOLOv8-Pose / YOLO26-Pose,YOLOv8-OBB,YOLO26-Depth
- 稳定 C ABI(`rkpipe/rkpipe.h`):create / start / stop / wait / status / 事件回调,"配置即契约"
- 输入输出:本地文件 / 图片目录 / RTSP / v4l2 输入;文件 / RTSP 推流 / Web 预览(MJPEG/WebRTC)输出
- 跟踪与事件规则引擎:ID 跟踪、绊线 / 区域入侵 / 滞留 / 离岗(纯逻辑,CI 可测)
- 工具链:YOLO 系列 ONNX→RKNN 转换脚本、模型输出探针、相机标定
- RGA 零拷贝预处理路径(NV12 dma-buf)+ CPU 回退
- 预编译调度核心 `librkpipe_core.a`(aarch64/RK3588,随 Release 分发)

### 说明
- 调度核心以预编译库分发,功能模块为本仓库源码(Apache-2.0)
- 无硬件环境可用 `RK_PIPE_CI_BUILD=ON` 运行纯逻辑单元测试
