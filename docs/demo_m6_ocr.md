# Demo M6：OCR 文字识别（ocr_det 检测 + rec 识别二级级联）

> 2026-09-03 **板端验收通过**（模型交付即日闭环）。合成图 "HELLO 123" / "RK3588 2026"
> 两行文本正确检出并识别，双构建单测全绿（板端 1223 / CI 252）。
> 调试开关：`RK_PIPE_DEBUG_OCR_DET=1`（概率图统计 / 检出框数 / 逐框识别文本）。

## 架构

```
帧 ──> ocr_det 检测（stage1，480×480，输出概率图已 Sigmoid）
   ──> DB 后处理：阈值 0.3 → 膨胀 → 轮廓 → minAreaRect 四点多边形 + 分数
   ──> 逐多边形透视矫正（→ 正置矩形）
   ──> ocr_rec 推理（stage2，PP-OCRv4 Rec @3x48x320，宽保持比例右补黑）
   ──> CTC 贪心解码（blank 折叠 + 字典映射 + 分数概率/logits 自动识别）
   ──> OCRTextLine{text, score} 叠加绘制 / webhook
```

新增模块：
- [ctc_decode](../include/postprocess/ctc_decode.h)：CTC 贪心解码纯函数（M7 LPRNet 复用）+
  `resolveRecLayout` 输出布局识别（[B,T,C] vs [B,C,T] 按字典尺寸自动判别转置）
- [ocr_rec](../include/model/ocr_rec.h)：rec 模型封装（裁剪归一化 + 自管推理 + 多类型反量化）
- OCRDetectDetector 覆载 detect：检测后自动跑 stage2（未配 rec 行为不变）

## 配置（configs/run_ocr.yaml）

```yaml
task: "ocr_det"
model_path: "model/ppocrv4_det.rknn"                # stage1 文字检测（480×480 自动适配）
ocr_rec_model_path: "model/ppocrv4_rec_fp16.rknn"   # stage2 识别（不配=纯检测）
ocr_dict_path: "model/ppocr_keys_v1.txt"            # 6623 行；0=blank、末位=空格
ocr_rec_blank_index: 0
```

## 模型（已入位 model/，交付存档 model/board_model/）

| 文件 | 说明 |
|---|---|
| `ppocrv4_det.rknn` | 文字检测 480×480；概率图输出（图内已 Sigmoid，后处理不再做） |
| `ppocrv4_rec_fp16.rknn` | 文字识别 320×48；输出 [1,40,6625] **图内已 Softmax**——分数=概率，板端按"首时间步行和≈1"自动识别，无需配置 |
| `ppocr_keys_v1.txt` | 字典 6623 行（类映射：1..N → 字典第 i 行，越界=空格） |
| `lprnet_fp16.rknn` + `lprnet_dict.txt` | M7 车牌 rec：**[B,C,T] 布局自动转置**、blank=67（末位 '-'），同一管道换三行配置即用（已板端验证布局/字典映射） |

## 板端验收记录（2026-09-03）

- 合成图两行文本：`HELLO 123 → HEL1O 12.`、`RK3588 2026 → RK.3588 2026`
  （L→1 / 3→. 为 cv::putText Hershey 描边字体的正常水平，印刷体素材会更好）；
- det 概率图值域正确 [0, 0.998]；1080p 视频 det+rec 链路 ~30 FPS（源帧率封顶）；
- **验收发现并修复板端自身 bug**：非量化（FP16）模型输出经 `want_float=1` 已被
  runtime 转为 FP32，但 ocr_det 后处理按 attr.type=FLOAT16 读 → ±512 垃圾值。
  已修：非量化模型一律按 FP32 读 buffer（教训：buffer 实际类型以 runtime 转换行为为准）。

## 已知限制与遗留

- **RGB/BGR 通道序**：det/rec/mobilenet 源模型期望 RGB，板端喂 UINT8 BGR——文字任务
  黑白场景影响小，真实彩色素材观察后如有精度问题，交付方可 3 分钟重转 BGR-mean 版
  （mean 顺序反转即可，见交付说明）；
- 叠加绘制：cv::putText 不支持 CJK，画面上非 ASCII 字符以 `?` 占位
  （识别文本本身完整进 webhook / 事件链路）；
- 长文本行超过模型宽 320 时右侧挤压（PP-OCR 推理侧固有约束）；
- rec 级联在 NV12 零拷贝路径会整帧转 BGR（RGA 硬件加速）；
- **LPRNet 端到端待车牌检测源**（COCO/OBB 无 plate 类）。
