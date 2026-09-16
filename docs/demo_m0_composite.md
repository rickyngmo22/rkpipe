# Demo M0/M5：两阶级联（composite_cls）——检测 + 二级分类

> 2026-09-03 板端验收通过。M0 CompositeDetector 首个任务形态落地：
> 一级任意 YOLO 检测 + 二级 MobileNetV2 分类，每框 crop → top-1 叠加。
> smoke 8/8 全过（composite-cls 43.4 FPS / 5400 帧）。

## 架构

```
帧 ──> stage1 检测（组合现有 Detector 工厂，自动路由 YOLO v5/v8/11/26 后处理）
   ──> 逐框 crop（BGR，≥8×8 有效）──> 拉伸 resize 到 cls 输入（224×224）
   ──> cls 推理（raw logits / 已 softmax 自动判别）→ argmax top-1 + softmax 分数
   ──> CompositeClsTaskResult{检测列表 + 逐框 cls_ids/scores/labels}
```

- 二级模型/标签懒加载，未配 `cls_model_path` 时自动退化（validate 拦截）；
- 板端验收实测：车框（COCO car）→ `moving van 62%`——ImageNet 细粒度分类语义正确；
- 调试：`RK_PIPE_DEBUG_COMPOSITE=1` 打印每框二级结果。

## 配置（configs/run_composite_cls.yaml）

```yaml
task: "composite_cls"
model_path: "model/yolo26n.rknn"                    # 一级检测（任意 YOLO 家族）
cls_model_path: "model/mobilenetv2_fp16.rknn"       # 二级分类
cls_labels_path: "model/mobilenet_labels.txt"       # 可选，每行一个类名
```

## 扩展语义（换 cls_model_path 即换二级含义）

| 二级模型 | 场景 |
|---|---|
| MobileNetV2（ImageNet） | 车型细分（car→sedan/SUV/truck）、商品识别 |
| 自训属性分类（人 crop） | 行人属性（性别/背包/雨伞） |
| 自训缺陷分类（工件 crop） | 工业质检 |

## 当前边界

- v1 走 BGR 路径（级联需要整帧 BGR crop；NV12 经 RGA 硬转），零拷贝框绘制未做
  （与 OCR 级联同状态，记 fallback 统计）；
- tracking/事件引擎未接入 composite_cls（v1 定位展示型 demo；检测框仍来自一级 YOLO）；
- RtmposeDetector 的级联增强（NPU 核拆分/帧预算 top-K/stage2 线程池）未迁移到本类——
  cls 推理极快（~几 ms）暂无必要，密集场景需要时再迁移（见路线图 §6.1）；
- RGB/BGR：MobileNetV2 源模型期望 RGB（见 model/board_model/交付说明.md），同 OCR 遗留项。
