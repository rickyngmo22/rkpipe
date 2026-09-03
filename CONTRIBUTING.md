# 贡献指南

感谢关注 rkpipe!本仓库采用 open-core 模式,提交 PR 前请了解边界:

## 可以贡献的部分

- **模块层**:`src/{io,model,postprocess,detection,config,utils}` —— 新模型后处理、新输入源、跟踪/事件规则改进、性能优化;
- **契约层**:契约头(`include/rkpipe/`、`include/core/` 开源子集)的演进提案,请先开 issue 讨论再动代码(契约变更影响 ABI 兼容);
- **工具链**:`convert_*.py`、`tools/`;
- **文档与示例**。

## 不接受(也无法接受)的部分

- 调度核心(`librkpipe_core.a`)的实现不在本仓库,没有对应源码 PR 的落点。

## 提交前自检

```bash
# 1. 纯逻辑单测(无硬件也能跑)
cmake -B build-ci -DRK_PIPE_CI_BUILD=ON && cmake --build build-ci -j && ctest --test-dir build-ci

# 2. 开源边界自检(禁止引用闭源符号/引入敏感内容)
python3 scripts/check_open_boundary.py
```

## 代码约定

- 与既有代码风格保持一致(17 行内 include 分组:系统 → 第三方 → 项目);
- 公共头文件注释面向使用者,说明契约与线程语义,不写实现细节;
- 新后处理实现请同时补 CI 可跑的纯函数用例(tests/test_unit.cpp)。

## 许可

提交即表示同意以 Apache-2.0 授权你的贡献(见 LICENSE)。
