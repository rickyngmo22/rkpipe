# FAQ

**Q: 为什么调度核心不开源?这是"假开源"吗?**
A: 采用 open-core:功能模块(输入输出、模型封装、后处理、跟踪、事件规则、工具链)100% 开源,调度核心(线程编排、队列、内存池、零拷贝推理引擎)以预编译库分发。这不是"壳开源":本仓库的模块代码是完整可审计、可单独复用的,单测也在本仓库运行。你完全可以只用本仓库的模块层(比如 postprocess + io)搭建自己的推理程序,核心库不是必需品。

**Q: librkpipe_core.a 会失效吗?ABI 怎么保证?**
A: 核心与模块层的契约 = `include/rkpipe/` + `include/core/` 开源头。这些头是稳定契约,变更走语义化版本;`examples/rkpipe_cli` 只用 C ABI,它在 CI 中链接最新核心库编译运行,即 ABI 回归测试。核心库按 Release 版本分发,与本仓库 tag 一一对应。

**Q: 可以只链接核心库、不用本仓库模块吗?**
A: 不建议也不支持:核心依赖模块层的符号(config/io/postprocess/detection),静态链接时由链接组解析,两者是配套发布的。

**Q: Windows / x86 支持?**
A: 核心库目前只发布 aarch64(RK3588)。模块层代码本身可在任意 Linux x86 上编译(`RK_PIPE_CI_BUILD=ON` 即为其验证路径),欢迎 PR 补齐平台适配。

**Q: 许可是什么?**
A: 源码 Apache-2.0(含专利授权条款);预编译核心库为专有许可(见 LEGAL/RKPIPE-CORE-EULA.md),允许免费使用与集成,禁止再分发库文件本身。第三方组件许可见 NOTICE。

**Q: Web 预览安全吗?**
A: 预览是 LAN 内明文 HTTP(MJPEG/WebRTC),无鉴权——默认绑定 `0.0.0.0`,任何同网段设备都能打开。仅限可信内网使用,不要端口转发到公网;生产部署建议置于带鉴权的反代之后。

**Q: 为什么没有模型权重?**
A: 权重涉及各自上游许可(AGPL/商用条款等),仓库只提供转换工具链与探针,从公开权重自行转换,见 docs/build.md。

**Q: Web 预览怎么开?**
A: 配置 `web_preview: 1` + `web_preview_port`,浏览器访问 `http://<板子IP>:<port>`;WebRTC 播放 URL 见 `web_preview_webrtc_urls`。示例见 `examples/configs/detect_web_preview.yaml`。
