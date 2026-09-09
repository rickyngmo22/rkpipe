# FAQ

**Q: 调度核心为什么是预编译库?**
A: 调度核心(线程编排、队列、内存池、零拷贝推理引擎)以预编译静态库 `librkpipe_core.a` 随 Release 分发;功能模块(输入输出、模型封装、后处理、跟踪、事件规则、工具链)均为本仓库源码,完整可审计、可单独复用。你完全可以只用本仓库的模块层(比如 postprocess + io)搭建自己的推理程序,核心库不是必需品。

**Q: librkpipe_core.a 会失效吗?ABI 怎么保证?**
A: 核心与模块层的契约 = `include/rkpipe/` + `include/core/` 契约头。这些头是稳定契约,变更走语义化版本;核心库按 Release 版本分发,与本仓库 tag 一一对应。集成验收时以 `apps/console_detector`(仅用 C ABI)为编译级参照:它能通过 C ABI 完整驱动流水线,即表明核心库与契约头配套。

**Q: 可以只链接核心库、不用本仓库模块吗?**
A: 不建议也不支持:核心依赖模块层的符号(config/io/postprocess/detection),静态链接时由链接组解析,两者是配套发布的。

**Q: Windows / x86 支持?**
A: 核心库目前只发布 aarch64(RK3588)。模块层代码本身可在任意 Linux x86 上编译(`RK_PIPE_CI_BUILD=ON` 即为其验证路径),欢迎 PR 补齐平台适配。

**Q: 许可是什么?**
A: 源码 Apache-2.0(含专利授权条款);预编译核心库许可见 LEGAL/。第三方组件许可见 NOTICE。

**Q: Web 预览安全吗?**
A: 预览是 LAN 内明文 HTTP(MJPEG/WebRTC)。可设置环境变量 `RK_PIPE_WEB_TOKEN=<令牌>` 启用访问令牌鉴权:除 `/healthz` 外所有路径要求令牌(查询参数 `?token=` 或 `X-Auth-Token` 头),页面内端点自动携带令牌,WebRTC 网关(MediaMTX 等)需自行鉴权。未配置令牌时行为同旧版(无鉴权)。无论是否启用,建议 `web_preview_bind` 绑定内网地址、不要端口转发到公网;生产部署建议置于带鉴权的反代之后。

**Q: model/ 里的模型是怎么来的?**
A: 由公开权重(公共 YOLO 检查点)经 `tools/convert/` 转换为 RKNN 后随仓库分发,便于开箱即用;各模型的版权归上游权重许可所有(如 Ultralytics 的 AGPL/商用条款),商用部署前请自行确认。自己换模型/量化版本的流程见 docs/build.md。

**Q: Web 预览怎么开/关?**
A: 示例配置默认已开启(`web_preview: 1`),浏览器访问 `http://<板子IP>:<port>`(`web_preview_port`,默认 8080);设 `web_preview: 0` 关闭。WebRTC 播放 URL 见 `web_preview_webrtc_urls`(通常配合 mediamtx 等 WebRTC 网关使用)。
