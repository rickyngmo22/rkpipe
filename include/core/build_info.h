#pragma once

// 构建信息（Y7 版本化）：CMake 构建期注入 git commit（短哈希），
// 随 metrics JSON / /status.json / daemon summary 输出，故障回溯与
// 模型-代码版本对账的基础。非 git 构建（tarball）回退 "unknown"。

#ifndef RK_PIPE_GIT_COMMIT
#define RK_PIPE_GIT_COMMIT "unknown"
#endif

inline const char* buildGitCommit() {
    return RK_PIPE_GIT_COMMIT;
}
