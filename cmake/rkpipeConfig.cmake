# rkpipe CMake 包配置: find_package(rkpipe) 支持
#
# 暴露:
#   rkpipe::core   IMPORTED 静态库(闭源调度核心 + C ABI 门面)
#   RKPIPE_CORE_ARCHIVE  核心库归档路径
#
# 模块层为源码形态(本仓库 src/),消费方自行添加子目录或源文件;
# 典型用法见 examples/rkpipe_cli。

set(RKPIPE_CORE_ARCHIVE "${CMAKE_CURRENT_LIST_DIR}/../../prebuilt/${CMAKE_SYSTEM_PROCESSOR}/librkpipe_core.a"
    CACHE FILEPATH "rkpipe 闭源核心库")

if(EXISTS "${RKPIPE_CORE_ARCHIVE}")
    if(NOT TARGET rkpipe::core)
        add_library(rkpipe::core STATIC IMPORTED)
        set_target_properties(rkpipe::core PROPERTIES
            IMPORTED_LOCATION "${RKPIPE_CORE_ARCHIVE}"
            INTERFACE_COMPILE_OPTIONS "-fvisibility=hidden"
        )
    endif()
endif()
