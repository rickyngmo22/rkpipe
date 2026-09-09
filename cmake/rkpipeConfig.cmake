# rkpipe CMake 包配置: find_package(rkpipe) 支持
#
# 暴露:
#   rkpipe::core   IMPORTED 静态库(调度核心 + C ABI 门面)
#   RKPIPE_CORE_ARCHIVE  核心库归档路径
#
# 模块层为源码形态(本仓库 src/),消费方自行添加子目录或源文件;
# 典型用法见 apps/console_detector。
#
# 兼容两种布局:
#   安装态:  <prefix>/lib/cmake/rkpipe/ → 归档在 <prefix>/lib/librkpipe_core.a,
#            契约头在 <prefix>/include/rkpipe 与 <prefix>/include/rkpipe-modules
#   源码树态: <repo>/cmake/ → 归档在 <repo>/prebuilt/<arch>/librkpipe_core.a

set(_rkpipe_installed_archive "${CMAKE_CURRENT_LIST_DIR}/../../librkpipe_core.a")
set(_rkpipe_source_archive "${CMAKE_CURRENT_LIST_DIR}/../prebuilt/${CMAKE_SYSTEM_PROCESSOR}/librkpipe_core.a")

if(EXISTS "${_rkpipe_installed_archive}")
    set(_rkpipe_archive "${_rkpipe_installed_archive}")
    set(_rkpipe_include_dir "${CMAKE_CURRENT_LIST_DIR}/../../../include")
elseif(EXISTS "${_rkpipe_source_archive}")
    set(_rkpipe_archive "${_rkpipe_source_archive}")
    set(_rkpipe_include_dir "${CMAKE_CURRENT_LIST_DIR}/../include")
else()
    message(WARNING
        "rkpipe: 未找到预编译核心库(尝试过 ${_rkpipe_installed_archive} 与 ${_rkpipe_source_archive})。\n"
        "请从 Releases 下载 librkpipe_core.a 并设置 RKPIPE_CORE_ARCHIVE。")
    set(_rkpipe_archive "${_rkpipe_source_archive}")
    set(_rkpipe_include_dir "")
endif()

set(RKPIPE_CORE_ARCHIVE "${_rkpipe_archive}"
    CACHE FILEPATH "rkpipe 闭源核心库")

if(NOT TARGET rkpipe::core)
    add_library(rkpipe::core STATIC IMPORTED)
    set_target_properties(rkpipe::core PROPERTIES
        IMPORTED_LOCATION "${RKPIPE_CORE_ARCHIVE}"
        INTERFACE_COMPILE_OPTIONS "-fvisibility=hidden"
    )
    if(_rkpipe_include_dir)
        set_target_properties(rkpipe::core PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_rkpipe_include_dir}")
    endif()
    find_package(Threads QUIET)
    if(TARGET Threads::Threads)
        set_target_properties(rkpipe::core PROPERTIES
            INTERFACE_LINK_LIBRARIES Threads::Threads)
    endif()
endif()
