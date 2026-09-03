cmake_minimum_required(VERSION 3.10)
project(rkpipe-open VERSION 0.2.0 LANGUAGES C CXX)

# ===========================================================================
# rk_pipe 开源仓库构建
#
# 本仓库包含: 契约头(rkpipe/) + 功能模块(io/model/postprocess/detection/
# config/utils) + 示例 + 工具链脚本。
# 调度核心(app_runtime/processing_pipeline/mempool/threadpool/...) 不在本仓库,
# 以预编译静态库形式提供: prebuilt/<arch>/librkpipe_core.a。
#
# 构建模式:
#   1. 完整构建(默认): 需要预编译核心库 + 硬件依赖(RKNN/RGA/MPP/FFmpeg/OpenCV)
#      cmake -B build && cmake --build build
#   2. CI 构建: 无硬件依赖的纯逻辑单元测试(配置解析/事件规则/ROI 过滤/后处理)
#      cmake -B build-ci -DRK_PIPE_CI_BUILD=ON && cmake --build build-ci
# ===========================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

option(RK_PIPE_CI_BUILD "CI-only build: pure unit tests without RKNN/RGA/MPP/FFmpeg deps" OFF)

# ---- OpenCV -----------------------------------------------------------------
set(OPENCV_ROOT "" CACHE PATH "OpenCV installation prefix (with lib/ and include/opencv4)")
if(NOT OPENCV_ROOT AND DEFINED ENV{OPENCV_ROOT})
    set(OPENCV_ROOT "$ENV{OPENCV_ROOT}")
endif()

# ---- turbojpeg(Web 预览 JPEG 编码加速) --------------------------------------
find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h)
find_library(TURBOJPEG_LIB NAMES turbojpeg)
if(NOT TURBOJPEG_INCLUDE_DIR OR NOT TURBOJPEG_LIB)
    message(FATAL_ERROR "turbojpeg not found: apt install libturbojpeg0-dev (or build from source)")
endif()

if(NOT RK_PIPE_CI_BUILD)
    # ---- 硬件相关依赖(仅完整构建需要) ---------------------------------------
    find_path(RGA_INCLUDE_DIR rga/im2d.h)
    find_library(RGA_LIB rga)
    if(NOT RGA_INCLUDE_DIR OR NOT RGA_LIB)
        message(FATAL_ERROR "rga not found: please install system rga headers and library")
    endif()

    # ---- 预编译调度核心(闭源,见 README "预编译核心库" 一节) ------------------
    set(RK_PIPE_ARCH_DEFAULT ${CMAKE_SYSTEM_PROCESSOR})
    set(RK_PIPE_ARCH "${RK_PIPE_ARCH_DEFAULT}" CACHE STRING "Architecture dir under prebuilt/")
    set(RKPIPE_CORE_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${RK_PIPE_ARCH}/librkpipe_core.a"
        CACHE FILEPATH "Prebuilt closed-core static library (core scheduling engine + C ABI facade)")
    if(NOT EXISTS "${RKPIPE_CORE_ARCHIVE}")
        message(FATAL_ERROR
            "预编译核心库不存在: ${RKPIPE_CORE_ARCHIVE}\n"
            "调度核心不开源,以静态库随 Release 分发。请从本仓库的 Releases 页下载对应版本的\n"
            "librkpipe_core.a 放入 prebuilt/${RK_PIPE_ARCH}/,或改用 CI 构建:\n"
            "  cmake -B build-ci -DRK_PIPE_CI_BUILD=ON")
    endif()
endif()

find_package(Threads REQUIRED)

# ===========================================================================
# CI 模式: 纯逻辑单元测试,不依赖 RKNN/RGA/MPP/FFmpeg
# ===========================================================================
if(RK_PIPE_CI_BUILD)
    set(CI_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/config
        ${CMAKE_CURRENT_SOURCE_DIR}/include/core
        ${CMAKE_CURRENT_SOURCE_DIR}/include/io
        ${CMAKE_CURRENT_SOURCE_DIR}/include/model
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/include/postprocess
        ${CMAKE_CURRENT_SOURCE_DIR}/include/detection
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn
        ${TURBOJPEG_INCLUDE_DIR}
    )
    if(OPENCV_ROOT)
        list(APPEND CI_INCLUDE_DIRS ${OPENCV_ROOT}/include/opencv4)
    else()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc)
        list(APPEND CI_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    endif()
    include_directories(${CI_INCLUDE_DIRS})

    set(CI_PURE_SOURCES
        tests/test_unit.cpp
        src/config/app_config.cc
        src/core/performance.cpp
        src/core/detection_filter.cc
        src/core/event_engine.cc
        src/postprocess/postprocess_common.cc
        src/postprocess/retinaface_decode.cc
    )
    add_executable(rk_pipe_unit_tests ${CI_PURE_SOURCES})
    target_compile_definitions(rk_pipe_unit_tests PRIVATE RK_PIPE_CI=1)
    if(OPENCV_ROOT)
        target_link_libraries(rk_pipe_unit_tests Threads::Threads
            ${OPENCV_ROOT}/lib/libopencv_core.so ${OPENCV_ROOT}/lib/libopencv_imgproc.so
            ${TURBOJPEG_LIB})
    else()
        target_link_libraries(rk_pipe_unit_tests Threads::Threads ${OpenCV_LIBS} ${TURBOJPEG_LIB})
    endif()
    enable_testing()
    add_test(NAME rk_pipe_unit_tests COMMAND rk_pipe_unit_tests)
    return()
endif()

# ===========================================================================
# 完整构建
# ===========================================================================
if(NOT OPENCV_ROOT)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)
    set(OPENCV_LIBS ${OpenCV_LIBS})
    set(OPENCV_INC ${OpenCV_INCLUDE_DIRS})
else()
    set(OPENCV_INC ${OPENCV_ROOT}/include/opencv4)
    # 显式完整路径,避免依赖全局 link_directories
    set(OPENCV_LIB_DIR ${OPENCV_ROOT}/lib)
    set(OPENCV_LIBS
        ${OPENCV_LIB_DIR}/libopencv_core.so
        ${OPENCV_LIB_DIR}/libopencv_imgproc.so
        ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
        ${OPENCV_LIB_DIR}/libopencv_highgui.so
        ${OPENCV_LIB_DIR}/libopencv_videoio.so
    )
endif()

set(RKNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn)

# 模块层源码(开源): 输入输出 / 模型封装 / 后处理 / 跟踪 / 配置 / 工具
file(GLOB_RECURSE RKPIPE_MODULE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compat/*.cc
)
list(APPEND RKPIPE_MODULE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/performance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/detection_filter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/event_engine.cc
)

# FFmpeg 的 chromaprint 封装桩: 必须是独立命名 .so(libavformat 运行期按名打开)
add_library(chromaprint_compat SHARED src/compat/chromaprint_stub.cc)
set_target_properties(chromaprint_compat PROPERTIES
    OUTPUT_NAME chromaprint
    VERSION 1.0.0
    SOVERSION 1
)

# 模块层静态库
add_library(rkpipe_modules STATIC ${RKPIPE_MODULE_SOURCES})
target_include_directories(rkpipe_modules PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/config
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/io
    ${CMAKE_CURRENT_SOURCE_DIR}/include/model
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detection
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn
    ${RGA_INCLUDE_DIR}
    ${TURBOJPEG_INCLUDE_DIR}
    ${OPENCV_INC}
)
target_compile_options(rkpipe_modules PRIVATE -fvisibility=hidden)
set_target_properties(rkpipe_modules PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 预编译核心与模块层互相引用符号,用链接组避免静态库归档顺序问题
add_library(rkpipe_core_prebuilt STATIC IMPORTED GLOBAL)
set_target_properties(rkpipe_core_prebuilt PROPERTIES
    IMPORTED_LOCATION "${RKPIPE_CORE_ARCHIVE}"
    POSITION_INDEPENDENT_CODE ON
)

set(RKPIPE_LINK_GROUP "-Wl,--start-group" rkpipe_modules rkpipe_core_prebuilt "-Wl,--end-group")
set(RKPIPE_LINK_DEPS
    Threads::Threads
    ${OPENCV_LIBS}
    ${RKNN_DIR}/librknnrt.so
    ${RGA_LIB}
    ${TURBOJPEG_LIB}
    rockchip_mpp
    chromaprint_compat
    avformat avcodec avutil swscale
)

# CLI: 通过 C ABI(rkpipe.h)驱动完整流水线,同时是 ABI 的验收测试
add_executable(rkpipe_cli examples/rkpipe_cli/rkpipe_cli.cc)
target_include_directories(rkpipe_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rkpipe_cli PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
set_target_properties(rkpipe_cli PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
target_link_options(rkpipe_cli PRIVATE "-Wl,--disable-new-dtags")

# 最小事件回调示例(纯 C)
add_executable(events_demo examples/events_demo/events_demo.c)
target_include_directories(events_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(events_demo PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
set_target_properties(events_demo PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
target_link_options(events_demo PRIVATE "-Wl,--disable-new-dtags")

# 板端模型精度/性能一键评测(经 C ABI 驱动流水线;评测脚本与数据准备见 tools/eval/README.md)
add_executable(rknn_eval tools/eval/rknn_eval_main.cc)
target_include_directories(rknn_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(rknn_eval PRIVATE RK_EVAL_SCRIPTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/eval")
target_link_libraries(rknn_eval PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
set_target_properties(rknn_eval PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
target_link_options(rknn_eval PRIVATE "-Wl,--disable-new-dtags")

# 边界自检(手动/CI 运行): 确认开源代码未引用闭源符号与敏感内容
add_custom_target(check_boundary
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_open_boundary.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Check open-source boundary (no closed-core references / no secrets)"
)

# ---- 安装 --------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS rkpipe_cli events_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/rkpipe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/core DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rkpipe-modules)
install(FILES ${RKPIPE_CORE_ARCHIVE} DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES cmake/rkpipeConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rkpipe)
