cmake_minimum_required(VERSION 3.20)
project(rkpipe-open VERSION 0.2.0 LANGUAGES C CXX)

# ===========================================================================
# rkpipe 构建
#
# 本仓库包含: 契约头(rkpipe/) + 功能模块(io/model/postprocess/detection/
# config/utils) + 应用(console_detector) + 评测工具(rknn_eval)。
# 调度核心(app_runtime/processing_pipeline/mempool/threadpool/...) 以预编译
# 静态库形式提供: prebuilt/<arch>/librkpipe_core.a。
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
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

add_compile_options(-Wall)
option(RK_PIPE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (-Werror)" OFF)
if(RK_PIPE_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
endif()

option(RK_PIPE_CI_BUILD "CI-only build: pure unit tests without RKNN/RGA/MPP/FFmpeg deps" OFF)

# ---- 预编译调度核心(见 README "预编译核心库" 一节) --------------------------
# 该检查必须位于所有第三方依赖之前:完整构建最常见的失败原因就是核心库缺失,
# 应当首先给出明确指引(CI 的 full-build 负向测试也依赖这一顺序)。
if(NOT RK_PIPE_CI_BUILD)
    set(RK_PIPE_ARCH_DEFAULT ${CMAKE_SYSTEM_PROCESSOR})
    set(RK_PIPE_ARCH "${RK_PIPE_ARCH_DEFAULT}" CACHE STRING "Architecture dir under prebuilt/")
    set(RKPIPE_CORE_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${RK_PIPE_ARCH}/librkpipe_core.a"
        CACHE FILEPATH "Prebuilt closed-core static library (core scheduling engine + C ABI facade)")
    if(NOT EXISTS "${RKPIPE_CORE_ARCHIVE}")
        message(FATAL_ERROR
            "预编译核心库不存在: ${RKPIPE_CORE_ARCHIVE}\n"
            "核心库以静态库随 Release 分发。请从本仓库的 Releases 页下载对应版本的\n"
            "librkpipe_core.a 放入 prebuilt/${RK_PIPE_ARCH}/,或改用 CI 构建:\n"
            "  cmake -B build-ci -DRK_PIPE_CI_BUILD=ON")
    endif()
endif()

# ---- OpenCV -----------------------------------------------------------------
set(OPENCV_ROOT "" CACHE PATH "OpenCV installation prefix (with lib/ and include/opencv4)")
if(NOT OPENCV_ROOT AND DEFINED ENV{OPENCV_ROOT})
    set(OPENCV_ROOT "$ENV{OPENCV_ROOT}")
endif()

# ---- turbojpeg(Web 预览 JPEG 编码加速;CI 构建的单元测试也覆盖该路径) ---------
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
endif()

find_package(Threads REQUIRED)

# 单元测试用例清单(与 tests/test_unit.cpp 中的注册表一一对应):
# CI 模式编译纯逻辑用例;完整构建额外启用 tracker/detector 工厂用例
set(RK_PIPE_CI_TEST_NAMES
    config_load config_validate command_line
    postprocess_math event_engine
    performance_average
    detection_filter turbojpeg
    result_json result_sink    event_rules
)
set(RK_PIPE_FULL_TEST_NAMES ${RK_PIPE_CI_TEST_NAMES}
    detector_factory tracker_ids tracker_occlusion tracker_relink
    tracker_modes tracker_other_tasks tracker_perf tracker_bench
    web_token_auth
)

# ===========================================================================
# CI 模式: 纯逻辑单元测试,不依赖 RKNN/RGA/MPP/FFmpeg
# ===========================================================================
if(RK_PIPE_CI_BUILD)
    # 纯逻辑源码清单(不依赖硬件头文件);完整构建通过 GLOB 覆盖全部模块源码
    set(CI_PURE_SOURCES
        tests/test_unit.cpp
        src/config/app_config.cc
        src/core/performance.cc
        src/core/detection_filter.cc
        src/core/event_engine.cc
        src/core/task_result_json.cc
        src/io/result_sink.cc
        src/postprocess/postprocess.cc
        src/postprocess/postprocess_common.cc
    )
    add_executable(rk_pipe_unit_tests ${CI_PURE_SOURCES})
    target_compile_definitions(rk_pipe_unit_tests PRIVATE RK_PIPE_CI=1)
    target_include_directories(rk_pipe_unit_tests PRIVATE
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
        target_include_directories(rk_pipe_unit_tests PRIVATE ${OPENCV_ROOT}/include/opencv4)
        target_link_libraries(rk_pipe_unit_tests PRIVATE Threads::Threads
            ${OPENCV_ROOT}/lib/libopencv_core.so ${OPENCV_ROOT}/lib/libopencv_imgproc.so
            ${TURBOJPEG_LIB})
    else()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc)
        target_link_libraries(rk_pipe_unit_tests PRIVATE Threads::Threads ${OpenCV_LIBS} ${TURBOJPEG_LIB})
    endif()
    enable_testing()
    foreach(_name IN LISTS RK_PIPE_CI_TEST_NAMES)
        add_test(NAME unit_${_name} COMMAND rk_pipe_unit_tests ${_name})
    endforeach()
    add_test(NAME unit_all COMMAND rk_pipe_unit_tests)
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
find_library(RKNNRT_LIB NAMES rknnrt PATHS ${RKNN_DIR} NO_DEFAULT_PATH)
if(NOT RKNNRT_LIB)
    message(FATAL_ERROR "librknnrt.so not found in ${RKNN_DIR}: "
        "请放入对应版本的 RKNN Runtime(见 docs/build.md)")
endif()

# 板端多媒体运行库(FFmpeg / rockchip-mpp):显式探测,给出可操作的缺失提示
find_library(ROCKCHIP_MPP_LIB rockchip_mpp)
find_library(AVFORMAT_LIB avformat)
find_library(AVCODEC_LIB avcodec)
find_library(AVUTIL_LIB avutil)
find_library(SWSCALE_LIB swscale)
foreach(_dep IN ITEMS ROCKCHIP_MPP_LIB AVFORMAT_LIB AVCODEC_LIB AVUTIL_LIB SWSCALE_LIB)
    if(NOT ${_dep})
        message(FATAL_ERROR "${_dep} not found: 请安装板端多媒体运行库(rockchip-mpp / ffmpeg)")
    endif()
endforeach()

# 模块层源码(开源): 输入输出 / 模型封装 / 后处理 / 跟踪 / 配置 / 工具
file(GLOB_RECURSE RKPIPE_MODULE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compat/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compat/*.cpp
)
list(APPEND RKPIPE_MODULE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/performance.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/detection_filter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/event_engine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/task_result_json.cc
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
    ${RKNNRT_LIB}
    ${RGA_LIB}
    ${TURBOJPEG_LIB}
    ${ROCKCHIP_MPP_LIB}
    chromaprint_compat
    ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB}
)
# console_detector: 通过 C ABI(rkpipe.h)驱动完整流水线的板端控制台应用,
# 同时是 ABI 的验收测试
add_executable(console_detector apps/console_detector/console_detector.cc)
target_include_directories(console_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(console_detector PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
set_target_properties(console_detector PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)
target_link_options(console_detector PRIVATE "-Wl,--disable-new-dtags")

# 板端模型精度/性能一键评测(经 C ABI 驱动流水线;评测脚本与数据准备见 tools/eval/README.md)
add_executable(rknn_eval tools/eval/rknn_eval_main.cc)
target_include_directories(rknn_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(rknn_eval PRIVATE RK_EVAL_SCRIPTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/eval")
target_link_libraries(rknn_eval PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
set_target_properties(rknn_eval PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)
target_link_options(rknn_eval PRIVATE "-Wl,--disable-new-dtags")

# ---------------------------------------------------------------------------
# 守护进程（B3 REST 服务包 + C2 配置热切换）：仅依赖线程库，不链接核心库
# ---------------------------------------------------------------------------
find_path(LLM_CURL_INCLUDE_DIR curl/curl.h
    HINTS ${PROJECT_SOURCE_DIR}/3rdparty/curl/include)
find_library(LLM_CURL_LIB NAMES curl)
if(LLM_CURL_INCLUDE_DIR AND LLM_CURL_LIB)
    set(RK_PIPE_HAVE_CURL ON)
    message(STATUS "libcurl: ${LLM_CURL_LIB} (LLM cloud provider enabled)")
else()
    set(RK_PIPE_HAVE_CURL OFF)
    message(STATUS "libcurl not found: LLM module builds mock-only (cloud provider unavailable)")
endif()

set(RK_PIPE_DAEMON_SOURCES
    src/io/http_request.cc
    src/utils/http_post.cc
    src/daemon_main.cc
    src/daemon/events_store.cc
    src/daemon/llm_review.cc
    src/daemon/daemon_pages.cc
    src/daemon/daemon_sysinfo.cc
    src/daemon/daemon_http.cc
    src/daemon/daemon_supervision.cc
    src/daemon/daemon_api.cc
    src/daemon/daemon_prometheus.cc
    src/daemon/flywheel.cc
    src/core/thermal_governor.cc
)
add_executable(rk_pipe_daemon ${RK_PIPE_DAEMON_SOURCES}
    src/llm/provider_mock.cc
)
if(RK_PIPE_HAVE_CURL)
    list(APPEND RK_PIPE_DAEMON_SOURCES src/llm/provider_cloud.cc)
endif()
target_sources(rk_pipe_daemon PRIVATE ${RK_PIPE_DAEMON_SOURCES})
target_link_libraries(rk_pipe_daemon Threads::Threads)
if(RK_PIPE_HAVE_CURL)
    target_link_libraries(rk_pipe_daemon ${LLM_CURL_LIB})
    target_compile_definitions(rk_pipe_daemon PRIVATE RK_PIPE_HAVE_CURL=1)
else()
    target_compile_definitions(rk_pipe_daemon PRIVATE RK_PIPE_LLM_NO_CLOUD=1)
endif()
target_compile_definitions(rk_pipe_daemon PRIVATE "RK_PIPE_GIT_COMMIT=\"${RK_PIPE_GIT_COMMIT}\"")
target_include_directories(rk_pipe_daemon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${LLM_CURL_INCLUDE_DIR}
)
set_target_properties(rk_pipe_daemon PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# ---- 单元测试(完整构建;CI 模式被 #ifndef RK_PIPE_CI 剔除的 tracker/工厂用例在此激活)
include(CTest)
if(BUILD_TESTING)
    add_executable(rk_pipe_unit_tests tests/test_unit.cpp)
    target_include_directories(rk_pipe_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TURBOJPEG_INCLUDE_DIR}
    )
    target_link_libraries(rk_pipe_unit_tests PRIVATE ${RKPIPE_LINK_GROUP} ${RKPIPE_LINK_DEPS})
    foreach(_name IN LISTS RK_PIPE_FULL_TEST_NAMES)
        add_test(NAME unit_${_name} COMMAND rk_pipe_unit_tests ${_name})
    endforeach()
    add_test(NAME unit_all COMMAND rk_pipe_unit_tests)
endif()


# ---- 安装 --------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS console_detector rknn_eval rk_pipe_daemon
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS chromaprint_compat LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${RKNNRT_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/rkpipe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/core DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rkpipe-modules)
install(FILES ${RKPIPE_CORE_ARCHIVE} DESTINATION ${CMAKE_INSTALL_LIBDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rkpipeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES cmake/rkpipeConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rkpipeConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rkpipe)
