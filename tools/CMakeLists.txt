cmake_minimum_required(VERSION 3.16)

set(TOOLS_SRC
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
    pid.cpp
    crc.cpp
)
if(NOT WIN32)
  # plotter is POSIX-UDP (PlotJuggler); excluded on Windows, gestalt.cpp
  # guards its uses with #ifndef _WIN32.
  list(APPEND TOOLS_SRC plotter.cpp)
endif()

add_library(tools OBJECT ${TOOLS_SRC})
if(WIN32)
  # /FI numeric: libstdc++ pulls <numeric> transitively (std::iota), MSVC's
  # STL does not — force-include keeps the sources byte-identical to upstream.
  target_compile_options(tools PRIVATE /utf-8 /FInumeric)
  target_link_libraries(tools spdlog::spdlog fmt::fmt ${OpenCV_LIBS})
endif()
