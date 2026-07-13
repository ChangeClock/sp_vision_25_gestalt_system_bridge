cmake_minimum_required(VERSION 3.16)

# OpenVINO: only default to the Linux system install when unset (Windows
# passes -DOpenVINO_DIR at configure time).
if(NOT DEFINED OpenVINO_DIR AND NOT WIN32)
  set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

if(NOT WIN32)
  add_subdirectory(planner/tinympc)
endif()

set(AUTO_AIM_SRC
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
)
if(NOT WIN32)
  # multithread/* call io::CBoard (hardware CAN) and planner needs tinympc +
  # <dirent.h>; none are referenced by the core solve/track/aim/shoot chain.
  list(APPEND AUTO_AIM_SRC
      multithread/commandgener.cpp
      multithread/mt_detector.cpp
      planner/planner.cpp
  )
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SRC})

if(WIN32)
  # ${YAMLCPP_TARGET} (from top-level) carries YAML_CPP_STATIC_DEFINE — without
  # it the headers default to __declspec(dllimport) against our static lib.
  target_link_libraries(auto_aim openvino::runtime spdlog::spdlog fmt::fmt
                        ${YAMLCPP_TARGET} ${OpenCV_LIBS})
  # /FI numeric: std::accumulate reaches tracker.cpp transitively on libstdc++
  # but not on MSVC STL — force-include instead of touching upstream sources.
  target_compile_options(auto_aim PRIVATE /utf-8 /FInumeric)
else()
  target_link_libraries(auto_aim io openvino::runtime tinympcstatic)
endif()
