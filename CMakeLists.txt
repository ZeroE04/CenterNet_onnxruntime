cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

add_executable(predict
  ${CMAKE_CURRENT_LIST_DIR}/predict.cpp
)

target_link_libraries(predict
  PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
)

target_include_directories(predict
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
  ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
)

include_directories(${onnxruntime_INCLUDE_DIRS})

