cmake_minimum_required(VERSION 3.18)
project(reviewscope_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point at an unpacked ONNX Runtime release, e.g.
#   cmake -B build -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.22.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Unpacked onnxruntime release (has include/ and lib/)")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=<path to onnxruntime release>")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(reviewscope_infer src/main.cpp)
target_include_directories(reviewscope_infer PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(reviewscope_infer PRIVATE "${ONNXRUNTIME_LIB}")
if(MSVC)
  target_compile_options(reviewscope_infer PRIVATE /W4 /O2)
else()
  target_compile_options(reviewscope_infer PRIVATE -Wall -Wextra -O2)
  set_target_properties(reviewscope_infer PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()
