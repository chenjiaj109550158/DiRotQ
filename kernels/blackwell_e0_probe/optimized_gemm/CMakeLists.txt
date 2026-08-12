cmake_minimum_required(VERSION 3.24)
project(blackwell_optimized_fp4_probe LANGUAGES CXX CUDA)

set(CUTLASS_ROOT "" CACHE PATH "Path to the pinned official CUTLASS checkout")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set CUTLASS_ROOT to an official CUTLASS checkout")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120a)

set(PROBE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUTLASS_INCLUDE "${CUTLASS_ROOT}/include")
set(CUTLASS_UTIL_INCLUDE "${CUTLASS_ROOT}/tools/util/include")

add_executable(optimized_fp4_runner optimized_fp4_runner.cu)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
file(RELATIVE_PATH PROBE_REL "${REPO_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}")
file(RELATIVE_PATH CUTLASS_REL "${REPO_ROOT}" "${CUTLASS_ROOT}")
target_include_directories(optimized_fp4_runner PRIVATE
  "${PROBE_INCLUDE}" "${CUTLASS_INCLUDE}" "${CUTLASS_UTIL_INCLUDE}")
target_compile_options(optimized_fp4_runner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>)
target_link_libraries(optimized_fp4_runner PRIVATE CUDA::cudart CUDA::cuda_driver)
set_target_properties(optimized_fp4_runner PROPERTIES
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")

set(BASELINE_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/baseline_00.cubin")
add_custom_command(
  OUTPUT "${BASELINE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -std=c++17 -O3
          --expt-relaxed-constexpr -arch=sm_120a
          -I "${PROBE_REL}" -I "${CUTLASS_REL}/include"
          "${PROBE_REL}/optimized_fp4_gemm.cu"
          -o "${BASELINE_CUBIN}"
  DEPENDS optimized_fp4_gemm.cu optimized_fp4_kernel.hpp
  WORKING_DIRECTORY "${REPO_ROOT}"
  VERBATIM)
add_custom_target(optimized_fp4_cubin ALL DEPENDS "${BASELINE_CUBIN}")
