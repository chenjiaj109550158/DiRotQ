cmake_minimum_required(VERSION 3.24)
project(blackwell_native_fp4_quantizer LANGUAGES CXX CUDA)

set(CUTLASS_ROOT "" CACHE PATH "Path to pinned official CUTLASS checkout")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set CUTLASS_ROOT to the pinned official CUTLASS checkout")
endif()
find_package(CUDAToolkit 12.8 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120a)

add_executable(native_fp4_quantizer native_fp4_quantizer.cu)
target_compile_options(native_fp4_quantizer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>)
target_link_libraries(native_fp4_quantizer PRIVATE CUDA::cudart)
set_target_properties(native_fp4_quantizer PROPERTIES
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(native_fp4_pipeline_runner pipeline_runner.cu)
target_include_directories(native_fp4_pipeline_runner PRIVATE
  "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../optimized_gemm")
target_compile_options(native_fp4_pipeline_runner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>)
target_link_libraries(native_fp4_pipeline_runner PRIVATE CUDA::cudart CUDA::cuda_driver)
set_target_properties(native_fp4_pipeline_runner PROPERTIES
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
file(RELATIVE_PATH OPTIMIZED_REL "${REPO_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../optimized_gemm")
file(RELATIVE_PATH CUTLASS_REL "${REPO_ROOT}" "${CUTLASS_ROOT}")
set(BF16_BASELINE_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/bf16_baseline_00.cubin")
add_custom_command(
  OUTPUT "${BF16_BASELINE_CUBIN}"
  COMMAND "${CMAKE_CUDA_COMPILER}" --cubin -std=c++17 -O3
          --expt-relaxed-constexpr -arch=sm_120a
          -I "${OPTIMIZED_REL}" -I "${CUTLASS_REL}/include"
          "${OPTIMIZED_REL}/optimized_fp4_gemm_bf16.cu"
          -o "${BF16_BASELINE_CUBIN}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../optimized_gemm/optimized_fp4_gemm_bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../optimized_gemm/optimized_fp4_bf16_kernel.hpp"
  WORKING_DIRECTORY "${REPO_ROOT}"
  VERBATIM)
add_custom_target(optimized_fp4_bf16_cubin ALL DEPENDS "${BF16_BASELINE_CUBIN}")
