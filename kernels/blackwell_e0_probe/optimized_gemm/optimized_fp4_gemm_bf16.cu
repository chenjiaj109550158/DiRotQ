#include "optimized_fp4_bf16_kernel.hpp"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "optimized_fp4_gemm_bf16.cu must be compiled for sm_120a"
#endif

extern "C" __global__
__launch_bounds__(blackwell_optimized_fp4_bf16::GemmKernel::MaxThreadsPerBlock,
                  blackwell_optimized_fp4_bf16::GemmKernel::MinBlocksPerMultiprocessor)
void optimized_fp4_gemm_bf16_kernel(
    CUTLASS_GRID_CONSTANT blackwell_optimized_fp4_bf16::GemmKernel::Params const params) {
  extern __shared__ char shared_storage[];
  blackwell_optimized_fp4_bf16::GemmKernel operation;
  operation(params, shared_storage);
}
