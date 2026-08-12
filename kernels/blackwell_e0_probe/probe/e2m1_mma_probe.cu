#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 64;

__device__ __forceinline__ uint32_t load_nibble(const uint8_t* packed,
                                                 int index) {
  const uint32_t byte = packed[index >> 1];
  return (byte >> (4 * (index & 1))) & 0xfu;
}

__device__ __forceinline__ uint32_t load_scale_register(
    const uint8_t* scales, int outer_index) {
  uint32_t result = 0;
#pragma unroll
  for (int block = 0; block < 4; ++block) {
    result |= uint32_t(scales[outer_index * 4 + block]) << (8 * block);
  }
  return result;
}

__global__ void e2m1_m16n8k64_kernel(const uint8_t* packed_a,
                                      const uint8_t* packed_b,
                                      const uint8_t* a_scales,
                                      const uint8_t* b_scales,
                                      float* output) {
  const int lane = int(threadIdx.x) & 31;
  const int thread_mn_minor = lane & 3;
  const int thread_k_or_n = lane >> 2;

  // CUTLASS v4.0.0 mma_traits_sm120.hpp ALayout, lines 151-153:
  // (T32,V32) -> (M16,K64). CuTe's codomain is colex (M fastest), so
  // offset=m+16*k; convert that coordinate to the handoff's m*64+k stream.
  uint32_t a[4] = {};
#pragma unroll
  for (int value = 0; value < 32; ++value) {
    const int m = thread_k_or_n + 8 * ((value >> 3) & 1);
    const int k = 8 * thread_mn_minor + (value & 7) + 32 * (value >> 4);
    const int logical_index = m * kK + k;
    a[value >> 3] |= load_nibble(packed_a, logical_index)
                     << (4 * (value & 7));
  }

  // CUTLASS v4.0.0 mma_traits_sm120.hpp BLayout, lines 154-156:
  // (T32,V16) -> (N8,K64), with colex offset=n+8*k. The handoff's packed
  // B stream is exactly n*64+k after converting the coordinate.
  uint32_t b[2] = {};
#pragma unroll
  for (int value = 0; value < 16; ++value) {
    const int n = thread_k_or_n;
    const int k = 8 * thread_mn_minor + (value & 7) + 32 * (value >> 3);
    const int logical_index = n * kK + k;
    b[value >> 3] |= load_nibble(packed_b, logical_index)
                     << (4 * (value & 7));
  }

  // SFALayout duplicates each m scale vector across two lanes; SFBLayout
  // duplicates each n scale vector across four lanes. Each register contains
  // the four UE4M3 bytes for K blocks [0,16), [16,32), [32,48), [48,64).
  const int scale_m = 8 * (lane & 1) + (lane >> 2);
  const int scale_n = lane >> 2;
  const uint32_t sfa = load_scale_register(a_scales, scale_m);
  const uint32_t sfb = load_scale_register(b_scales, scale_n);

  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  const uint16_t scale_selector = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14}, {%15, %16},"
      "{%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(d0), "f"(d1), "f"(d2), "f"(d3),
        "r"(sfa), "h"(scale_selector), "h"(scale_selector),
        "r"(sfb), "h"(scale_selector), "h"(scale_selector));
#endif

  // CUTLASS SM80_16x8_Row accumulator layout: (T32,V4)->(M16,N8),
  // again with the first codomain dimension (M) fastest.
  const float accumulators[4] = {d0, d1, d2, d3};
#pragma unroll
  for (int value = 0; value < 4; ++value) {
    const int m = thread_k_or_n + 8 * (value >> 1);
    const int n = 2 * thread_mn_minor + (value & 1);
    output[m * kN + n] = accumulators[value];
  }
}

int check(cudaError_t status) {
  return status == cudaSuccess ? 0 : int(status);
}

}  // namespace

extern "C" int blackwell_e2m1_m16n8k64(
    const uint8_t* host_packed_a, const uint8_t* host_packed_b,
    const uint8_t* host_a_scales, const uint8_t* host_b_scales,
    float* host_output) {
  uint8_t* packed_a = nullptr;
  uint8_t* packed_b = nullptr;
  uint8_t* a_scales = nullptr;
  uint8_t* b_scales = nullptr;
  float* output = nullptr;

  cudaError_t status = cudaMalloc(&packed_a, kM * kK / 2);
  if (status == cudaSuccess) status = cudaMalloc(&packed_b, kK * kN / 2);
  if (status == cudaSuccess) status = cudaMalloc(&a_scales, kM * 4);
  if (status == cudaSuccess) status = cudaMalloc(&b_scales, kN * 4);
  if (status == cudaSuccess) status = cudaMalloc(&output, kM * kN * sizeof(float));
  if (status == cudaSuccess) {
    status = cudaMemcpy(packed_a, host_packed_a, kM * kK / 2,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(packed_b, host_packed_b, kK * kN / 2,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(a_scales, host_a_scales, kM * 4,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(b_scales, host_b_scales, kN * 4,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    e2m1_m16n8k64_kernel<<<1, 32>>>(packed_a, packed_b, a_scales,
                                    b_scales, output);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaMemcpy(host_output, output, kM * kN * sizeof(float),
                        cudaMemcpyDeviceToHost);
  }

  cudaFree(output);
  cudaFree(b_scales);
  cudaFree(a_scales);
  cudaFree(packed_b);
  cudaFree(packed_a);
  return check(status);
}
