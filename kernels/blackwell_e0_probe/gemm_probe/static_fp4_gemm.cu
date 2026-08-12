#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 8;
constexpr int kTileK = 64;

__device__ __forceinline__ uint32_t load_nibble(const uint8_t* row,
                                                 int index) {
  const uint32_t byte = row[index >> 1];
  return (byte >> (4 * (index & 1))) & 0xfu;
}

__device__ __forceinline__ uint32_t load_scale_register(
    const uint8_t* scales, int outer_index, int scale_stride,
    int block_base) {
  uint32_t result = 0;
#pragma unroll
  for (int block = 0; block < 4; ++block) {
    result |= uint32_t(scales[outer_index * scale_stride + block_base + block])
              << (8 * block);
  }
  return result;
}

}  // namespace

extern "C" __global__ void static_fp4_gemm(
    const uint8_t* packed_a, const uint8_t* packed_b,
    const uint8_t* a_scales, const uint8_t* b_scales,
    int logical_m, int logical_n, int padded_k,
    float alpha_a, float alpha_b, float* output) {
  const int lane = int(threadIdx.x) & 31;
  const int thread_mn_minor = lane & 3;
  const int thread_k_or_n = lane >> 2;
  const int tile_m = int(blockIdx.y) * kTileM;
  const int tile_n = int(blockIdx.x) * kTileN;
  const int payload_stride = padded_k / 2;
  const int scale_stride = padded_k / 16;

  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  const uint16_t scale_selector = 0;

#pragma unroll 1
  for (int k_base = 0; k_base < padded_k; k_base += kTileK) {
    uint32_t a[4] = {};
#pragma unroll
    for (int value = 0; value < 32; ++value) {
      const int m = thread_k_or_n + 8 * ((value >> 3) & 1);
      const int k = 8 * thread_mn_minor + (value & 7) + 32 * (value >> 4);
      const uint8_t* row = packed_a + (tile_m + m) * payload_stride;
      a[value >> 3] |= load_nibble(row, k_base + k) << (4 * (value & 7));
    }

    uint32_t b[2] = {};
#pragma unroll
    for (int value = 0; value < 16; ++value) {
      const int n = thread_k_or_n;
      const int k = 8 * thread_mn_minor + (value & 7) + 32 * (value >> 3);
      const uint8_t* column = packed_b + (tile_n + n) * payload_stride;
      b[value >> 3] |= load_nibble(column, k_base + k)
                         << (4 * (value & 7));
    }

    const int scale_m = tile_m + 8 * (lane & 1) + (lane >> 2);
    const int scale_n = tile_n + (lane >> 2);
    const int block_base = k_base / 16;
    const uint32_t sfa = load_scale_register(
        a_scales, scale_m, scale_stride, block_base);
    const uint32_t sfb = load_scale_register(
        b_scales, scale_n, scale_stride, block_base);

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
  }

  const float global_scale = alpha_a * alpha_b;
  const float accumulators[4] = {d0, d1, d2, d3};
#pragma unroll
  for (int value = 0; value < 4; ++value) {
    const int m = tile_m + thread_k_or_n + 8 * (value >> 1);
    const int n = tile_n + 2 * thread_mn_minor + (value & 1);
    if (m < logical_m && n < logical_n) {
      output[m * logical_n + n] = accumulators[value] * global_scale;
    }
  }
}
