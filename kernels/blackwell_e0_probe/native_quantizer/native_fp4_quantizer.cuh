#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace blackwell_native_quantizer {

constexpr float kGlobalDivisor = 2688.0f;
constexpr uint8_t kUe4m3One = 0x38;

enum class Format : int { kE2M1 = 0, kE0M3 = 1 };

inline size_t native_scale_bytes(int mp, int kp) {
  return size_t((mp + 127) / 128) * size_t(kp / 64) * 512;
}

__host__ __device__ inline size_t native_scale_offset(int row, int block,
                                                       int mp, int kp) {
  int k_tiles = kp / 64;
  int atom = (row / 128) * k_tiles + block / 4;
  int within = (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
  return size_t(atom) * 512 + size_t(within);
}

__device__ inline float source_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ inline float source_to_float(__half value) {
  return __half2float(value);
}

template <class Source>
__global__ void global_absmax_kernel(Source const* source, size_t count,
                                     unsigned int* maximum_bits,
                                     int* invalid) {
  float local = 0.0f;
  for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += size_t(gridDim.x) * blockDim.x) {
    float value = source_to_float(source[index]);
    if (!isfinite(value)) {
      atomicExch(invalid, 1);
    } else {
      local = fmaxf(local, fabsf(value));
    }
  }
  for (int offset = 16; offset; offset >>= 1) {
    local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
  }
  __shared__ float warp_maxima[32];
  int lane = int(threadIdx.x) & 31;
  int warp = int(threadIdx.x) >> 5;
  if (lane == 0) warp_maxima[warp] = local;
  __syncthreads();
  if (warp == 0) {
    local = lane < (int(blockDim.x) + 31) / 32 ? warp_maxima[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) {
      local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
    }
    if (lane == 0) atomicMax(maximum_bits, __float_as_uint(local));
  }
}

__global__ void finalize_alpha_kernel(unsigned int const* maximum_bits,
                                      int const* invalid, float* alpha,
                                      float* scaled_alpha, float multiplier) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float maximum = __uint_as_float(*maximum_bits);
    *alpha = *invalid ? __int_as_float(0x7fffffff) : (maximum == 0.0f ? 1.0f : maximum / kGlobalDivisor);
    if (scaled_alpha) *scaled_alpha = *alpha * multiplier;
  }
}

__device__ inline float decode_ue4m3(uint8_t byte) {
  int exponent = (byte >> 3) & 15;
  int mantissa = byte & 7;
  if (exponent == 0) return float(mantissa) * 0x1p-9f;
  return (1.0f + float(mantissa) * 0.125f) * exp2f(float(exponent - 7));
}

__device__ inline uint8_t nearest_code(float value, Format format) {
  constexpr float e2[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float maximum = format == Format::kE2M1 ? 6.0f : 7.0f;
  float magnitude = fabsf(value);
  int best = 0;
  if (magnitude >= maximum) {
    best = 7;
  } else if (format == Format::kE0M3) {
    float best_distance = magnitude;
#pragma unroll
    for (int index = 1; index < 8; ++index) {
      float distance = fabsf(magnitude - float(index));
      if (distance < best_distance) {
        best = index;
        best_distance = distance;
      }
    }
  } else {
    float best_distance = magnitude;
#pragma unroll
    for (int index = 1; index < 8; ++index) {
      float distance = fabsf(magnitude - e2[index]);
      if (distance < best_distance) {
        best = index;
        best_distance = distance;
      }
    }
  }
  int sign = signbit(value) && best != 0 ? 8 : 0;
  return uint8_t(best | sign);
}

template <class Source>
__global__ void quantize_pack_native_kernel(
    Source const* source, int m, int k, int mp, int kp, Format format,
    float const* alpha, uint8_t* payload, uint8_t* native_scales) {
  int blocks = kp / 16;
  int outer = (mp + 127) / 128 * 128;
  int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (linear >= outer * blocks) return;
  int row = linear / blocks;
  int block = linear - row * blocks;
  int begin = block * 16;
  size_t scale_offset = native_scale_offset(row, block, mp, kp);
  if (row >= m || begin >= k) {
    native_scales[scale_offset] = kUe4m3One;
    if (row < mp) {
#pragma unroll
      for (int pair = 0; pair < 8; ++pair) payload[size_t(row) * (kp / 2) + begin / 2 + pair] = 0;
    }
    return;
  }
  float inverse_alpha = 1.0f / *alpha;
  float values[16];
  float block_maximum = 0.0f;
#pragma unroll
  for (int offset = 0; offset < 16; ++offset) {
    int column = begin + offset;
    float value = column < k ? source_to_float(source[size_t(row) * k + column]) * inverse_alpha : 0.0f;
    values[offset] = value;
    block_maximum = fmaxf(block_maximum, fabsf(value));
  }
  float denominator = format == Format::kE2M1 ? 6.0f : 7.0f;
  uint8_t scale_byte = block_maximum == 0.0f
      ? kUe4m3One
      : __nv_cvt_float_to_fp8(block_maximum / denominator, __NV_SATFINITE, __NV_E4M3);
  native_scales[scale_offset] = scale_byte;
  float scale = decode_ue4m3(scale_byte);
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    float low = scale == 0.0f ? copysignf(__int_as_float(0x7f800000), values[2 * pair]) : values[2 * pair] / scale;
    float high = scale == 0.0f ? copysignf(__int_as_float(0x7f800000), values[2 * pair + 1]) : values[2 * pair + 1] / scale;
    uint8_t packed = nearest_code(low, format) | uint8_t(nearest_code(high, format) << 4);
    payload[size_t(row) * (kp / 2) + begin / 2 + pair] = packed;
  }
}

inline int reduction_blocks(size_t elements) {
  return int(std::min<size_t>((elements + 255) / 256, 1024));
}

template <class Source>
inline void launch_absmax(Source const* source, size_t elements,
                          unsigned int* maximum_bits, int* invalid,
                          cudaStream_t stream) {
  cudaMemsetAsync(maximum_bits, 0, sizeof(unsigned int), stream);
  cudaMemsetAsync(invalid, 0, sizeof(int), stream);
  global_absmax_kernel<<<reduction_blocks(elements), 256, 0, stream>>>(
      source, elements, maximum_bits, invalid);
}

inline void launch_finalize(unsigned int const* maximum_bits, int const* invalid,
                            float* alpha, cudaStream_t stream) {
  finalize_alpha_kernel<<<1, 1, 0, stream>>>(maximum_bits, invalid, alpha, nullptr, 1.0f);
}

inline void launch_finalize_product(unsigned int const* maximum_bits, int const* invalid,
                                    float* alpha, float* product, float multiplier,
                                    cudaStream_t stream) {
  finalize_alpha_kernel<<<1, 1, 0, stream>>>(maximum_bits, invalid, alpha, product, multiplier);
}

template <class Source>
inline void launch_quantize(Source const* source, int m, int k, int mp, int kp,
                            Format format, float const* alpha,
                            uint8_t* payload, uint8_t* native_scales,
                            cudaStream_t stream) {
  int count = ((mp + 127) / 128 * 128) * (kp / 16);
  quantize_pack_native_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      source, m, k, mp, kp, format, alpha, payload, native_scales);
}

}  // namespace blackwell_native_quantizer
