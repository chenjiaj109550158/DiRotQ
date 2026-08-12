#include "optimized_fp4_kernel.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutlass/util/packed_stride.hpp"

namespace optimized_runner {

using namespace blackwell_optimized_fp4;

constexpr uint8_t kCanary = 0xa5;
constexpr size_t kCanaryBytes = 64;
constexpr uint8_t kUe4m3One = 0x38;

#pragma pack(push, 1)
struct InputHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t kp;
  int32_t mp;
  int32_t np;
  float alpha_a;
  float alpha_b;
  uint8_t reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(InputHeader) == 64);

void check_cuda(cudaError_t result, char const* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check_driver(CUresult result, char const* operation) {
  if (result != CUDA_SUCCESS) {
    char const* text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(operation) + ": " + (text ? text : "unknown"));
  }
}

void check_cutlass(cutlass::Status result, char const* operation) {
  if (result != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(result));
  }
}

template <class Layout>
__global__ void canonical_to_native_scales(uint8_t const* canonical,
                                           uint8_t* native,
                                           int outer,
                                           int blocks,
                                           Layout layout) {
  int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int count = outer * blocks;
  if (linear < count) {
    int row_or_column = linear / blocks;
    int block = linear - row_or_column * blocks;
    native[layout(row_or_column, block * 16, 0)] = canonical[linear];
  }
}

__global__ void fp32_to_bf16(float const* input, __nv_bfloat16* output, size_t count) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16(input[index]);
  }
}

struct DeviceAllocation {
  void* pointer = nullptr;
  DeviceAllocation() = default;
  explicit DeviceAllocation(size_t bytes) {
    if (bytes) {
      check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }
  }
  DeviceAllocation(DeviceAllocation const&) = delete;
  DeviceAllocation& operator=(DeviceAllocation const&) = delete;
  DeviceAllocation(DeviceAllocation&& other) noexcept : pointer(other.pointer) {
    other.pointer = nullptr;
  }
  ~DeviceAllocation() {
    if (pointer) cudaFree(pointer);
  }
};

std::unordered_map<std::string, std::string> parse_options(int argc, char** argv) {
  std::unordered_map<std::string, std::string> result;
  for (int index = 1; index < argc; ++index) {
    std::string key = argv[index];
    if (key.rfind("--", 0) != 0 || index + 1 == argc) {
      throw std::runtime_error("arguments must be --key value pairs");
    }
    result[key.substr(2)] = argv[++index];
  }
  return result;
}

std::string required(std::unordered_map<std::string, std::string> const& options,
                     std::string const& key) {
  auto iterator = options.find(key);
  if (iterator == options.end()) throw std::runtime_error("missing --" + key);
  return iterator->second;
}

std::vector<uint8_t> read_file(std::string const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open input: " + path);
  auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("cannot size input");
  std::vector<uint8_t> result(static_cast<size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("cannot read complete input");
  return result;
}

void write_file(std::string const& path, void const* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot open output: " + path);
  stream.write(reinterpret_cast<char const*>(data), bytes);
  if (!stream) throw std::runtime_error("cannot write complete output");
}

template <class Function>
std::vector<float> time_rounds(Function function, int warmup, int iterations,
                               int rounds, cudaStream_t stream) {
  for (int index = 0; index < warmup; ++index) function();
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  std::vector<float> result;
  try {
    for (int round = 0; round < rounds; ++round) {
      check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
      for (int index = 0; index < iterations; ++index) function();
      check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
      check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
      float milliseconds = 0;
      check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
                 "cudaEventElapsedTime");
      result.push_back(milliseconds / float(iterations));
    }
  } catch (...) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    throw;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return result;
}

void render_array(std::ostream& stream, std::vector<float> const& values) {
  stream << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) stream << ',';
    stream << std::setprecision(9) << values[index];
  }
  stream << ']';
}

}  // namespace optimized_runner

using namespace optimized_runner;

int main(int argc, char** argv) try {
  auto options = parse_options(argc, argv);
  std::string cubin_path = required(options, "cubin");
  std::string input_path = required(options, "input");
  std::string output_path = required(options, "output");
  std::string bf16_path = required(options, "bf16-output");
  std::string report_path = required(options, "report");
  int warmup = std::stoi(options.count("warmup") ? options.at("warmup") : "1");
  int iterations = std::stoi(options.count("iterations") ? options.at("iterations") : "1");
  int rounds = std::stoi(options.count("rounds") ? options.at("rounds") : "1");
  if (warmup < 0 || iterations <= 0 || rounds <= 0) {
    throw std::runtime_error("invalid benchmark counts");
  }

  auto input = read_file(input_path);
  if (input.size() < sizeof(InputHeader)) throw std::runtime_error("truncated header");
  InputHeader header;
  std::memcpy(&header, input.data(), sizeof(header));
  if (std::memcmp(header.magic, "OPTFP4V1", 8) || header.version != 1 ||
      header.header_bytes != sizeof(InputHeader)) {
    throw std::runtime_error("unsupported input header");
  }
  if (header.m <= 0 || header.n <= 0 || header.k <= 0 || header.kp <= 0 ||
      header.mp <= 0 || header.np <= 0 || header.mp % 16 || header.np % 8 ||
      header.kp % 64 || header.k > header.kp || header.m > header.mp ||
      header.n > header.np || !std::isfinite(header.alpha_a) ||
      !std::isfinite(header.alpha_b) || !(header.alpha_a > 0) ||
      !(header.alpha_b > 0)) {
    throw std::runtime_error("invalid shape or alpha contract");
  }
  size_t a_bytes = size_t(header.mp) * size_t(header.kp) / 2;
  size_t b_bytes = size_t(header.np) * size_t(header.kp) / 2;
  size_t as_bytes = size_t(header.mp) * size_t(header.kp) / 16;
  size_t bs_bytes = size_t(header.np) * size_t(header.kp) / 16;
  size_t expected_bytes = sizeof(InputHeader) + a_bytes + b_bytes + as_bytes + bs_bytes;
  if (input.size() != expected_bytes) throw std::runtime_error("input byte size mismatch");
  uint8_t const* host_a = input.data() + sizeof(InputHeader);
  uint8_t const* host_b = host_a + a_bytes;
  uint8_t const* host_as = host_b + b_bytes;
  uint8_t const* host_bs = host_as + as_bytes;

  check_cuda(cudaSetDevice(0), "cudaSetDevice");
  check_cuda(cudaFree(nullptr), "cudaFree(init)");
  cudaDeviceProp property{};
  check_cuda(cudaGetDeviceProperties(&property, 0), "cudaGetDeviceProperties");
  if (property.major != 12 || property.minor != 0) {
    throw std::runtime_error("runner requires compute capability 12.0");
  }
  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

  DeviceAllocation device_a(a_bytes), device_b(b_bytes);
  DeviceAllocation canonical_as(as_bytes), canonical_bs(bs_bytes);
  check_cuda(cudaMemcpyAsync(device_a.pointer, host_a, a_bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(A)");
  check_cuda(cudaMemcpyAsync(device_b.pointer, host_b, b_bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(B)");
  check_cuda(cudaMemcpyAsync(canonical_as.pointer, host_as, as_bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(SFA)");
  check_cuda(cudaMemcpyAsync(canonical_bs.pointer, host_bs, bs_bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(SFB)");

  auto problem = cute::make_shape(header.mp, header.np, header.kp, 1);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
  size_t native_as_bytes = static_cast<size_t>(cute::size(cute::filter_zeros(layout_sfa)));
  size_t native_bs_bytes = static_cast<size_t>(cute::size(cute::filter_zeros(layout_sfb)));
  DeviceAllocation native_as(native_as_bytes), native_bs(native_bs_bytes);
  int blocks = header.kp / 16;
  auto launch_a_scale = [&] {
    check_cuda(cudaMemsetAsync(native_as.pointer, kUe4m3One, native_as_bytes, stream),
               "cudaMemsetAsync(native SFA)");
    int count = header.mp * blocks;
    canonical_to_native_scales<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<uint8_t const*>(canonical_as.pointer),
        static_cast<uint8_t*>(native_as.pointer), header.mp, blocks, layout_sfa);
    check_cuda(cudaGetLastError(), "canonical_to_native_scales(A)");
  };
  auto launch_b_scale = [&] {
    check_cuda(cudaMemsetAsync(native_bs.pointer, kUe4m3One, native_bs_bytes, stream),
               "cudaMemsetAsync(native SFB)");
    int count = header.np * blocks;
    canonical_to_native_scales<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<uint8_t const*>(canonical_bs.pointer),
        static_cast<uint8_t*>(native_bs.pointer), header.np, blocks, layout_sfb);
    check_cuda(cudaGetLastError(), "canonical_to_native_scales(B)");
  };
  launch_a_scale();
  launch_b_scale();

  size_t output_elements = size_t(header.mp) * size_t(header.np);
  size_t output_bytes = output_elements * sizeof(float);
  DeviceAllocation guarded_output(output_bytes + 2 * kCanaryBytes);
  check_cuda(cudaMemsetAsync(guarded_output.pointer, kCanary,
                             output_bytes + 2 * kCanaryBytes, stream),
             "cudaMemsetAsync(output canary)");
  auto* output = reinterpret_cast<float*>(
      static_cast<uint8_t*>(guarded_output.pointer) + kCanaryBytes);
  DeviceAllocation bf16_output(output_elements * sizeof(__nv_bfloat16));

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{},
      cute::make_shape(header.mp, header.kp, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{},
      cute::make_shape(header.np, header.kp, 1));
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{},
      cute::make_shape(header.mp, header.np, 1));
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{},
      cute::make_shape(header.mp, header.np, 1));
  float alpha = header.alpha_a * header.alpha_b;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<ElementA::DataType const*>(device_a.pointer), stride_a,
       reinterpret_cast<ElementB::DataType const*>(device_b.pointer), stride_b,
       reinterpret_cast<ElementA::ScaleFactorType const*>(native_as.pointer), layout_sfa,
       reinterpret_cast<ElementB::ScaleFactorType const*>(native_bs.pointer), layout_sfb},
      {{alpha, 0.0f}, nullptr, stride_c, output, stride_d}};
  check_cutlass(Gemm::can_implement(arguments), "Gemm::can_implement");
  size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  DeviceAllocation workspace(workspace_bytes);
  check_cutlass(GemmKernel::initialize_workspace(arguments, workspace.pointer, stream),
                "GemmKernel::initialize_workspace");
  auto params = GemmKernel::to_underlying_arguments(arguments, workspace.pointer);
  dim3 grid = GemmKernel::get_grid_shape(params);
  dim3 block = GemmKernel::get_block_shape();
  int smem_bytes = GemmKernel::SharedStorageSize;

  check_driver(cuInit(0), "cuInit");
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  check_driver(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
  check_driver(cuModuleGetFunction(&function, module, "optimized_fp4_gemm_kernel"),
               "cuModuleGetFunction");
  check_driver(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  smem_bytes),
               "cuFuncSetAttribute(dynamic smem)");
  auto launch_gemm = [&] {
    void* kernel_arguments[] = {&params};
    check_driver(cuLaunchKernel(function, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                smem_bytes, reinterpret_cast<CUstream>(stream),
                                kernel_arguments, nullptr),
                 "cuLaunchKernel(optimized_fp4_gemm_kernel)");
  };
  auto launch_cast = [&] {
    fp32_to_bf16<<<(output_elements + 255) / 256, 256, 0, stream>>>(
        output, static_cast<__nv_bfloat16*>(bf16_output.pointer), output_elements);
    check_cuda(cudaGetLastError(), "fp32_to_bf16");
  };

  launch_gemm();
  launch_cast();
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(correctness)");

  std::vector<uint8_t> guarded(output_bytes + 2 * kCanaryBytes);
  check_cuda(cudaMemcpy(guarded.data(), guarded_output.pointer, guarded.size(),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
  bool prefix_ok = std::all_of(guarded.begin(), guarded.begin() + kCanaryBytes,
                               [](uint8_t value) { return value == kCanary; });
  bool suffix_ok = std::all_of(guarded.end() - kCanaryBytes, guarded.end(),
                               [](uint8_t value) { return value == kCanary; });
  write_file(output_path, guarded.data() + kCanaryBytes, output_bytes);
  std::vector<uint8_t> bf16_host(output_elements * sizeof(__nv_bfloat16));
  check_cuda(cudaMemcpy(bf16_host.data(), bf16_output.pointer, bf16_host.size(),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(BF16 output)");
  write_file(bf16_path, bf16_host.data(), bf16_host.size());

  auto a_scale_rounds = time_rounds(launch_a_scale, warmup, iterations, rounds, stream);
  auto b_scale_rounds = time_rounds(launch_b_scale, warmup, iterations, rounds, stream);
  auto gemm_rounds = time_rounds(launch_gemm, warmup, iterations, rounds, stream);
  auto cast_rounds = time_rounds(launch_cast, warmup, iterations, rounds, stream);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(final)");

  std::ofstream report(report_path, std::ios::trunc);
  if (!report) throw std::runtime_error("cannot open report");
  report << "{\n"
         << "  \"passed\": " << ((prefix_ok && suffix_ok) ? "true" : "false") << ",\n"
         << "  \"gpu\": \"" << property.name << "\",\n"
         << "  \"compute_capability\": [" << property.major << ',' << property.minor << "],\n"
         << "  \"logical_shape\": [" << header.m << ',' << header.n << ',' << header.k << "],\n"
         << "  \"padded_shape\": [" << header.mp << ',' << header.np << ',' << header.kp << "],\n"
         << "  \"alpha_a\": " << std::setprecision(9) << header.alpha_a << ",\n"
         << "  \"alpha_b\": " << std::setprecision(9) << header.alpha_b << ",\n"
         << "  \"kernel_symbol\": \"optimized_fp4_gemm_kernel\",\n"
         << "  \"mma_atom\": [16,8,64],\n"
         << "  \"threadblock_shape\": [128,128,128],\n"
         << "  \"cluster_shape\": [1,1,1],\n"
         << "  \"stage_count\": " << Gemm::kStages << ",\n"
         << "  \"scale_vector_size\": 16,\n"
         << "  \"kernel_schedule\": \"TmaWarpSpecializedCooperative\",\n"
         << "  \"grid\": [" << grid.x << ',' << grid.y << ',' << grid.z << "],\n"
         << "  \"block\": [" << block.x << ',' << block.y << ',' << block.z << "],\n"
         << "  \"dynamic_shared_bytes\": " << smem_bytes << ",\n"
         << "  \"workspace_bytes\": " << workspace_bytes << ",\n"
         << "  \"canonical_a_bytes\": " << a_bytes << ",\n"
         << "  \"canonical_b_bytes\": " << b_bytes << ",\n"
         << "  \"canonical_a_scale_bytes\": " << as_bytes << ",\n"
         << "  \"canonical_b_scale_bytes\": " << bs_bytes << ",\n"
         << "  \"native_a_scale_bytes\": " << native_as_bytes << ",\n"
         << "  \"native_b_scale_bytes\": " << native_bs_bytes << ",\n"
         << "  \"a_payload_transform_required\": false,\n"
         << "  \"b_payload_prepack_required\": false,\n"
         << "  \"canary_prefix_ok\": " << (prefix_ok ? "true" : "false") << ",\n"
         << "  \"canary_suffix_ok\": " << (suffix_ok ? "true" : "false") << ",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"a_scale_transform_round_ms\": ";
  render_array(report, a_scale_rounds);
  report << ",\n  \"b_scale_prepack_round_ms\": ";
  render_array(report, b_scale_rounds);
  report << ",\n  \"gemm_round_ms\": ";
  render_array(report, gemm_rounds);
  report << ",\n  \"output_bf16_cast_round_ms\": ";
  render_array(report, cast_rounds);
  report << "\n}\n";
  report.close();

  check_driver(cuModuleUnload(module), "cuModuleUnload");
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return (prefix_ok && suffix_ok) ? 0 : 2;
} catch (std::exception const& error) {
  std::cerr << "optimized_fp4_runner: " << error.what() << '\n';
  return 1;
}
