#include "native_fp4_quantizer.cuh"
#include "../optimized_gemm/optimized_fp4_bf16_kernel.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
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

namespace nq = blackwell_native_quantizer;
namespace og = blackwell_optimized_fp4_bf16;

namespace {

constexpr uint8_t kCanary = 0xa5;
constexpr size_t kCanaryBytes = 64;

#pragma pack(push, 1)
struct InputHeader {
  char magic[8]; uint32_t version; uint32_t header_bytes;
  int32_t m, n, k, kp, mp, np;
  float alpha_a, alpha_b; uint8_t reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(InputHeader) == 64);

void check(cudaError_t result, char const* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void check(CUresult result, char const* operation) {
  if (result != CUDA_SUCCESS) {
    char const* text = nullptr; cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(operation) + ": " + (text ? text : "unknown"));
  }
}
void check(cutlass::Status result, char const* operation) {
  if (result != cutlass::Status::kSuccess) throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(result));
}

struct Allocation {
  void* pointer = nullptr;
  explicit Allocation(size_t bytes) { if (bytes) check(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
  Allocation(Allocation const&) = delete;
  ~Allocation() { if (pointer) cudaFree(pointer); }
};

std::unordered_map<std::string, std::string> options(int argc, char** argv) {
  std::unordered_map<std::string, std::string> result;
  for (int index = 1; index < argc; ++index) {
    std::string key = argv[index];
    if (key.rfind("--", 0) || index + 1 == argc) throw std::runtime_error("arguments must be --key value pairs");
    result[key.substr(2)] = argv[++index];
  }
  return result;
}
std::string required(std::unordered_map<std::string, std::string> const& values, std::string const& key) {
  auto iterator = values.find(key); if (iterator == values.end()) throw std::runtime_error("missing --" + key); return iterator->second;
}
std::vector<uint8_t> read_file(std::string const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate); if (!stream) throw std::runtime_error("cannot open input");
  size_t bytes = size_t(stream.tellg()); std::vector<uint8_t> result(bytes); stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), bytes); if (!stream) throw std::runtime_error("cannot read input"); return result;
}
void write_file(std::string const& path, void const* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc); if (!stream) throw std::runtime_error("cannot create output");
  stream.write(reinterpret_cast<char const*>(data), bytes); if (!stream) throw std::runtime_error("cannot write output");
}

template <class Layout>
__global__ void canonical_to_native_scales(uint8_t const* canonical, uint8_t* native,
                                           int outer, int blocks, Layout layout) {
  int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (linear < outer * blocks) {
    int row = linear / blocks, block = linear - row * blocks;
    native[layout(row, block * 16, 0)] = canonical[linear];
  }
}

template <class Function>
std::vector<float> time_rounds(Function function, int warmup, int iterations, int rounds, cudaStream_t stream) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
  cudaEvent_t start, stop; check(cudaEventCreate(&start), "cudaEventCreate"); check(cudaEventCreate(&stop), "cudaEventCreate");
  std::vector<float> values;
  for (int round = 0; round < rounds; ++round) {
    check(cudaEventRecord(start, stream), "cudaEventRecord");
    for (int index = 0; index < iterations; ++index) function();
    check(cudaEventRecord(stop, stream), "cudaEventRecord"); check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime"); values.push_back(elapsed / iterations);
  }
  cudaEventDestroy(start); cudaEventDestroy(stop); return values;
}
void render(std::ostream& stream, std::vector<float> const& values) {
  stream << '['; for (size_t i = 0; i < values.size(); ++i) { if (i) stream << ','; stream << std::setprecision(9) << values[i]; } stream << ']';
}

template <class Source>
int run(std::vector<uint8_t> const& host_source, std::vector<uint8_t> const& static_input,
        nq::Format format, std::string const& cubin_path, std::string const& output_path,
        std::string const& report_path, int warmup, int iterations, int rounds) {
  if (static_input.size() < sizeof(InputHeader)) throw std::runtime_error("truncated static input");
  InputHeader header; std::memcpy(&header, static_input.data(), sizeof(header));
  if (std::memcmp(header.magic, "OPTFP4V1", 8) || header.version != 1 || header.header_bytes != sizeof(header)) throw std::runtime_error("invalid static input header");
  if (header.m <= 0 || header.n <= 0 || header.k <= 0 || header.mp % 16 || header.np % 8 || header.kp % 64 ||
      !std::isfinite(header.alpha_b) || !(header.alpha_b > 0)) throw std::runtime_error("invalid static input contract");
  size_t source_bytes = size_t(header.m) * header.k * sizeof(Source);
  if (host_source.size() != source_bytes) throw std::runtime_error("activation source byte size mismatch");
  size_t a_bytes = size_t(header.mp) * header.kp / 2, b_bytes = size_t(header.np) * header.kp / 2;
  size_t canonical_as_bytes = size_t(header.mp) * header.kp / 16, canonical_bs_bytes = size_t(header.np) * header.kp / 16;
  if (static_input.size() != sizeof(header) + a_bytes + b_bytes + canonical_as_bytes + canonical_bs_bytes) throw std::runtime_error("static input byte size mismatch");
  uint8_t const* host_b = static_input.data() + sizeof(header) + a_bytes;
  uint8_t const* host_bs = host_b + b_bytes + canonical_as_bytes;

  check(cudaSetDevice(0), "cudaSetDevice"); cudaStream_t stream; check(cudaStreamCreate(&stream), "cudaStreamCreate");
  Allocation source(source_bytes), device_a(a_bytes), device_b(b_bytes), canonical_bs(canonical_bs_bytes);
  Allocation maximum(sizeof(unsigned int)), invalid(sizeof(int)), alpha_a(sizeof(float)), alpha_product(sizeof(float));
  check(cudaMemcpyAsync(source.pointer, host_source.data(), source_bytes, cudaMemcpyHostToDevice, stream), "copy source");
  check(cudaMemcpyAsync(device_b.pointer, host_b, b_bytes, cudaMemcpyHostToDevice, stream), "copy B");
  check(cudaMemcpyAsync(canonical_bs.pointer, host_bs, canonical_bs_bytes, cudaMemcpyHostToDevice, stream), "copy BS");

  auto problem = cute::make_shape(header.mp, header.np, header.kp, 1);
  auto layout_sfa = og::ScaleConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = og::ScaleConfig::tile_atom_to_shape_SFB(problem);
  size_t native_as_bytes = size_t(cute::size(cute::filter_zeros(layout_sfa)));
  size_t native_bs_bytes = size_t(cute::size(cute::filter_zeros(layout_sfb)));
  if (native_as_bytes != nq::native_scale_bytes(header.mp, header.kp)) throw std::runtime_error("native SFA size contract changed");
  Allocation native_as(native_as_bytes), native_bs(native_bs_bytes);
  check(cudaMemsetAsync(native_bs.pointer, nq::kUe4m3One, native_bs_bytes, stream), "initialize native BS");
  int b_count = header.np * (header.kp / 16);
  canonical_to_native_scales<<<(b_count + 255) / 256, 256, 0, stream>>>(
      static_cast<uint8_t const*>(canonical_bs.pointer), static_cast<uint8_t*>(native_bs.pointer),
      header.np, header.kp / 16, layout_sfb);

  auto launch_absmax = [&] {
    nq::launch_absmax(static_cast<Source const*>(source.pointer), size_t(header.m) * header.k,
                      static_cast<unsigned int*>(maximum.pointer), static_cast<int*>(invalid.pointer), stream);
    nq::launch_finalize_product(static_cast<unsigned int const*>(maximum.pointer), static_cast<int const*>(invalid.pointer),
                        static_cast<float*>(alpha_a.pointer), static_cast<float*>(alpha_product.pointer), header.alpha_b, stream);
  };
  auto launch_pack = [&] {
    nq::launch_quantize(static_cast<Source const*>(source.pointer), header.m, header.k, header.mp, header.kp,
                        format, static_cast<float const*>(alpha_a.pointer), static_cast<uint8_t*>(device_a.pointer),
                        static_cast<uint8_t*>(native_as.pointer), stream);
  };
  auto launch_quantizer = [&] { launch_absmax(); launch_pack(); };
  launch_quantizer(); check(cudaGetLastError(), "quantizer launch"); check(cudaStreamSynchronize(stream), "quantizer setup");
  int host_invalid = 0; float host_alpha_a = 0;
  check(cudaMemcpy(&host_invalid, invalid.pointer, sizeof(int), cudaMemcpyDeviceToHost), "copy invalid");
  check(cudaMemcpy(&host_alpha_a, alpha_a.pointer, sizeof(float), cudaMemcpyDeviceToHost), "copy alpha");
  if (host_invalid || !std::isfinite(host_alpha_a) || !(host_alpha_a > 0)) throw std::runtime_error("activation contains NaN/Inf");

  size_t output_elements = size_t(header.mp) * header.np;
  size_t output_bytes = output_elements * sizeof(cutlass::bfloat16_t);
  Allocation guarded_output(output_bytes + 2 * kCanaryBytes);
  check(cudaMemsetAsync(guarded_output.pointer, kCanary, output_bytes + 2 * kCanaryBytes, stream), "output canary");
  auto* output = reinterpret_cast<cutlass::bfloat16_t*>(static_cast<uint8_t*>(guarded_output.pointer) + kCanaryBytes);
  auto stride_a = cutlass::make_cute_packed_stride(og::StrideA{}, cute::make_shape(header.mp, header.kp, 1));
  auto stride_b = cutlass::make_cute_packed_stride(og::StrideB{}, cute::make_shape(header.np, header.kp, 1));
  auto stride_c = cutlass::make_cute_packed_stride(og::StrideC{}, cute::make_shape(header.mp, header.np, 1));
  auto stride_d = cutlass::make_cute_packed_stride(og::StrideD{}, cute::make_shape(header.mp, header.np, 1));
  typename og::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, problem,
      {reinterpret_cast<og::ElementA::DataType const*>(device_a.pointer), stride_a,
       reinterpret_cast<og::ElementB::DataType const*>(device_b.pointer), stride_b,
       reinterpret_cast<og::ElementA::ScaleFactorType const*>(native_as.pointer), layout_sfa,
       reinterpret_cast<og::ElementB::ScaleFactorType const*>(native_bs.pointer), layout_sfb},
      {{}, nullptr, stride_c, output, stride_d}};
  arguments.epilogue.thread.alpha = 0.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.alpha_ptr = static_cast<float const*>(alpha_product.pointer);
  check(og::Gemm::can_implement(arguments), "Gemm::can_implement");
  size_t workspace_bytes = og::Gemm::get_workspace_size(arguments); Allocation workspace(workspace_bytes);
  check(og::GemmKernel::initialize_workspace(arguments, workspace.pointer, stream), "initialize workspace");
  auto params = og::GemmKernel::to_underlying_arguments(arguments, workspace.pointer);
  dim3 grid = og::GemmKernel::get_grid_shape(params), block = og::GemmKernel::get_block_shape(); int smem = og::GemmKernel::SharedStorageSize;
  check(cuInit(0), "cuInit"); CUmodule module = nullptr; CUfunction function = nullptr;
  check(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
  check(cuModuleGetFunction(&function, module, "optimized_fp4_gemm_bf16_kernel"), "cuModuleGetFunction");
  check(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem), "cuFuncSetAttribute");
  auto launch_gemm = [&] { void* args[] = {&params}; check(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      smem, reinterpret_cast<CUstream>(stream), args, nullptr), "cuLaunchKernel"); };
  auto launch_pipeline = [&] { launch_quantizer(); launch_gemm(); };
  launch_gemm(); check(cudaStreamSynchronize(stream), "correctness launch");
  std::vector<uint8_t> output_guard(output_bytes + 2 * kCanaryBytes);
  check(cudaMemcpy(output_guard.data(), guarded_output.pointer, output_guard.size(), cudaMemcpyDeviceToHost), "copy output");
  bool canary = std::all_of(output_guard.begin(), output_guard.begin() + kCanaryBytes, [](uint8_t x){return x == kCanary;}) &&
                std::all_of(output_guard.end() - kCanaryBytes, output_guard.end(), [](uint8_t x){return x == kCanary;});
  write_file(output_path, output_guard.data() + kCanaryBytes, output_bytes);
  auto absmax_rounds = time_rounds(launch_absmax, warmup, iterations, rounds, stream);
  launch_absmax(); check(cudaStreamSynchronize(stream), "alpha before pack timing");
  auto pack_rounds = time_rounds(launch_pack, warmup, iterations, rounds, stream);
  auto quantizer_rounds = time_rounds(launch_quantizer, warmup, iterations, rounds, stream);
  auto gemm_rounds = time_rounds(launch_gemm, warmup, iterations, rounds, stream);
  auto pipeline_rounds = time_rounds(launch_pipeline, warmup, iterations, rounds, stream);
  std::ofstream report(report_path, std::ios::trunc); if (!report) throw std::runtime_error("cannot create report");
  report << "{\n  \"passed\": " << (canary ? "true" : "false") << ",\n  \"alpha_a\": " << std::setprecision(9) << host_alpha_a
         << ",\n  \"alpha_b\": " << header.alpha_b << ",\n  \"logical_shape\": [" << header.m << ',' << header.n << ',' << header.k
         << "],\n  \"padded_shape\": [" << header.mp << ',' << header.np << ',' << header.kp
         << "],\n  \"native_scale_bytes\": " << native_as_bytes << ",\n  \"output_canary_ok\": " << (canary ? "true" : "false")
         << ",\n  \"quantizer_kernel_launches\": 3,\n  \"gemm_output\": \"bf16\",\n  \"absmax_alpha_round_ms\": "; render(report, absmax_rounds);
  report << ",\n  \"quantize_pack_round_ms\": "; render(report, pack_rounds);
  report << ",\n  \"full_quantizer_round_ms\": "; render(report, quantizer_rounds);
  report << ",\n  \"gemm_bf16_round_ms\": "; render(report, gemm_rounds);
  report << ",\n  \"full_pipeline_round_ms\": "; render(report, pipeline_rounds);
  report << "\n}\n";
  check(cuModuleUnload(module), "cuModuleUnload"); check(cudaStreamDestroy(stream), "cudaStreamDestroy"); return canary ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) try {
  auto values = options(argc, argv); int warmup = std::stoi(values.count("warmup") ? values.at("warmup") : "1");
  int iterations = std::stoi(values.count("iterations") ? values.at("iterations") : "1"); int rounds = std::stoi(values.count("rounds") ? values.at("rounds") : "1");
  std::string format_text = required(values, "format"); nq::Format format = format_text == "e2m1" ? nq::Format::kE2M1 : format_text == "e0m3" ? nq::Format::kE0M3 : throw std::runtime_error("bad format");
  auto source = read_file(required(values, "activation")); auto static_input = read_file(required(values, "static-input"));
  std::string dtype = required(values, "dtype");
  if (dtype == "bf16") return run<__nv_bfloat16>(source, static_input, format, required(values, "cubin"), required(values, "output"), required(values, "report"), warmup, iterations, rounds);
  if (dtype == "fp16") return run<__half>(source, static_input, format, required(values, "cubin"), required(values, "output"), required(values, "report"), warmup, iterations, rounds);
  throw std::runtime_error("dtype must be bf16 or fp16");
} catch (std::exception const& error) { std::cerr << "pipeline_runner: " << error.what() << '\n'; return 1; }
