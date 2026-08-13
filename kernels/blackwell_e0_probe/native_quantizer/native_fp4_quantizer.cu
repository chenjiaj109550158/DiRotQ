#include "native_fp4_quantizer.cuh"

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
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nq = blackwell_native_quantizer;

namespace {

constexpr size_t kCanaryBytes = 64;
constexpr uint8_t kCanary = 0xa5;

void check(cudaError_t result, char const* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
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
  auto iterator = values.find(key);
  if (iterator == values.end()) throw std::runtime_error("missing --" + key);
  return iterator->second;
}

std::vector<uint8_t> read_file(std::string const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open input");
  size_t bytes = size_t(stream.tellg());
  std::vector<uint8_t> result(bytes);
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), bytes);
  if (!stream) throw std::runtime_error("cannot read input");
  return result;
}

void write_file(std::string const& path, void const* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create output");
  stream.write(reinterpret_cast<char const*>(data), bytes);
  if (!stream) throw std::runtime_error("cannot write output");
}

template <class Function>
std::vector<float> time_rounds(Function function, int warmup, int iterations,
                               int rounds, cudaStream_t stream) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate(start)");
  check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  std::vector<float> values;
  for (int round = 0; round < rounds; ++round) {
    check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int index = 0; index < iterations; ++index) function();
    check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    values.push_back(elapsed / iterations);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return values;
}

void render(std::ostream& stream, std::vector<float> const& values) {
  stream << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) stream << ',';
    stream << std::setprecision(9) << values[index];
  }
  stream << ']';
}

template <class Source>
int run(std::vector<uint8_t> const& host_source, int m, int k, int mp, int kp,
        nq::Format format, std::string const& payload_path,
        std::string const& scale_path, std::string const& alpha_path,
        std::string const& report_path, int warmup, int iterations, int rounds) {
  size_t elements = size_t(m) * k;
  size_t source_bytes = elements * sizeof(Source);
  size_t payload_bytes = size_t(mp) * kp / 2;
  size_t scale_bytes = nq::native_scale_bytes(mp, kp);
  if (host_source.size() != source_bytes) throw std::runtime_error("source byte size mismatch");
  cudaStream_t stream;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  Allocation source(source_bytes), maximum(sizeof(unsigned int)), invalid(sizeof(int)), alpha(sizeof(float));
  Allocation guarded_payload(payload_bytes + 2 * kCanaryBytes);
  Allocation guarded_scales(scale_bytes + 2 * kCanaryBytes);
  check(cudaMemcpyAsync(source.pointer, host_source.data(), source_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(source)");
  check(cudaMemsetAsync(guarded_payload.pointer, kCanary, payload_bytes + 2 * kCanaryBytes, stream), "cudaMemsetAsync(payload canary)");
  check(cudaMemsetAsync(guarded_scales.pointer, kCanary, scale_bytes + 2 * kCanaryBytes, stream), "cudaMemsetAsync(scale canary)");
  auto* payload = static_cast<uint8_t*>(guarded_payload.pointer) + kCanaryBytes;
  auto* scales = static_cast<uint8_t*>(guarded_scales.pointer) + kCanaryBytes;
  auto launch_reduction = [&] {
    nq::launch_absmax(static_cast<Source const*>(source.pointer), elements,
                      static_cast<unsigned int*>(maximum.pointer), static_cast<int*>(invalid.pointer), stream);
    nq::launch_finalize(static_cast<unsigned int const*>(maximum.pointer), static_cast<int const*>(invalid.pointer),
                        static_cast<float*>(alpha.pointer), stream);
  };
  auto launch_pack = [&] {
    nq::launch_quantize(static_cast<Source const*>(source.pointer), m, k, mp, kp, format,
                        static_cast<float const*>(alpha.pointer), payload, scales, stream);
  };
  auto launch_full = [&] { launch_reduction(); launch_pack(); };
  launch_full();
  check(cudaGetLastError(), "native quantizer launch");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(correctness)");
  int host_invalid = 0;
  float host_alpha = 0;
  check(cudaMemcpy(&host_invalid, invalid.pointer, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(invalid)");
  check(cudaMemcpy(&host_alpha, alpha.pointer, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(alpha)");
  if (host_invalid || !std::isfinite(host_alpha) || !(host_alpha > 0)) throw std::runtime_error("source contains NaN/Inf or invalid alpha");
  std::vector<uint8_t> payload_guard(payload_bytes + 2 * kCanaryBytes);
  std::vector<uint8_t> scale_guard(scale_bytes + 2 * kCanaryBytes);
  check(cudaMemcpy(payload_guard.data(), guarded_payload.pointer, payload_guard.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(payload)");
  check(cudaMemcpy(scale_guard.data(), guarded_scales.pointer, scale_guard.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(scales)");
  auto canary_ok = [](std::vector<uint8_t> const& value) {
    return std::all_of(value.begin(), value.begin() + kCanaryBytes, [](uint8_t x) { return x == kCanary; }) &&
           std::all_of(value.end() - kCanaryBytes, value.end(), [](uint8_t x) { return x == kCanary; });
  };
  bool payload_canary = canary_ok(payload_guard);
  bool scale_canary = canary_ok(scale_guard);
  write_file(payload_path, payload_guard.data() + kCanaryBytes, payload_bytes);
  write_file(scale_path, scale_guard.data() + kCanaryBytes, scale_bytes);
  write_file(alpha_path, &host_alpha, sizeof(host_alpha));

  launch_reduction();
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(alpha setup)");
  auto reduction_rounds = time_rounds(launch_reduction, warmup, iterations, rounds, stream);
  auto pack_rounds = time_rounds(launch_pack, warmup, iterations, rounds, stream);
  auto full_rounds = time_rounds(launch_full, warmup, iterations, rounds, stream);
  std::ofstream report(report_path, std::ios::trunc);
  if (!report) throw std::runtime_error("cannot create report");
  report << "{\n  \"passed\": " << (payload_canary && scale_canary ? "true" : "false")
         << ",\n  \"format\": \"" << (format == nq::Format::kE2M1 ? "e2m1" : "e0m3")
         << "\",\n  \"dtype\": \"" << (sizeof(Source) == sizeof(__nv_bfloat16) ? "16-bit" : "16-bit")
         << "\",\n  \"shape\": [" << m << ',' << k << "],\n  \"padded_shape\": [" << mp << ',' << kp
         << "],\n  \"alpha\": " << std::setprecision(9) << host_alpha
         << ",\n  \"source_bytes\": " << source_bytes
         << ",\n  \"payload_bytes\": " << payload_bytes
         << ",\n  \"native_scale_bytes\": " << scale_bytes
         << ",\n  \"payload_canary_ok\": " << (payload_canary ? "true" : "false")
         << ",\n  \"scale_canary_ok\": " << (scale_canary ? "true" : "false")
         << ",\n  \"launches\": 3,\n  \"warmup\": " << warmup
         << ",\n  \"iterations\": " << iterations
         << ",\n  \"absmax_alpha_round_ms\": ";
  render(report, reduction_rounds);
  report << ",\n  \"quantize_pack_round_ms\": ";
  render(report, pack_rounds);
  report << ",\n  \"full_quantizer_round_ms\": ";
  render(report, full_rounds);
  report << "\n}\n";
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return payload_canary && scale_canary ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) try {
  auto values = options(argc, argv);
  int m = std::stoi(required(values, "m"));
  int k = std::stoi(required(values, "k"));
  int mp = (m + 15) / 16 * 16;
  int kp = (k + 63) / 64 * 64;
  int warmup = std::stoi(values.count("warmup") ? values.at("warmup") : "1");
  int iterations = std::stoi(values.count("iterations") ? values.at("iterations") : "1");
  int rounds = std::stoi(values.count("rounds") ? values.at("rounds") : "1");
  if (m <= 0 || k <= 0 || warmup < 0 || iterations <= 0 || rounds <= 0) throw std::runtime_error("invalid shape or timing counts");
  std::string format_text = required(values, "format");
  nq::Format format = format_text == "e2m1" ? nq::Format::kE2M1 : format_text == "e0m3" ? nq::Format::kE0M3 : throw std::runtime_error("format must be e2m1 or e0m3");
  std::string dtype = required(values, "dtype");
  auto source = read_file(required(values, "input"));
  auto arguments = std::make_tuple(m, k, mp, kp, format, required(values, "payload-output"),
      required(values, "scale-output"), required(values, "alpha-output"), required(values, "report"),
      warmup, iterations, rounds);
  check(cudaSetDevice(0), "cudaSetDevice");
  if (dtype == "bf16") return std::apply([&](auto&&... args) { return run<__nv_bfloat16>(source, args...); }, arguments);
  if (dtype == "fp16") return std::apply([&](auto&&... args) { return run<__half>(source, args...); }, arguments);
  throw std::runtime_error("dtype must be bf16 or fp16");
} catch (std::exception const& error) {
  std::cerr << "native_fp4_quantizer: " << error.what() << '\n';
  return 1;
}
