#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(err)                                               \
  do {                                                                \
    if ((err) != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n",     \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)

__global__ void fusedSigmoidKernel(float a, float b, const float* x,
                                   const float* y, float* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float t = a * x[i] + b * y[i];
    z[i] = 1.0f / (1.0f + expf(-t));
  }
}

void fusedSigmoidCpu(float a, float b, const std::vector<float>& x,
                     const std::vector<float>& y, std::vector<float>& z) {
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    float t = a * x[i] + b * y[i];
    z[i] = 1.0f / (1.0f + std::exp(-t));
  }
}

int main() {
  constexpr int n = 1 << 20;
  constexpr float a = 0.75f;
  constexpr float b = -0.25f;
  size_t bytes = n * sizeof(float);

  std::vector<float> h_x(n);
  std::vector<float> h_y(n);
  std::vector<float> h_z(n);
  std::vector<float> h_ref(n);

  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>((i % 101) - 50) / 25.0f;
    h_y[i] = static_cast<float>((i % 53) - 26) / 13.0f;
  }
  fusedSigmoidCpu(a, b, h_x, h_y, h_ref);

  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_z = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));
  CHECK_CUDA(cudaMalloc(&d_z, bytes));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fusedSigmoidKernel<<<blocks, threads>>>(a, b, d_x, d_y, d_z, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_abs_error = fmaxf(max_abs_error, std::fabs(h_z[i] - h_ref[i]));
  }

  std::printf("Max abs error: %.8f\n", max_abs_error);
  std::printf("Fused sigmoid test: %s\n",
              max_abs_error < 1e-6f ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_z));
  std::printf("05_custom_kernel_workflow completed.\n");
  return 0;
}
