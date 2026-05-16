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

__global__ void copyCoalesced(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i];
  }
}

__global__ void copyStrided(const float* x, float* y, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int src = (i * stride) % n;
    y[i] = x[src];
  }
}

float timeKernelCoalesced(const float* d_x, float* d_y, int n, int blocks,
                          int threads, int repeats) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int r = 0; r < repeats; ++r) {
    copyCoalesced<<<blocks, threads>>>(d_x, d_y, n);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / repeats;
}

float timeKernelStrided(const float* d_x, float* d_y, int n, int blocks,
                        int threads, int stride, int repeats) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int r = 0; r < repeats; ++r) {
    copyStrided<<<blocks, threads>>>(d_x, d_y, n, stride);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / repeats;
}

int main() {
  constexpr int n = 1 << 24;
  constexpr int stride = 32;
  size_t bytes = n * sizeof(float);

  std::vector<float> h_x(n);
  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>(i);
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  constexpr int repeats = 20;

  copyCoalesced<<<blocks, threads>>>(d_x, d_y, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  float coalesced_ms =
      timeKernelCoalesced(d_x, d_y, n, blocks, threads, repeats);
  float strided_ms =
      timeKernelStrided(d_x, d_y, n, blocks, threads, stride, repeats);

  double gib = static_cast<double>(bytes * 2) / 1024.0 / 1024.0 / 1024.0;
  std::printf("Averaged over %d launches, stride=%d\n", repeats, stride);
  std::printf("Coalesced copy: %.3f ms, approx %.2f GiB/s\n", coalesced_ms,
              gib / (coalesced_ms / 1000.0));
  std::printf("Strided copy:   %.3f ms, approx %.2f GiB/s\n", strided_ms,
              gib / (strided_ms / 1000.0));

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  std::printf("08_inference_performance_analysis completed.\n");
  return 0;
}
