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

__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

bool validate(const std::vector<float>& x, const std::vector<float>& y,
              float a) {
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    float expected = a * x[i] + 1.0f;
    if (std::fabs(y[i] - expected) > 1e-5f) {
      std::printf("Mismatch at %d: got %f expected %f\n", i, y[i], expected);
      return false;
    }
  }
  return true;
}

void saxpyExplicitMemory(int n, float a) {
  std::vector<float> h_x(n);
  std::vector<float> h_y(n, 1.0f);

  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>(i % 100) * 0.5f;
  }

  size_t bytes = n * sizeof(float);
  float* d_x = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(a, d_x, d_y, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  std::printf("Explicit memory SAXPY: %s\n",
              validate(h_x, h_y, a) ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
}

void saxpyUnifiedMemory(int n, float a) {
  float* x = nullptr;
  float* y = nullptr;
  size_t bytes = n * sizeof(float);
  CHECK_CUDA(cudaMallocManaged(&x, bytes));
  CHECK_CUDA(cudaMallocManaged(&y, bytes));

  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 100) * 0.5f;
    y[i] = 1.0f;
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(a, x, y, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = a * x[i] + 1.0f;
    if (std::fabs(y[i] - expected) > 1e-5f) {
      ok = false;
      break;
    }
  }
  std::printf("Unified memory SAXPY: %s\n", ok ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
}

int main() {
  constexpr int n = 1 << 20;
  constexpr float a = 2.0f;
  saxpyExplicitMemory(n, a);
  saxpyUnifiedMemory(n, a);
  std::printf("03_memory_management completed.\n");
  return 0;
}
