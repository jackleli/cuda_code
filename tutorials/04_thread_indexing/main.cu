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

__global__ void matrixAdd(const float* a, const float* b, float* c, int width,
                          int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    int idx = row * width + col;
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  constexpr int width = 1000;
  constexpr int height = 777;
  constexpr int n = width * height;
  size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 17);
    h_b[i] = static_cast<float>(i % 31);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);
  int covered_width = grid.x * block.x;
  int covered_height = grid.y * block.y;
  std::printf("Logical matrix: %d x %d\n", width, height);
  std::printf("Thread coverage: %d x %d, extra logical cells: %d\n",
              covered_width, covered_height,
              covered_width * covered_height - n);
  matrixAdd<<<grid, block>>>(d_a, d_b, d_c, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
      ok = false;
      std::printf("Mismatch at %d\n", i);
      break;
    }
  }

  std::printf("Matrix add grid=(%u,%u) block=(%u,%u): %s\n", grid.x, grid.y,
              block.x, block.y, ok ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  std::printf("04_thread_indexing completed.\n");
  return 0;
}
