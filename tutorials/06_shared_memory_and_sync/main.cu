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

constexpr int kBlockSize = 256;

__global__ void blockReduceSum(const float* x, float* partial, int n) {
  __shared__ float shared[kBlockSize];

  int tid = threadIdx.x;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  float thread_sum = 0.0f;
  if (i < n) {
    thread_sum += x[i];
  }
  if (i + blockDim.x < n) {
    thread_sum += x[i + blockDim.x];
  }
  shared[tid] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = shared[0];
  }
}

int main() {
  constexpr int n = 1 << 20;
  size_t bytes = n * sizeof(float);
  std::vector<float> h_x(n);
  for (int i = 0; i < n; ++i) {
    h_x[i] = 1.0f / static_cast<float>((i % 7) + 1);
  }

  float cpu_sum = 0.0f;
  for (float v : h_x) {
    cpu_sum += v;
  }

  int blocks = (n + kBlockSize * 2 - 1) / (kBlockSize * 2);
  std::vector<float> h_partial(blocks);
  std::printf("Reduction blocks=%d, block size=%d, elements/thread up to 2\n",
              blocks, kBlockSize);

  float* d_x = nullptr;
  float* d_partial = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  blockReduceSum<<<blocks, kBlockSize>>>(d_x, d_partial, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float gpu_sum = 0.0f;
  for (float v : h_partial) {
    gpu_sum += v;
  }

  float abs_error = std::fabs(gpu_sum - cpu_sum);
  std::printf("CPU sum: %.4f\n", cpu_sum);
  std::printf("GPU sum: %.4f\n", gpu_sum);
  std::printf("Abs error: %.4f\n", abs_error);
  std::printf("Reduction test: %s\n", abs_error < 10.0f ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_partial));
  std::printf("06_shared_memory_and_sync completed.\n");
  return 0;
}
