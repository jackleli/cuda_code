#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(err)                                               \
  do {                                                                \
    if ((err) != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n",     \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)

__global__ void scaleKernel(const float* x, float* y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = alpha * x[i];
  }
}

int main() {
  constexpr int n = 1 << 22;
  constexpr int stream_count = 4;
  constexpr float alpha = 3.0f;
  size_t bytes = n * sizeof(float);
  int chunk = (n + stream_count - 1) / stream_count;

  float* h_x = nullptr;
  float* h_y = nullptr;
  CHECK_CUDA(cudaMallocHost(&h_x, bytes));
  CHECK_CUDA(cudaMallocHost(&h_y, bytes));

  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>(i % 100);
    h_y[i] = 0.0f;
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));

  cudaStream_t streams[stream_count];
  for (int s = 0; s < stream_count; ++s) {
    CHECK_CUDA(cudaStreamCreate(&streams[s]));
  }

  int threads = 256;
  for (int s = 0; s < stream_count; ++s) {
    int offset = s * chunk;
    int count = (offset + chunk <= n) ? chunk : (n - offset);
    if (count <= 0) {
      continue;
    }

    size_t chunk_bytes = count * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(d_x + offset, h_x + offset, chunk_bytes,
                               cudaMemcpyHostToDevice, streams[s]));

    int blocks = (count + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, streams[s]>>>(d_x + offset, d_y + offset,
                                                    alpha, count);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_y + offset, d_y + offset, chunk_bytes,
                               cudaMemcpyDeviceToHost, streams[s]));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = alpha * h_x[i];
    if (std::fabs(h_y[i] - expected) > 1e-5f) {
      ok = false;
      std::printf("Mismatch at %d: got %f expected %f\n", i, h_y[i], expected);
      break;
    }
  }
  std::printf("Streamed scale test: %s\n", ok ? "PASSED" : "FAILED");

  for (int s = 0; s < stream_count; ++s) {
    CHECK_CUDA(cudaStreamDestroy(streams[s]));
  }
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFreeHost(h_x));
  CHECK_CUDA(cudaFreeHost(h_y));
  std::printf("08_streams_and_next_steps completed.\n");
  return 0;
}
