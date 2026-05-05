#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(err)                                               \
  do {                                                                \
    if ((err) != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n",     \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  constexpr int N = 1 << 20;
  size_t size = N * sizeof(float);

  float* h_A = reinterpret_cast<float*>(std::malloc(size));
  float* h_B = reinterpret_cast<float*>(std::malloc(size));
  float* h_C = reinterpret_cast<float*>(std::malloc(size));

  for (int i = 0; i < N; i++) {
    h_A[i] = std::rand() / (float)RAND_MAX;
    h_B[i] = std::rand() / (float)RAND_MAX;
  }

  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, size));
  CHECK_CUDA(cudaMalloc(&d_B, size));
  CHECK_CUDA(cudaMalloc(&d_C, size));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (std::fabs(h_C[i] - (h_A[i] + h_B[i])) > FLT_EPSILON) {
      correct = false;
      std::printf("Mismatch at index %d: %f != %f\n", i, h_C[i],
                  h_A[i] + h_B[i]);
      break;
    }
  }
  std::printf("Sample: C[0] = %f + %f = %f\n", h_A[0], h_B[0], h_C[0]);
  std::printf("Sample: C[%d] = %f + %f = %f\n", N - 1, h_A[N - 1],
              h_B[N - 1], h_C[N - 1]);
  std::printf("Vector Addition Test: %s\n", correct ? "PASSED" : "FAILED");

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  std::free(h_A);
  std::free(h_B);
  std::free(h_C);

  printf("01_cuda_basics completed.\n");

  return 0;
}
