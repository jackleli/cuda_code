#include <cuda_runtime.h>

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

__global__ void emptyKernel() {}

int main() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    std::printf("No CUDA device found.\n");
    return 0;
  }

  std::printf("CUDA device count: %d\n", device_count);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::printf("Device %d: %s\n", device, prop.name);
    std::printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("  SM count: %d\n", prop.multiProcessorCount);
    std::printf("  Global memory: %.2f GiB\n",
                prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    std::printf("  Shared memory per block: %zu bytes\n",
                prop.sharedMemPerBlock);
    std::printf("  Registers per block: %d\n", prop.regsPerBlock);
    std::printf("  Warp size: %d\n", prop.warpSize);
    std::printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    std::printf("  Max block dim: (%d, %d, %d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    std::printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0],
                prop.maxGridSize[1], prop.maxGridSize[2]);
    std::printf("  Managed memory: %d\n", prop.managedMemory);
    std::printf("  Concurrent managed access: %d\n",
                prop.concurrentManagedAccess);
    std::printf("  Async engine count: %d\n", prop.asyncEngineCount);
    std::printf("  Memory clock rate: %.2f GHz\n",
                prop.memoryClockRate / 1000000.0);
    std::printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    double theoretical_bandwidth_gbs =
        2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) /
        1.0e9;
    std::printf("  Approx theoretical memory bandwidth: %.2f GB/s\n",
                theoretical_bandwidth_gbs);
  }

  CHECK_CUDA(cudaSetDevice(0));
  emptyKernel<<<1, 1>>>();
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::printf("02_compilation_and_error_checking completed.\n");
  return 0;
}
