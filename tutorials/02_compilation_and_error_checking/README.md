# 02 Compilation And Error Checking

本章目标：理解 `.cu` 文件如何被 `nvcc` 编译，掌握 CUDA Runtime API 的基本错误检查方式，并学会查询当前 GPU 的关键属性。

## `.cu` 文件和 `nvcc`

CUDA C++ 源文件通常使用 `.cu` 后缀。它里面可以同时包含：

- 普通 C++ host code。
- 使用 `__global__`、`__device__`、`__host__` 等 CUDA 扩展的 device code。

`nvcc` 是 CUDA Toolkit 提供的编译驱动。它会把 host code 交给主机 C++ 编译器，把 device code 编译为 GPU 可执行代码或中间表示，再把两部分链接到一个可执行文件中。

常用命令：

```bash
nvcc -std=c++17 -O2 main.cu -o main
```

常见选项：

- `-std=c++17`：指定 C++ 标准。
- `-O2`：开启优化。
- `-g -G`：调试 device code 时使用。`-G` 会关闭很多 device 优化，不适合性能测试。
- `-lineinfo`：保留行号信息，便于 Nsight Compute、cuda-gdb 或 profiler 映射源码。
- `-arch=sm_80`：为指定 compute capability 生成代码。实际值要根据 GPU 选择。

入门阶段可以先不手写 `-arch`，让 `nvcc` 使用默认值。需要性能分析或发布程序时，再明确指定目标架构。

## CUDA Runtime API

本仓库示例主要使用 CUDA Runtime API，也就是 `cuda_runtime.h` 中的函数，例如：

- `cudaGetDeviceCount`
- `cudaGetDeviceProperties`
- `cudaSetDevice`
- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`
- `cudaDeviceSynchronize`

Runtime API 返回值通常是 `cudaError_t`。只要返回值不是 `cudaSuccess`，就应该打印错误信息并停止程序。

## 为什么必须检查错误

CUDA 错误常见来源：

- 没有可用 GPU。
- driver 和 runtime 版本不匹配。
- `cudaMalloc` 申请过多内存。
- `cudaMemcpy` 方向写错。
- kernel launch 配置非法，例如 block 中线程数超过设备限制。
- kernel 内越界访问。

如果不检查错误，程序可能在很后面的 API 调用才暴露问题，定位成本会很高。

推荐写一个宏或小函数：

```cpp
#define CHECK_CUDA(err)                                               \
  do {                                                                \
    if ((err) != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA Error: %s at %s:%d\n",               \
                   cudaGetErrorString(err), __FILE__, __LINE__);      \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)
```

## kernel launch 的错误检查

kernel launch 语法本身不返回 `cudaError_t`：

```cpp
myKernel<<<grid, block>>>(args...);
```

所以通常紧跟两步：

```cpp
myKernel<<<grid, block>>>(args...);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

- `cudaGetLastError()`：检查 launch 参数、符号解析等 launch 阶段错误。
- `cudaDeviceSynchronize()`：等待 GPU 执行完成，检查 kernel 执行期间的异步错误。

性能敏感代码里不会每次 launch 后都同步，因为同步会破坏异步执行；但学习和调试阶段应该先这样写，保证问题尽早暴露。

## GPU 属性

`cudaGetDeviceProperties` 会返回 `cudaDeviceProp`，里面包含很多设备信息。入门阶段重点看：

- `name`：GPU 名称。
- `major` / `minor`：compute capability。
- `multiProcessorCount`：SM 数量。
- `totalGlobalMem`：global memory 总量。
- `sharedMemPerBlock`：每个 block 可用 shared memory 上限。
- `regsPerBlock`：每个 block 可用寄存器资源上限。
- `warpSize`：warp 大小。
- `maxThreadsPerBlock`：每个 block 最大线程数。
- `maxThreadsDim`：block 每个维度最大线程数。
- `maxGridSize`：grid 每个维度最大 block 数。

这些属性会影响 kernel 的 launch 配置和性能上限。

## 本章代码

文件：[main.cu](main.cu)

示例做两件事：

1. 列出所有可见 CUDA device 的关键属性。
2. 启动一个空 kernel，并演示 launch 后错误检查。

## 编译运行

```bash
make run-02
```

或：

```bash
cd tutorials/02_compilation_and_error_checking
nvcc -std=c++17 -O2 main.cu -o main
./main
```

## 练习

1. 打印更多 `cudaDeviceProp` 字段，例如 `clockRate`、`memoryClockRate`。
2. 把 `threadsPerBlock` 改成一个大于 `maxThreadsPerBlock` 的值，观察 `cudaGetLastError` 报什么。
3. 在代码开头调用 `cudaSetDevice(0)` 并检查返回值。
