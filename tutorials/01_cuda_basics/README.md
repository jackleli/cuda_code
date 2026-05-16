# 01 CUDA Basics

本章目标：建立 CUDA 编程模型的第一层概念，并读懂一个最小但完整的 vector addition 示例。

## 为什么需要 CUDA

CPU 擅长复杂控制流、低延迟响应和通用任务调度。GPU 擅长把同一种操作同时应用到大量数据上，例如向量加法、矩阵运算、图像处理、深度学习张量计算。

CUDA 是 NVIDIA 提供的并行计算平台和编程模型。使用 CUDA C++ 时，一个程序通常分成两部分：

- host code：运行在 CPU 上的普通 C++ 代码。
- device code：运行在 GPU 上的代码，通常写在 kernel 函数里。

CPU 负责准备数据、申请 GPU 内存、发起 kernel、取回结果；GPU 负责用大量线程并行执行 kernel。

## 核心术语

### Host 和 Device

- Host：CPU 以及 CPU 可直接访问的内存。
- Device：GPU 以及 GPU 上的 device memory。

CUDA 编程模型默认 host 和 device 有各自的内存空间。普通 `malloc` 得到的是 host memory，`cudaMalloc` 得到的是 device memory。kernel 在 device 上运行，因此 kernel 访问的数据通常要先拷贝到 device memory。

这个边界是 CUDA 入门最容易混淆的地方。host 指针和 device 指针在 C++ 类型上都可能写成 `float*`，但它们指向的地址空间不同。经验规则是：host 代码直接读写 host 指针，device kernel 直接读写 device 指针，host/device 之间移动数据要用 `cudaMemcpy` 或统一内存相关 API。

### Kernel

kernel 是在 GPU 上由很多线程并行执行的函数。CUDA C++ 用 `__global__` 标记一个 kernel：

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
```

`__global__` 表示：

- 这个函数从 host 端发起调用。
- 这个函数在 device 端执行。
- 调用时要使用 CUDA 的 launch syntax：`kernel<<<grid, block>>>(args...)`。

### Grid、Block、Thread

一次 kernel launch 会创建一个 grid。grid 由多个 block 组成，block 由多个 thread 组成。

```text
kernel launch
└── grid
    ├── block 0
    │   ├── thread 0
    │   ├── thread 1
    │   └── ...
    ├── block 1
    └── ...
```

常见写法：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

这里 `threadsPerBlock = 256` 表示每个 block 有 256 个线程。`blocksPerGrid` 使用向上取整，保证 `N` 个元素都能被覆盖。

### 内置变量

kernel 里可以直接使用 CUDA 提供的内置变量：

- `threadIdx`：当前线程在 block 内的索引。
- `blockIdx`：当前 block 在 grid 内的索引。
- `blockDim`：每个 block 的维度。
- `gridDim`：grid 的维度。
- `warpSize`：一个 warp 的线程数，现代 NVIDIA GPU 通常是 32。

一维数组最常见的全局索引：

```cpp
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

含义是：前面所有 block 已经覆盖了 `blockIdx.x * blockDim.x` 个线程，当前线程在本 block 内偏移 `threadIdx.x`。

### SM、Warp 和 SIMT

SM 是 Streaming Multiprocessor，中文常译为流式多处理器。GPU 不是一个巨大核心，而是由多个 SM 组成。kernel launch 后，grid 里的 block 会被调度到有执行能力的 SM 上。一个 block 在执行期间属于某个 SM；一个 SM 可以同时驻留多个 block，取决于寄存器、shared memory、线程数等资源。

warp 是 GPU 硬件调度线程的基本单位。CUDA 线程在软件上看是一个个 thread，但硬件通常以 32 个线程为一组调度执行，这组线程叫 warp。NVIDIA 的执行模型叫 SIMT：Single Instruction, Multiple Thread。它允许你像写普通标量代码一样写每个线程的逻辑，但硬件会把同一 warp 中的线程一起调度。

需要记住两个结论：

- 正确性层面：先把每个 CUDA thread 当成一个独立执行的“小工人”理解。
- 性能层面：同一个 warp 里的线程最好做相同控制流、访问连续内存，否则会出现分支发散或低效访存。

### Thread、Block、Grid 的职责边界

- thread：最小执行单元，通常处理一个或几个元素。
- block：线程协作单元，同一个 block 内线程可以使用 shared memory 和 `__syncthreads()`。
- grid：一次 kernel launch 的全部工作集合，不同 block 之间默认没有全局同步。

如果算法需要“所有线程完成阶段 A 后再进入阶段 B”，单个 kernel 里通常只能保证 block 内同步；跨 block 同步常见做法是拆成多个 kernel launch，或者使用 cooperative groups 等更高级机制。

## 第一个性能直觉

本章 vector add 每个元素只做一次加法，却要读两个 float、写一个 float，因此更容易受内存带宽限制，而不是算力限制。很多推理算子也类似：

- elementwise add、mul、ReLU、sigmoid 通常是 memory-bound。
- 矩阵乘、卷积通常更可能是 compute-bound 或 tensor-core-bound。
- 性能优化前要先判断瓶颈，否则容易优化错方向。

## 本章代码

文件：[main.cu](main.cu)

执行流程：

1. 在 host 上分配并初始化 `h_A`、`h_B`、`h_C`。
2. 用 `cudaMalloc` 在 device 上分配 `d_A`、`d_B`、`d_C`。
3. 用 `cudaMemcpyHostToDevice` 把输入从 host 拷贝到 device。
4. 启动 `vectorAdd` kernel。
5. 用 `cudaDeviceSynchronize` 等待 kernel 完成，并捕获异步错误。
6. 用 `cudaMemcpyDeviceToHost` 把结果拷回 host。
7. 用 CPU 结果校验 GPU 结果。
8. 释放 host 和 device 内存。

## 编译运行

在仓库根目录：

```bash
make run-01
```

或在本章目录：

```bash
nvcc -std=c++17 -O2 main.cu -o main
./main
```

期望输出包含：

```text
Vector Addition Test: PASSED
01_cuda_basics completed.
```

## 代码阅读重点

### 边界判断

```cpp
if (i < N) {
  C[i] = A[i] + B[i];
}
```

`blocksPerGrid` 往往向上取整，所以实际启动的线程数可能大于 `N`。多出来的线程必须通过 `if (i < N)` 退出，否则会越界访问。

### kernel 错误检查

```cpp
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

kernel launch 可能失败，kernel 执行过程中也可能失败。`cudaGetLastError` 检查 launch 配置等同步可见的错误；`cudaDeviceSynchronize` 等待 device 端执行完成，也能暴露执行期间的错误。

### 为什么用 256 个线程

256 不是固定答案，但它是入门阶段常用的经验值：

- 是 32 的倍数，能整除 warp 大小。
- 不太小，能提供足够并行度。
- 不太大，通常不会过早触及每 block 最大线程数或资源限制。

后续章节会讨论如何用计时和 occupancy 分析更合适的 block size。

### launch 是异步提交

`vectorAdd<<<...>>>()` 从 CPU 角度通常只是提交任务，GPU 可能还没执行完。代码里紧跟 `cudaDeviceSynchronize()` 是为了教学和调试：它让错误尽早出现，也让后续拷贝结果前确定 kernel 已完成。后面学习 stream 时，会进一步利用这种异步特性做流水线。

## 练习

1. 把 `N` 改成 `1000`，观察 `blocksPerGrid` 和边界判断为什么仍然必要。
2. 把 `threadsPerBlock` 改成 `128`、`512`，确认结果仍然正确。
3. 删除 `if (i < N)` 后运行一个不能被 block size 整除的 `N`，观察程序是否报错或结果是否异常。
4. 新增一个 `vectorSub` kernel，实现 `C[i] = A[i] - B[i]`。
5. 打印实际启动线程数 `blocksPerGrid * threadsPerBlock`，比较它和 `N` 的差值。
