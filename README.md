# cuda_code

这是一个从 0 到 1 学习 CUDA C++ 的教程仓库。目标不是把 CUDA 12.5 编程指南逐字翻译一遍，而是把初学者真正需要建立的概念、写 kernel 的步骤、常见坑和最小可运行代码串成一条学习路径。

主参考资料：

- [CUDA 12.5 C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/)

## 学习目标

学完本仓库后，你应该能做到：

- 解释 host、device、kernel、grid、block、thread、warp、SM 的关系。
- 用 `nvcc` 编译和运行 `.cu` 文件。
- 使用 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 管理 device memory。
- 根据一维或二维数据写出正确的线程索引。
- 编写、启动、检查一个自定义 `__global__` kernel。
- 理解共享内存、同步、stream、异步拷贝和 CUDA events。
- 开始用访存合并、occupancy、算术强度、FLOPs/MFU 等指标分析推理性能。

## 目录

```text
tutorials/
  01_cuda_basics/                  CUDA 编程模型和第一个 vector add kernel
  02_compilation_and_error_checking/ nvcc 编译流程、运行时 API、错误检查
  03_memory_management/             host/device 内存、拷贝方向、统一内存
  04_thread_indexing/               一维/二维索引、边界保护、矩阵加法
  05_custom_kernel_workflow/         从 CPU 标量公式改写为自定义 kernel
  06_shared_memory_and_sync/         shared memory、__syncthreads、块内归约
  07_streams_and_async_execution/    streams、异步拷贝、pinned memory、events
  08_inference_performance_analysis/ CUDA 性能分析、访存合并、推理指标与优化路线
```

建议按顺序学习。每章 README 都包含概念、代码阅读重点、编译运行方式和练习。

## 环境要求

- NVIDIA GPU
- NVIDIA Driver
- CUDA Toolkit 12.x，推荐与参考文档一致使用 CUDA 12.5
- Linux shell
- `nvcc --version` 可以正常输出版本信息

检查环境：

```bash
nvidia-smi
nvcc --version
```

## 编译与运行

编译全部示例：

```bash
make
```

运行某一章：

```bash
make run-01
make run-04
```

也可以进入章节目录手动编译：

```bash
cd tutorials/01_cuda_basics
nvcc -std=c++17 -O2 main.cu -o main
./main
```

清理构建产物：

```bash
make clean
```

## 学习方法

1. 先读每章 README 的概念部分，不急着改代码。
2. 编译运行示例，确认输出和 README 描述一致。
3. 从练习题里挑一个修改代码。
4. 每次改 kernel 后都保留 CPU 端校验，先保证正确，再谈性能。
5. 遇到 CUDA API 或 kernel launch 后立刻检查错误，不要把错误留到很后面才定位。

## 官方指南对照

本教程主要对应 CUDA 12.5 编程指南中的这些主题：

- Programming Model: kernels、thread hierarchy、memory hierarchy、compute capability。
- Programming Interface: NVCC、CUDA Runtime、device memory、shared memory、streams、events、error checking。
- Hardware Implementation: SM、SIMT、warp、hardware multithreading。
- Performance Guidelines: utilization、memory throughput、instruction throughput、control flow。
- C++ Language Extensions: `__global__`、`__device__`、`__host__`、built-in variables、synchronization functions、atomic functions。
