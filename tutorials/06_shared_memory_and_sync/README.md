# 06 Shared Memory And Sync

本章目标：理解 shared memory 的用途、`__syncthreads()` 的含义，并实现一个块内归约。

## CUDA 内存层次

入门阶段需要先认识这几类内存：

- registers：每个线程私有，速度最快，数量有限。
- local memory：每个线程私有，但可能落到 device memory，通常不希望频繁使用。
- shared memory：同一个 block 内的线程共享，延迟低，容量有限。
- global memory：所有线程都能访问，容量大，延迟高。
- constant/texture memory：特殊只读访问模式下有用。

shared memory 是很多高性能 CUDA kernel 的核心工具。它常用于：

- 缓存一个 block 会重复使用的数据。
- 让同一 block 内线程协作。
- 减少 global memory 访问次数。
- 改善访存模式。

shared memory 不是“自动更快”的替代品。只有当数据会被复用、需要线程协作，或能改善 global memory 访问模式时，它才值得引入。否则多一次 shared memory 读写和同步反而可能变慢。

## `__shared__`

静态 shared memory：

```cpp
__shared__ float tile[256];
```

动态 shared memory：

```cpp
extern __shared__ float tile[];
kernel<<<grid, block, shared_bytes>>>(...);
```

本章使用静态 shared memory，便于阅读。

## `__syncthreads()`

`__syncthreads()` 是 block 内同步屏障：

- 同一个 block 中所有未退出的线程都必须到达这里。
- 到达后，线程会等待本 block 内其他线程。
- 同步前写入 shared memory 的数据，同步后对本 block 内线程可见。

不能把 `__syncthreads()` 放在会导致同一 block 内部分线程执行、部分线程不执行的分支里，否则可能死锁。

错误示例：

```cpp
if (threadIdx.x == 0) {
  __syncthreads();
}
```

只有 thread 0 到达屏障，其他线程没有到达，程序会挂住。

安全写法是让所有线程都执行到同步点，把条件放在同步点内部工作之外：

```cpp
if (tid < active) {
  shared[tid] = value;
} else {
  shared[tid] = 0.0f;
}
__syncthreads();
```

### Bank Conflict

shared memory 被分成多个 bank。同一个 warp 内如果多个线程访问落在同一个 bank 的不同地址，就可能串行化，称为 bank conflict。入门阶段先记住：

- 连续线程访问连续 `float` 通常比较友好。
- 二维 tile 有时需要 padding，例如 `tile[32][33]`，避免转置访问时冲突。
- Nsight Compute 可以观察 shared memory bank conflict 指标。

## 归约

归约是把很多输入合并成一个输出，例如求和、最大值、最小值。

CPU 写法：

```cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
  sum += x[i];
}
```

GPU 不能简单让所有线程同时写同一个 `sum`，因为会发生 data race。常见入门方案：

1. 每个 block 计算一个 partial sum。
2. 每个 block 把 partial sum 写到 `partial[blockIdx.x]`。
3. host 或第二个 kernel 再把 partial sums 合并。

本章示例使用第一种分解方式，并在 host 上合并 partial sums。

### 为什么浮点误差会不同

浮点加法不满足严格结合律。CPU 顺序求和是：

```text
(((x0 + x1) + x2) + ...)
```

GPU 块内二分归约是树形顺序。两者加法顺序不同，末尾舍入误差也可能不同。因此校验归约时通常看绝对误差或相对误差，而不是要求 bitwise 一致。

### Warp-level 原语

更高性能的归约通常会减少 shared memory 和 `__syncthreads()`，在 warp 内使用 `__shfl_down_sync` 等 shuffle 指令。它能让同一个 warp 内线程直接交换寄存器数据。这个主题适合在理解 shared memory 归约后继续学习。

## 本章代码

文件：[main.cu](main.cu)

kernel 内部流程：

1. 每个线程从 global memory 读取一个或多个元素。
2. 把线程局部和写入 shared memory。
3. 使用 `__syncthreads()` 等待所有线程写完。
4. 用二分归约在 shared memory 中合并。
5. thread 0 把本 block 的 partial sum 写回 global memory。

## 编译运行

```bash
make run-06
```

## 练习

1. 把 block size 改成 `128` 或 `512`，确认结果仍然正确。
2. 把求和改成求最大值。
3. 尝试让每个线程处理两个元素，减少 block 数。
4. 思考为什么浮点求和的 GPU 结果和 CPU 顺序求和可能存在微小差异。
5. 用 Nsight Compute 查看 shared memory load/store 和 occupancy。
