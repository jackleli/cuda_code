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
