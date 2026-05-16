# 03 Memory Management

本章目标：理解 CUDA 中 host memory 和 device memory 的关系，掌握显式内存管理的基本流程，并认识统一内存的使用边界。

## 分离的内存空间

CUDA 编程模型把系统看成 host 和 device 两侧：

- host memory：CPU 直接访问，例如 `malloc`、`new`、`std::vector` 管理的内存。
- device memory：GPU 直接访问，例如 `cudaMalloc` 分配的 global memory。

kernel 在 GPU 上执行，它不能把普通 host 指针当成 device 指针随便解引用。常规流程是：

1. host 分配输入输出数组。
2. device 分配对应数组。
3. host 到 device 拷贝输入。
4. launch kernel。
5. device 到 host 拷贝输出。
6. 释放 device 和 host 内存。

## 显式内存 API

### `cudaMalloc`

```cpp
float* d_x = nullptr;
CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
```

`cudaMalloc` 在 device global memory 中分配线性内存。它接收 `void**`，所以 C++ 代码里常见写法是传入指针变量地址。

### `cudaMemcpy`

```cpp
CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
```

第四个参数指定方向：

- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`
- `cudaMemcpyHostToHost`

方向写错可能导致 `invalid argument`，也可能让程序行为难以理解。

### `cudaFree`

```cpp
CHECK_CUDA(cudaFree(d_x));
```

device memory 不会被 `free` 或 `delete` 释放，必须用 `cudaFree`。

### `cudaMemset`

`cudaMemset` 常用于把 device buffer 清零：

```cpp
CHECK_CUDA(cudaMemset(d_y, 0, bytes));
```

注意它按字节写入，适合设为 0；如果要填充 `1.0f` 这类浮点值，通常写 kernel 或用库函数。

## Unified Memory

统一内存使用 `cudaMallocManaged`：

```cpp
float* x = nullptr;
CHECK_CUDA(cudaMallocManaged(&x, bytes));
```

managed memory 可以被 host 和 device 使用，CUDA runtime 会在需要时迁移页面。它能降低入门代码复杂度，但并不代表性能总是最好。对于性能敏感程序，仍然要理解数据何时在哪一侧被访问、是否发生了隐式迁移。

本教程前半部分优先使用显式 `cudaMalloc` 和 `cudaMemcpy`，因为它更能帮助你建立 host/device 边界。

### Prefetch 和 Memory Advice

统一内存可以配合预取降低 page fault：

```cpp
int device = 0;
cudaMemPrefetchAsync(x, bytes, device);
cudaMemPrefetchAsync(y, bytes, device);
```

kernel 结束后，如果 CPU 要大量读取，也可以预取回 CPU：

```cpp
cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId);
```

这不改变统一内存的正确性，只是把“运行时按需迁移”变成“提前迁移”，更适合性能可控的程序。

使用前要先看第二章打印的 `concurrentManagedAccess`。如果它是 `0`，不要默认调用 `cudaMemPrefetchAsync`，示例代码会跳过 prefetch，让 managed memory 走 runtime 的按需迁移路径。

## 带宽意识

一次 SAXPY 对每个元素大致读取 `x`、读取 `y`、写回 `y`，也就是 3 个 float 的 global memory 流量，而计算只有一次乘法和一次加法。它通常是 memory-bound。优化这类算子时，重点是减少不必要的 host/device 拷贝、连续访问内存、合并多个逐元素操作，并用 pinned memory 和 stream 让拷贝与计算重叠。

## 常见错误

- 忘记给输出数组分配 device memory。
- `cudaMemcpy` 的字节数写成元素个数。
- kernel 中使用 host 指针。
- 拷贝方向写反。
- 提前释放 device memory。
- 忘记检查 `cudaMalloc` 返回值，内存不足时继续执行。

## 本章代码

文件：[main.cu](main.cu)

示例包含两个版本的 SAXPY：

```text
y = a * x + y
```

- `saxpyExplicitMemory`：使用 `cudaMalloc`、`cudaMemcpy`、`cudaFree`。
- `saxpyUnifiedMemory`：使用 `cudaMallocManaged`。

两者做同样的计算，并用 CPU 端结果校验。

## 编译运行

```bash
make run-03
```

## 练习

1. 把 `N` 改成更大的值，观察是否会触发内存不足。
2. 故意把一次 `cudaMemcpyHostToDevice` 改成 `cudaMemcpyDeviceToHost`，观察错误检查结果。
3. 给 unified memory 版本加上 `cudaMemPrefetchAsync`，尝试把数据提前迁移到 GPU。
4. 在 explicit memory 版本中用 `cudaMemset` 初始化输出，再比较结果。
5. 估算 SAXPY 的最小内存流量：`n * sizeof(float) * 3`。
