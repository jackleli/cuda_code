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

## Unified Memory

统一内存使用 `cudaMallocManaged`：

```cpp
float* x = nullptr;
CHECK_CUDA(cudaMallocManaged(&x, bytes));
```

managed memory 可以被 host 和 device 使用，CUDA runtime 会在需要时迁移页面。它能降低入门代码复杂度，但并不代表性能总是最好。对于性能敏感程序，仍然要理解数据何时在哪一侧被访问、是否发生了隐式迁移。

本教程前半部分优先使用显式 `cudaMalloc` 和 `cudaMemcpy`，因为它更能帮助你建立 host/device 边界。

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
