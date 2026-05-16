# 05 Custom Kernel Workflow

本章目标：把一个 CPU 上的逐元素公式改写成 CUDA 自定义 kernel，并形成稳定的开发流程。

## 从问题开始

假设有一个逐元素计算：

```text
z[i] = sigmoid(a * x[i] + b * y[i])
```

其中：

```text
sigmoid(t) = 1 / (1 + exp(-t))
```

这是非常适合 GPU 的任务，因为每个元素之间没有依赖关系。一个自然的 CUDA 映射是：一个 thread 计算一个 `i`。

## 自定义 kernel 的五步法

### 1. 写 CPU 参考实现

先写一个普通 C++ 函数：

```cpp
void fusedSigmoidCpu(...) {
  for (int i = 0; i < n; ++i) {
    z[i] = 1.0f / (1.0f + std::exp(-(a * x[i] + b * y[i])));
  }
}
```

CPU 参考实现的作用不是性能，而是校验 GPU 结果。任何新 kernel 都建议先有 reference。

### 2. 写 kernel 函数签名

```cpp
__global__ void fusedSigmoidKernel(float a, float b, const float* x,
                                   const float* y, float* z, int n)
```

设计参数时注意：

- 输入指针用 `const`。
- 输出指针单独传入。
- 数组长度显式传入，kernel 不知道 host 容器大小。
- 标量参数直接按值传入。

### 3. 写线程索引和边界判断

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
  ...
}
```

绝大多数入门自定义 kernel 都从这两行开始。

### 4. 写 device 端计算

device code 中可以使用 CUDA 支持的数学函数。float 版本建议使用 `expf`，避免不必要的 double 计算：

```cpp
float t = a * x[i] + b * y[i];
z[i] = 1.0f / (1.0f + expf(-t));
```

### 5. launch、同步、校验

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
fusedSigmoidKernel<<<blocks, threads>>>(a, b, d_x, d_y, d_z, n);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

然后把结果拷回 host，与 CPU reference 比较。

### `__device__` helper

当 kernel 内有可复用的小函数时，可以写成 `__device__`：

```cpp
__device__ float sigmoidDevice(float x) {
  return 1.0f / (1.0f + expf(-x));
}
```

这能让 kernel 主体更清晰。简单函数通常会被编译器内联；必要时可以使用 `__forceinline__`，但入门阶段不需要急着加。

### 指针别名和 `__restrict__`

如果编译器知道输入输出指针不会指向同一块内存，可以更大胆地优化。CUDA C++ 中常见写法：

```cpp
__global__ void kernel(const float* __restrict__ x,
                       float* __restrict__ y, int n)
```

`__restrict__` 是一种承诺：这些指针不会互相别名。承诺错误会导致未定义行为，所以只在确实确定时使用。

## 思维转换

写 CUDA kernel 时，最重要的转变是从“一个循环处理所有元素”变成“每个线程只处理自己负责的元素”。

CPU 写法：

```cpp
for (int i = 0; i < n; ++i) {
  z[i] = f(x[i], y[i]);
}
```

CUDA 写法：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
  z[i] = f(x[i], y[i]);
}
```

循环由 grid 中的大量线程共同完成。

## 什么时候不能直接一个元素一个线程

一个元素一个线程适合元素之间独立的计算。如果存在依赖关系，需要重新设计：

- 前缀和：`out[i]` 依赖前面的元素。
- 归约：很多元素合并成一个值。
- stencil：一个元素依赖邻居元素。
- 矩阵乘法：一个输出元素依赖一行和一列的点积。

这些问题仍然能用 CUDA 做，但会涉及 shared memory、同步、分块、warp 原语或库函数。

## 融合算子思想

本章公式把线性组合和 sigmoid 放在一个 kernel 中完成。如果拆成多个 kernel：

1. `tmp[i] = a * x[i] + b * y[i]`
2. `z[i] = sigmoid(tmp[i])`

就会多一次中间结果的 global memory 写入和读取，还会多一次 kernel launch overhead。推理优化中常见的 elementwise fusion、activation fusion，本质上就是减少中间张量读写和 launch 次数。

## 本章代码

文件：[main.cu](main.cu)

示例实现 fused sigmoid：

```text
z[i] = sigmoid(a * x[i] + b * y[i])
```

它包含：

- CPU reference。
- CUDA kernel。
- 显式 device memory 管理。
- 结果误差检查。

## 编译运行

```bash
make run-05
```

## 练习

1. 把公式改成 `z[i] = tanh(a * x[i] + b)`。
2. 新增一个 ReLU kernel：`z[i] = max(0, x[i])`。
3. 把 `threads` 改成 `128`、`512`，确认正确性不变。
4. 把 CPU reference 故意写错，观察校验是否能抓到。
5. 把 sigmoid 计算提取成 `__device__` 函数，并观察结果是否一致。
