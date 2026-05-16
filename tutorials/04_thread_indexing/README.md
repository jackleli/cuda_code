# 04 Thread Indexing

本章目标：掌握一维和二维数据的线程索引方式，理解为什么几乎每个 kernel 都需要边界保护。

## 一维索引

一维数组最常见的映射是：一个线程处理一个元素。

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
  out[i] = in[i] * 2.0f;
}
```

启动配置：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
kernel<<<blocks, threads>>>(...);
```

这里 `(n + threads - 1) / threads` 是整数向上取整。

## 二维索引

二维数据常见于矩阵、图像、二维网格。CUDA 的 grid 和 block 都可以是三维的，因此可以自然表达二维线程布局：

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
```

kernel 中：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < height && col < width) {
  int idx = row * width + col;
  C[idx] = A[idx] + B[idx];
}
```

`x` 维通常对应列，`y` 维通常对应行。线性下标 `row * width + col` 适用于 C/C++ 的 row-major 数组。

### Row-major 和访存合并

C/C++ 的二维数组通常按行连续存储：

```text
row 0: col 0, col 1, col 2, ...
row 1: col 0, col 1, col 2, ...
```

因此让 `threadIdx.x` 对应连续的 `col` 往往更友好。同一个 warp 内相邻线程如果访问 `idx`、`idx + 1`、`idx + 2`，global memory 更容易合并访问。若把 x/y 映射反了，可能每个相邻线程跨一整行访问，性能会明显变差。

## 为什么要边界保护

当矩阵宽高不能被 block 维度整除时，grid 向上取整后会产生额外线程。例如宽度是 1000、block.x 是 16，则 x 方向需要 63 个 block，共 1008 列线程，最后 8 列线程不对应真实数据。

所以二维 kernel 通常写：

```cpp
if (row < height && col < width) {
  ...
}
```

## 线程映射设计原则

- 先选择最直观的数据映射：一个元素一个线程。
- 让相邻线程访问相邻内存，方便后续实现 coalesced access。
- block 维度优先选 32 的倍数或由 warp 友好的组合构成，例如 `256`、`16x16`、`32x8`。
- 正确性优先于性能，所有越界风险先用边界判断处理。

## Grid-stride Loop

除了“一个线程处理一个元素”，CUDA 代码里还常见 grid-stride loop：

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n;
     i += blockDim.x * gridDim.x) {
  out[i] = in[i] * 2.0f;
}
```

它的好处是 kernel 可以在固定 grid 大小下处理任意长度输入，也便于让每个线程处理多个元素。很多通用 elementwise kernel 会用这种写法。

## 本章代码

文件：[main.cu](main.cu)

示例实现矩阵加法：

```text
C[row, col] = A[row, col] + B[row, col]
```

矩阵尺寸故意选成不能整除 `16x16` block 的大小，帮助你观察边界判断的必要性。

## 编译运行

```bash
make run-04
```

## 练习

1. 把 block 改成 `dim3(32, 8)`，确认结果仍然正确。
2. 把矩阵宽高改成 `1024 x 1024`，比较运行时间。
3. 新增一个矩阵缩放 kernel：`B[row, col] = alpha * A[row, col]`。
4. 把索引改错成 `idx = col * height + row`，观察校验如何失败。
5. 打印 grid 覆盖的逻辑宽高，计算多出来多少边界线程。
