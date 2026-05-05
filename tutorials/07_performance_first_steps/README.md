# 07 Performance First Steps

本章目标：建立 CUDA 性能分析的第一层框架，学会用 CUDA events 计时，并理解访存合并、occupancy、分支发散这些关键词。

## 正确性先于性能

CUDA 程序很容易写出“看起来能跑但结果不可靠”的代码。性能优化之前必须确认：

- kernel 有边界判断。
- 所有 CUDA API 都检查返回值。
- kernel launch 后检查错误。
- 输出和 CPU reference 对齐。
- 浮点误差使用合理 tolerance，而不是直接要求完全相等。

## 用 CUDA events 计时

CPU 计时器测 kernel 时经常不准确，因为 kernel launch 默认是异步的。推荐使用 CUDA events：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

events 记录在 CUDA stream 中，能测量 GPU 工作在该 stream 上消耗的时间。

## 访存合并

global memory 延迟高，吞吐依赖访问模式。同一个 warp 中相邻线程访问相邻地址时，硬件可以更高效地合并内存事务。

好模式：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = x[i];
```

差模式：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = x[i * stride];
```

当 `stride` 很大时，同一个 warp 的线程访问分散地址，global memory 吞吐会下降。

## Occupancy

occupancy 通常指一个 SM 上活跃 warp 数量相对于硬件上限的比例。它受这些因素影响：

- 每个 block 的线程数。
- 每个线程使用的寄存器数量。
- 每个 block 使用的 shared memory。
- GPU 的 SM 资源上限。

occupancy 不是越高越好，但过低通常意味着 GPU 没有足够线程隐藏内存延迟。入门阶段的策略：

- block size 先从 128、256、512 试起。
- 避免单个线程使用过多局部数组。
- shared memory 用量要有意识地控制。
- 用 profiler 或 occupancy calculator 做进一步判断。

## 分支发散

同一个 warp 中线程走不同分支时，warp 需要分别执行不同路径，部分线程会暂时 inactive，这叫 branch divergence。

影响较小的情况：

```cpp
if (i < n) {
  ...
}
```

通常只有最后一个 block 或最后几个 warp 受影响。

影响较大的情况：

```cpp
if (x[i] > 0) {
  heavy_path();
} else {
  another_heavy_path();
}
```

如果同一 warp 内条件分布混杂，两条路径都很重，吞吐可能明显下降。

## 本章代码

文件：[main.cu](main.cu)

示例比较两个 copy kernel：

- `copyCoalesced`：相邻线程访问相邻元素。
- `copyStrided`：相邻线程访问跨 stride 的元素。

它用 CUDA events 测量 kernel 时间，并打印估算带宽。

注意：微基准测试受 GPU 型号、驱动、电源状态、时钟、数据规模影响。这里的目标是理解方向，不是得到固定数值。

## 编译运行

```bash
make run-07
```

## 练习

1. 修改 `stride`，观察时间变化。
2. 修改 `threads` 为 128、512，观察时间变化。
3. 用 Nsight Systems 或 Nsight Compute 分析本章程序。
4. 在 copy kernel 中加入一个简单分支，观察性能是否变化。
