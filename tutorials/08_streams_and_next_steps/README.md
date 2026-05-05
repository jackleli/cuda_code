# 08 Streams And Next Steps

本章目标：认识 CUDA 的异步执行模型、stream、异步拷贝和 events，并给出后续学习路线。

## 默认异步模型

很多 CUDA 操作从 host 视角看是异步的：

- kernel launch 通常只把任务提交给 GPU，然后 host 继续往下执行。
- `cudaMemcpyAsync` 会把拷贝任务提交到 stream。
- 不同 stream 中的任务在条件满足时可能并发执行。

这也是为什么调试阶段经常使用：

```cpp
CHECK_CUDA(cudaDeviceSynchronize());
```

它会等待当前 device 上前面提交的工作完成。

## Stream

stream 是 CUDA work queue。提交到同一个 stream 的任务按顺序执行；提交到不同 stream 的任务可能重叠执行。

常见用途：

- kernel 与 host/device 拷贝重叠。
- 多个独立 kernel 并发。
- 把大数据分 chunk 流水线处理。

创建和销毁：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
...
cudaStreamDestroy(stream);
```

提交到指定 stream：

```cpp
kernel<<<grid, block, shared_bytes, stream>>>(...);
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
```

## Page-locked Host Memory

要让 host/device 异步拷贝更有效，host 侧内存通常需要是 page-locked，也叫 pinned memory：

```cpp
float* h_x = nullptr;
cudaMallocHost(&h_x, bytes);
cudaFreeHost(h_x);
```

普通 pageable host memory 也能用于 `cudaMemcpy`，但异步拷贝和重叠执行通常需要 pinned memory 才能发挥作用。

## Events

event 可以用于：

- 计时。
- 让 host 等待某个 stream 的某个点。
- 建立 stream 间依赖。

例如：

```cpp
cudaEventRecord(event, stream_a);
cudaStreamWaitEvent(stream_b, event);
```

这表示 `stream_b` 后续任务要等 `stream_a` 记录 event 之前的任务完成。

## 本章代码

文件：[main.cu](main.cu)

示例把一个大数组切成多个 chunk，并为每个 chunk 使用一个 stream：

1. pinned host memory 准备输入。
2. `cudaMemcpyAsync` 把 chunk 拷到 device。
3. kernel 处理该 chunk。
4. `cudaMemcpyAsync` 把 chunk 结果拷回 host。
5. `cudaDeviceSynchronize` 等待所有 stream 完成。

这个例子重点是 stream API 的形状。是否能看到明显重叠取决于 GPU 是否支持 copy/execute overlap、数据规模、PCIe 带宽和运行环境。

## 后续学习路线

学完前 8 章后，可以继续补这些主题：

- 更完整的归约：多阶段 device 端归约、warp-level primitives。
- 矩阵乘法：tiling、shared memory、bank conflict。
- 原子操作：`atomicAdd`、计数、直方图。
- 线程束级编程：shuffle、vote、cooperative groups。
- CUDA Graphs：降低重复 launch overhead。
- 多 GPU：device selection、peer access、NCCL。
- 性能工具：Nsight Systems、Nsight Compute、compute-sanitizer。
- 库优先思路：cuBLAS、cuDNN、cuFFT、Thrust、CUB。

## 编译运行

```bash
make run-08
```

## 练习

1. 把 stream 数量改成 1、2、4、8，观察耗时。
2. 把 chunk 大小调大或调小，观察耗时。
3. 把 pinned memory 改成 `std::vector<float>`，比较行为和性能。
4. 用 `cudaEventRecord` 给整个流水线计时。
