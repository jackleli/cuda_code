# 07 Streams And Async Execution

本章目标：认识 CUDA 的异步执行模型、stream、异步拷贝、pinned memory 和 events，为后续性能分析打基础。

## 默认异步模型

很多 CUDA 操作从 host 视角看是异步的：

- kernel launch 通常只把任务提交给 GPU，然后 host 继续往下执行。
- `cudaMemcpyAsync` 会把拷贝任务提交到 stream。
- 不同 stream 中的任务在条件满足时可能并发执行。

调试阶段经常使用：

```cpp
CHECK_CUDA(cudaDeviceSynchronize());
```

它会等待当前 device 上前面提交的工作完成。更细粒度的同步包括：

- `cudaStreamSynchronize(stream)`：只等待某个 stream。
- `cudaEventSynchronize(event)`：等待某个 event 之前的工作。
- `cudaDeviceSynchronize()`：等待当前 device 上已提交工作，范围最大。

性能代码中应尽量使用更小同步范围，否则会破坏流水线。

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

现代 CUDA 中还要注意默认 stream 语义。legacy default stream 可能和其他 stream 有隐式同步；per-thread default stream 则更利于并发。工程里应明确团队编译选项和 stream 使用约定，避免隐式同步导致性能异常。

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

## CUDA Graphs

当程序反复执行固定形状的同一串 CUDA 操作时，kernel launch overhead 可能变得明显。CUDA Graphs 可以把一串操作捕获成图，再反复 replay，减少 CPU 提交开销。这里先知道它解决的是“重复提交开销”问题，具体优化评估放到第 08 章。

## 本章代码

文件：[main.cu](main.cu)

示例把一个大数组切成多个 chunk，并为每个 chunk 使用一个 stream：

1. pinned host memory 准备输入。
2. `cudaMemcpyAsync` 把 chunk 拷到 device。
3. kernel 处理该 chunk。
4. `cudaMemcpyAsync` 把 chunk 结果拷回 host。
5. `cudaStreamSynchronize` 逐个等待 stream 完成，并用 event 记录流水线耗时。

这个例子重点是 stream API 的形状。是否能看到明显重叠取决于 GPU 是否支持 copy/execute overlap、数据规模、PCIe 带宽和运行环境。

## 编译运行

```bash
make run-07
```

## 练习

1. 把 stream 数量改成 1、2、4、8，观察耗时。
2. 把 chunk 大小调大或调小，观察耗时。
3. 把 pinned memory 改成 `std::vector<float>`，比较行为和性能。
4. 用 `cudaEventRecord` 分别给 HtoD、kernel、DtoH 计时。
5. 把逐个 `cudaStreamSynchronize` 改成 `cudaDeviceSynchronize`，观察语义差异。
