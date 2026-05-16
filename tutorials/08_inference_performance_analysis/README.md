# 08 Inference Performance Analysis

本章目标：建立 CUDA 性能分析的第一层框架，学会用 CUDA events 做微基准测试，理解访存合并、occupancy、算术强度，并能批判一次模型推理在 GPU 上的性能。

## 正确性先于性能

CUDA 程序很容易写出“看起来能跑但结果不可靠”的代码。性能优化之前必须确认：

- kernel 有边界判断。
- 所有 CUDA API 都检查返回值。
- kernel launch 后检查错误。
- 输出和 CPU reference 对齐。
- 浮点误差使用合理 tolerance，而不是直接要求完全相等。

推理优化也一样。先固定输入、shape、batch、precision、随机种子和误差标准，再比较性能。没有正确性基线的性能数字没有意义。

## 用 CUDA Events 计时

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

计时建议：

- 先 warm up 一次，避免首次 launch、上下文初始化影响结果。
- 对短 kernel 重复多次再取平均值或最小值。
- 用 `cudaEventRecord` 包住同一个 stream 里的工作。
- 不要把 host 初始化、malloc、printf 混进 kernel 时间。

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

当 `stride` 很大时，同一个 warp 的线程访问分散地址，global memory 吞吐会下降。自定义算子里如果数据 layout 是 NCHW、NHWC、packed、aligned vectorized 等形式，线程映射必须跟 layout 配合，否则同样的计算量可能差很多。

## Occupancy

occupancy 通常指一个 SM 上活跃 warp 数量相对于硬件上限的比例。它受这些因素影响：

- 每个 block 的线程数。
- 每个线程使用的寄存器数量。
- 每个 block 使用的 shared memory。
- GPU 的 SM 资源上限。

occupancy 帮助隐藏延迟，但不是最终目标。一个 kernel 可能 occupancy 不高，但因为数据复用好、寄存器利用充分、tensor cores 吃满，性能仍然很好。优化时要同时看 achieved occupancy、memory throughput、eligible warps per cycle、instruction throughput 和 tensor core utilization。

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

## 算术强度

算术强度是计算量和内存流量的比值：

```text
arithmetic intensity = FLOPs / bytes moved
```

elementwise copy 的算术强度接近 0，所以几乎完全看内存带宽。矩阵乘 `C = A x B` 的数据能被大量复用，算术强度高得多，因此更容易受算力或 tensor cores 限制。判断算子属于哪一类，是优化路线选择的前提。

## 如何批判一次模型推理的 GPU 性能

不要只看“单次推理多少 ms”。一次严肃的性能分析至少要回答四个问题：

1. 端到端延迟花在哪里：CPU、HtoD、GPU kernel、DtoH、后处理、同步等待？
2. GPU 是否真的忙：SM、Tensor Core、memory bandwidth 有没有接近硬件上限？
3. 瓶颈属于 compute bound、memory bound，还是 launch/synchronization bound？
4. 当前结果和同硬件上的理论上限、库 baseline、TensorRT baseline 相差多少？

常用术语和指标：

| 指标/术语 | 含义 | 主要看什么 | 常见工具 |
| --- | --- | --- | --- |
| Latency | 单个请求端到端耗时，通常看 p50/p90/p99 | 服务体验、同步等待、长尾 | Nsight Systems、服务日志 |
| Throughput | 单位时间处理请求数或 token 数 | batch、并发、pipeline 是否吃满 | 压测工具、Nsight Systems |
| FLOPs | 理论计算量，常指模型或算子的浮点操作数 | 算子计算规模 | 模型分析脚本、框架 profiler |
| TFLOP/s | 实测每秒浮点操作数 | `FLOPs / kernel_time` | Nsight Compute、自定义统计 |
| Peak TFLOP/s | GPU 理论峰值算力，区分 FP32/TF32/FP16/BF16/INT8 | 与实测 TFLOP/s 比较 | GPU spec、NVIDIA 文档 |
| MFU | Model FLOPs Utilization，模型 FLOPs 利用率 | `实际模型 FLOP/s / 理论峰值 FLOP/s` | 自定义统计、训练/推理日志 |
| SM Utilization | SM 忙碌程度 | GPU 是否有足够并行工作 | Nsight Systems、Nsight Compute |
| Tensor Core Utilization | Tensor Core 使用率 | GEMM/Conv 是否走到 tensor cores | Nsight Compute |
| Memory Bandwidth | global memory/L2/DRAM 吞吐 | 是否接近显存带宽上限 | Nsight Compute |
| Memory Access Pattern | 访存是否连续、合并、对齐 | coalescing、stride、layout 问题 | Nsight Compute |
| HtoD Memory | Host to Device 拷贝 | 输入上传是否占比过高 | Nsight Systems |
| DtoH Memory | Device to Host 拷贝 | 输出下载、同步点是否过多 | Nsight Systems |
| Kernel Launch Overhead | CPU 发起 kernel 的开销 | 小算子过多、batch 太小 | Nsight Systems |
| Occupancy | SM 上活跃 warp 比例 | 是否能隐藏延迟，但不是越高越好 | Nsight Compute |
| Arithmetic Intensity | 算术强度，FLOPs/Bytes | 判断 compute bound 或 memory bound | Roofline、手算 |
| Compute Bound | 算力受限 | 提升重点是 tensor core、tile、指令效率 | Nsight Compute |
| Memory Bound | 内存受限 | 提升重点是减少读写、融合、layout、缓存 | Nsight Compute |
| Synchronization Bound | 同步/等待受限 | stream、event、CPU/GPU timeline 空洞 | Nsight Systems |

一个实用的分析顺序：

1. 先用 Nsight Systems 看端到端 timeline：确认 HtoD、kernel、DtoH、CPU gap、同步点和 kernel launch 密度。
2. 找出耗时最高的 kernel 或 TensorRT layer，不要凭直觉优化。
3. 对热点 kernel 用 Nsight Compute 看 Roofline、memory throughput、SM/Tensor Core 指标。
4. 如果是 compute bound，优先检查是否使用 FP16/BF16/INT8、tensor cores、cuBLASLt/cuDNN/TensorRT tactic。
5. 如果是 memory bound，优先检查是否有多余中间张量、layout 转换、非合并访存、未融合 elementwise。
6. 如果是 launch/sync bound，优先考虑 TensorRT engine fusion、CUDA Graphs、减少小 kernel、合并后处理、使用 stream。
7. 每次只改一个因素，并记录 batch size、shape、precision、GPU 型号、driver、CUDA/TensorRT 版本。

几个判断例子：

- GPU kernel 时间很短，但请求延迟很高：优先看 CPU preprocess、HtoD/DtoH、同步等待和 launch overhead。
- GEMM 实测 TFLOP/s 很低：检查 shape 是否太小、是否启用 Tensor Core、数据类型是否是 FP16/BF16/TF32/INT8、矩阵维度是否对齐。
- elementwise/activation 占比高：通常不是算力问题，优先考虑 fusion，减少 global memory 中间读写。
- Nsight Systems 上 GPU timeline 有大量空白：可能是 CPU 供给不足、同步过多、stream 使用不当，或 batch 太小。
- 显存带宽接近上限但 SM/Tensor Core 不高：典型 memory bound，继续增加计算优化意义不大。

## 面向推理优化的 CUDA 学习路线

如果目标是做 GPU 算子推理优化、TensorRT plugin 或模型部署性能调优，建议继续补齐这些能力：

- CUDA 执行模型：warp、SM、occupancy、寄存器压力、launch overhead。
- 内存系统：global/shared/register/local/constant memory、coalescing、alignment、bank conflict、L2 cache。
- 数据搬运：pinned memory、async memcpy、stream、event、double buffering、CUDA Graphs。
- 数学和精度：FP32、TF32、FP16、BF16、INT8、累加精度、fast math、误差验证。
- Tensor Core 编程入口：WMMA、MMA、tile layout、矩阵乘分块思想。
- 高性能基础算子：reduction、scan、softmax、layernorm、topk、transpose、gather/scatter。
- Layout 和向量化：NCHW/NHWC、packed tensor、`float4`/`half2`、对齐访问。
- Profiling：Nsight Systems 定位端到端空洞，Nsight Compute 分析单 kernel 瓶颈。
- 调试可靠性：compute-sanitizer、racecheck、初始化检查、边界测试。
- 库和框架接口：cuBLAS/cuBLASLt、cuDNN、CUB、Thrust、TensorRT plugin API。

优先级上，不建议一开始就手写所有算子。工程实践通常是：先用 TensorRT、cuBLASLt、cuDNN 等库拿到强 baseline；只有当 profiler 明确显示某个算子或数据搬运成为瓶颈，再写自定义 CUDA kernel 或 TensorRT plugin。

## 本章代码

文件：[main.cu](main.cu)

示例比较两个 copy kernel：

- `copyCoalesced`：相邻线程访问相邻元素。
- `copyStrided`：相邻线程访问跨 stride 的元素。

它用 CUDA events 测量 kernel 时间，并打印估算带宽。注意：微基准测试受 GPU 型号、驱动、电源状态、时钟、数据规模影响。这里的目标是理解方向，不是得到固定数值。

## 编译运行

```bash
make run-08
```

## 练习

1. 修改 `stride`，观察时间变化。
2. 修改 `threads` 为 128、512，观察时间变化。
3. 用 Nsight Systems 或 Nsight Compute 分析本章程序。
4. 在 copy kernel 中加入一个简单分支，观察性能是否变化。
5. 选一个真实模型推理过程，按本章指标表列出端到端 latency、HtoD、kernel、DtoH 和 throughput。
