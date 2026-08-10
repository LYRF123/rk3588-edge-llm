# CPU 路径：llama.cpp

## 为什么 CPU 路径值得认真对待

直觉上「有 NPU 当然用 NPU」。但在 decode 阶段，NPU 的 6 TOPS 派不上用场——
瓶颈是内存带宽，而两条路径共享同一套 DRAM。CPU 路径还有几个实打实的优势：

- **模型选择自由。** GGUF 生态里什么模型都有，RKLLM 支持的模型列表短得多。
- **量化方案成熟。** Q4_K_M 这类分组量化在精度/体积上的平衡比朴素 Q4_0 好。
- **可调试、可优化。** ggml 全开源，能 profile 到算子级。
- **不依赖驱动版本。** 不用担心 rknpu 驱动版本对不上。

所以真实的问题不是「用哪个」，而是「**prefill 走 NPU、decode 走 CPU** 是否可行」。
这是路线图里的最后一项。

## 编译

```bash
bash scripts/build_llama_cpp.sh          # 板上原生编译
bash scripts/build_llama_cpp.sh --cross  # x86 交叉编译
```

关键的编译选项：

```
-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76
```

理由见 [`00-hardware.md`](00-hardware.md)：A76 有 dotprod、没有 i8mm/SVE。
给高了会 SIGILL，给 `native` 在交叉编译时无意义。

**记录 llama.cpp 的 commit hash。** 它的性能在版本之间会明显变化，
不记 commit 的对比数据没有意义。`build_llama_cpp.sh` 会把 commit 打出来。

## 量化方案

| 类型 | 每权重位宽 | 特点 |
| --- | --- | --- |
| `Q4_0` | 4.5 bit | 最简单的对称分块量化，dotprod 内核针对它优化得最好 |
| `Q4_K_M` | ~4.8 bit | 分组量化，同等体积下精度更好 |
| `Q8_0` | 8.5 bit | 精度接近 fp16，但体积翻倍 → decode 速度减半 |

decode 是 memory-bound，**位宽直接决定速度**：Q4 比 Q8 快接近一倍，
这个关系在 `tests/test_roofline.py::test_lower_bits_is_faster_in_decode` 里锁着。

选型时要拿精度换速度，而不是只看速度。

## 关键参数

| 参数 | 建议 | 为什么 |
| --- | --- | --- |
| `-t` | 4 | 对应 4 个 A76 大核 |
| `taskset -c` | `4-7` | 不绑核线程会漂到 A55 小核 |
| `--temp` | 0 | 贪心解码，保证不同 backend 可比 |
| `-c` | 按场景 | 上下文越长 KV cache 越吃带宽 |

## 待验证的假设

这些是基于架构推理的预期，**都还没有真机数据**：

1. **4 个大核 > 8 核混用。** GEMM 同步，A55 会拖住 A76。
2. **Q4_0 的 decode 速度 ≈ Q8_0 的两倍。** roofline 如此，实际会有折扣。
3. **绑核带来的提升是显著的**（>20%），而不是噪声级别。
4. **OpenCL 后端（Mali-G610）在 decode 上没有优势**，因为同样卡带宽，
   而且 GPU 的量化内核成熟度不如 CPU。

`configs/sweeps/` 里留了对应的扫描配置。上板后逐条验证，
验证结果（无论支持还是推翻）直接写回这个文件。

## 接入 benchmark

`bench/backends/llama_cpp.py` 通过子进程流式读 `llama-cli` 的 stdout 打时间戳。

不用 `llama-bench` 的原因：它只给聚合后的 pp/tg 均值，拿不到 ITL 分布，
也就看不出热降频——而那恰恰是端侧最需要暴露的问题。

⚠️ 该文件里有两处 `VERIFY:` 标记，是上板第一件要核对的事：

1. stdout/stderr 的划分（不同版本变过）
2. 「一次 read == 一个 token」这个假设（多字节 UTF-8 会被拆开）

校准方法：用纯 ASCII prompt 跑一次，`token_times` 的长度应当正好等于
stderr 里报告的 eval count。对不上就改用 `llama-server` 的 SSE 流。
