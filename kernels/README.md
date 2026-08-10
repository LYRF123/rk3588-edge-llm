# 算子层：Q4_0 GEMV

decode 阶段每生成一个 token，都要把**整份权重**从 DRAM 里读一遍。这件事决定了
端侧 LLM 的速度上限，也决定了算子优化的方向：不是省指令，是别浪费带宽。

这个目录里是一个可独立编译、可独立验证的 Q4_0 × Q8_0 GEMV 内核，用来回答一个
具体问题：**在 RK3588 的 A76 上，这个算子离内存带宽上限还有多远？**

## 为什么是这个算子

decode 时所有权重矩阵乘法都退化成矩阵 × 向量，算术强度约 2 ops/byte。
在 A76 上（约 614 GOPS int8，标称带宽 34.1 GB/s）这个强度对应的平衡点是
约 18 ops/byte —— 我们离它差一个数量级，所以是彻底的 memory-bound。

结论：**内核跑到接近实测 DRAM 带宽，就到头了。**再抠指令序列不会更快，
该去动的是模型本身（更小的模型、更激进的量化）或者访存模式。
这个判断本身比任何具体优化都重要，所以 `bench_gemv` 的输出里把 GB/s 放在
比耗时更显眼的位置。

## 数据格式

沿用 ggml 的分块量化，这样测出来的结论能直接映射到 llama.cpp 的真实表现：

| 类型 | 内容 | 大小 | 有效位宽 |
| --- | --- | --- | --- |
| `block_q4_0` | 32 个权重 → 1×fp16 scale + 16B nibble | 18 B | 4.5 bit/weight |
| `block_q8_0` | 32 个激活 → 1×fp16 scale + 32B int8 | 34 B | 8.5 bit/act |

nibble 排布与 ggml 一致：`qs[j]` 低 4 位是第 `j` 个权重，高 4 位是第 `j+16` 个。
反量化值 `= (nibble - 8) * d`，`-8` 来自 Q4_0 的对称无零点定义。

## 为什么用 SDOT 而不是 SMMLA

Cortex-A76 是 **Armv8.2-A**：有 ASIMD dotprod（`SDOT`/`UDOT`）和 fp16，
**没有** i8mm（`SMMLA`，Armv8.6 才有）也**没有** SVE。

这条实现细节直接决定了很多网上的 ARM 优化经验在 RK3588 上不适用 ——
llama.cpp 里针对 i8mm 的 `Q4_0_8_8` 重排路径在这块芯片上是走不到的，
能用的是 dotprod 的 4×4 路径。编译时必须显式给 `-march=armv8.2-a+dotprod+fp16`：
写 `native` 在交叉编译时无意义，写更高的架构等级会在板子上直接 SIGILL。

单条 `SDOT` 在 128-bit 向量上做 4 组 4-way 点积 = 16 次乘加。
两条 `SDOT` 正好吃掉一个 32 元素的块。

## 分块策略

一次处理 **4 行**。原因：

- 激活向量 `x` 在 4 行之间复用，一次加载摊掉 4 倍加载开销；
- 4 个累加寄存器 + 2 个激活寄存器 + 每行 2 个权重寄存器，仍在 32 个
  v 寄存器以内，不会 spill。

8 行会 spill，A76 上通常反而更慢 —— 但**这一条还没有在真机上验证过**，
上板后用不同的 `rows-per-block` 扫一遍确认，别照抄。

## 用法

```bash
make            # 本机编译（x86 上只有标量路径）
make run        # 编译并跑，默认 8960×1536（Qwen2.5-1.5B 的 FFN 形状）
make aarch64    # 交叉编译到 RK3588，产物 bench_gemv_rk3588
./bench_gemv [rows] [cols] [iters]
```

## 当前验证状态

| 项 | 状态 |
| --- | --- |
| 标量参考实现，x86 本机 | ✅ 通过 |
| NEON dotprod 编译（`-march=armv8.2-a+dotprod+fp16`） | ✅ 无警告通过，objdump 确认 `SDOT` 已生成 |
| NEON dotprod **数值正确性** | ✅ qemu-aarch64 下与标量逐行比对，最大相对误差 `0.000e+00` |
| NEON dotprod **性能** | ❌ 未测 —— 需要真板子 |
| 4 行分块是否最优 | ❌ 未测 |
| 与 llama.cpp 内建内核的对比 | ❌ 未测 |

qemu 是功能模拟，跑出来的耗时和真实硬件没有任何关系，**不要引用 qemu 下的速度数字**。
它能证明的只有一件事：这个内核算得对。

## 一个边界说明

算子优化只能做在 CPU 路径上。NPU 路径的 RKLLM runtime 是闭源的，
拿不到算子级 profile，也没法替换某个算子的实现。详见
[`../docs/06-operator-notes.md`](../docs/06-operator-notes.md)。
