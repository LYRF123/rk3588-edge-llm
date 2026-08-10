# RK3588 硬件：跟 LLM 推理有关的那些部分

规格书上的数字很多，真正决定 LLM 跑多快的只有几个。这篇只讲那几个。

## 一句话结论

**decode 速度 ≈ 内存带宽 ÷ 模型大小。** 别的都是细节。

## CPU

8 核 big.LITTLE：

| 簇 | 核 | 频率 | Linux 编号 |
| --- | --- | --- | --- |
| big | 4× Cortex-A76 | 最高 2.4 GHz | `cpu4`–`cpu7` |
| LITTLE | 4× Cortex-A55 | 最高 1.8 GHz | `cpu0`–`cpu3` |

**核编号是反直觉的**：`cpu0` 是小核不是大核。不绑核直接跑，调度器会把
计算线程放到小核上，实测会明显偏慢。所以：

```bash
taskset -c 4-7 llama-cli -t 4 ...
```

用满 8 核通常**不如**只用 4 个大核。GEMM 是同步的，一轮结束要等最慢的线程，
A55 会拖住 A76。这是个待实测确认的假设，`configs/sweeps/` 里留了扫描项。

### 指令集：A76 有什么，没有什么

Cortex-A76 是 **Armv8.2-A**：

| 特性 | A76 | 对 LLM 的影响 |
| --- | --- | --- |
| ASIMD dotprod（`SDOT`/`UDOT`） | ✅ 有 | int8 量化点积的主力指令 |
| fp16 算术 | ✅ 有 | fp16 激活可以直接算 |
| **i8mm**（`SMMLA`） | ❌ 无（Armv8.6 才有） | llama.cpp 的 `Q4_0_8_8` 重排路径走不到 |
| **SVE / SVE2** | ❌ 无 | 所有 SVE 优化经验不适用 |

这张表比它看起来重要。网上大量「ARM 上跑 LLM 优化」的经验是基于
Neoverse 或 Apple silicon 的，那些芯片有 i8mm 甚至 SVE。
照抄过来在 RK3588 上要么编不过，要么 SIGILL。

上板后用 `scripts/collect_sysinfo.sh` 确认 `/proc/cpuinfo` 的 Features 里
有 `asimddp`、没有 `i8mm` 和 `sve`。

峰值 int8 算力估算：

```
单条 SDOT = 4 组 4-way 点积 = 16 次乘加 = 32 ops
A76 有 2 条 SIMD 流水线      → 64 ops/cycle/core
64 × 2.4 GHz × 4 核         ≈ 614 GOPS
```

## NPU

RKNPU2，3 个核，官方标称 **6 TOPS int8**（每核 2 TOPS）。

要注意的三件事：

1. **6 TOPS 是 int8 稠密峰值。** LLM 走的 w4a16 是另一条路径，实际利用率未知，
   需要实测。
2. **NPU 和 CPU 共享同一套 DRAM。** 这是本项目最关键的一条：
   NPU 不自带高带宽内存，decode 阶段的瓶颈对两条路径是同一个。
3. **驱动版本决定一切。** RKLLM runtime 对 rknpu 驱动版本有下限要求，
   版本不够会直接失败。用 `cat /sys/kernel/debug/rknpu/version` 查。

## 内存 —— 真正的瓶颈

| 配置 | 标称带宽 |
| --- | --- |
| LPDDR4x-4266（64-bit） | ≈ 34.1 GB/s |
| LPDDR5（视板子而定） | 更高，必须实测 |

**标称带宽不等于可用带宽。** 实测通常只有标称的 45%~60%，也就是
12~20 GB/s。这个折扣来自 refresh、bank 冲突、页面切换等等，
不是可以靠优化拿回来的。

带宽怎么变成 tok/s：

```
Qwen2.5-1.5B，Q4_0 量化：
  权重 = 1.54e9 × 4bit ÷ 8 × 1.125（scale 开销）≈ 826 MiB
  decode 每 token 要读一遍权重 + 全部 KV cache ≈ 840 MiB

  实测带宽 15 GB/s → 15e9 ÷ 840e6 ≈ 17 tok/s   ← 天花板
```

**这个上限跟你用 CPU 还是 NPU 没关系。** 想突破它只有三条路：
减小模型、降低位宽、减少每 token 的读取量（比如投机解码，一次验证多个 token）。

## 板子差异

RK3588 的板子之间差异很大：DDR 型号和频率、散热方案、内核版本、NPU 驱动版本。
脱离这些信息的 tok/s 数字没有可比性。所以每次 benchmark 都要附
`scripts/collect_sysinfo.sh` 的输出。

## 散热

被动散热的小盒子里，持续负载几分钟后 SoC 温度会到 80°C+ 并开始降频。
冷机跑一次和连续跑半小时，速度能差 30%。

因此本仓库的 `sustained-load` 场景刻意不留冷却时间，
并用「降频比」（前 25% token 速率 ÷ 后 25%）把这件事量化出来。
只报冷机峰值的 benchmark 是在美化数据。

## 参考

规格来自 Rockchip 公开资料与 Arm 架构文档。**所有数字都应当上板复核**——
本文件里的值只用于 roofline 的数量级估算，不作为结论引用。
