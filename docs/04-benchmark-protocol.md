# 测量方法学

一个报「平均 47 tok/s」的 benchmark 几乎没有信息量。这篇说明本仓库报什么、
为什么报这些、以及怎么测才算数。

## 报什么

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| **TTFT** | 从提交到收到第 1 个 token 的时间 | 用户唯一直接感知到的延迟 |
| **prefill tok/s** | prompt token 数 ÷ TTFT | 长 prompt 场景（RAG、文档问答）的决定因素 |
| **decode tok/s** | 输出 token 数 ÷ 生成耗时 | 稳态吞吐，受内存带宽限制 |
| **ITL p50/p90/p99** | 相邻 token 间隔的分位数 | 均值会掩盖卡顿 |
| **峰值内存** | 进程峰值 RSS | 4GB 板子上这常常是硬约束 |
| **降频比** | 前 25% token 速率 ÷ 后 25% | 量化热降频 |

### 为什么 prefill 和 decode 必须分开

它们是两种负载。prefill 一次前向处理 N 个 token，权重复用 N 次，算术强度高，
compute-bound；decode 每次只前向 1 个 token，算术强度约 2 ops/byte，
memory-bound。

把两者平均成一个 tok/s，等于把「NPU 在 prefill 上快 8 倍、在 decode 上没优势」
这个最重要的结论抹平掉。

### 为什么要报分位数

19 个 100ms 的正常间隔混进 1 个 1s 的卡顿：

- 均值 = 145 ms，看起来还行
- p99 = 1000 ms，用户明显感觉到卡了一下

端侧设备上这种卡顿很常见（换 DVFS 档位、内存回收、其他进程抢核）。
只报均值就是在掩盖它。

### 为什么要报降频比

RK3588 在被动散热下持续跑几分钟就会降频。冷机跑 128 个 token 得到的数字，
和连续跑 10 分钟的稳态数字可以差 30%。

降频比明显小于 1 就说明测的是「跑着跑着变慢」的过程，此时冷机峰值没有参考价值，
应该以稳态值为准。

## 怎么测

### 1. 先算上限，再测

```bash
python3 -m bench roofline --model qwen2.5-1.5b-q4
```

有了理论上限，实测数字才有解释力：

- 实测**超过**上限 → 测量方法有问题（多半是没数对 token，或者被 cache 骗了）
- 实测 = 上限的 60~80% → 正常，运行时开销就这么多
- 实测 < 上限的 30% → 有明显问题，值得挖

没有上限对照的话，「47 tok/s」这个数字既不能说明好也不能说明差。

### 2. 固定环境

```bash
# 调频器设成 performance，否则测的是调频策略而不是硬件能力
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 绑大核
taskset -c 4-7 ...
```

并记录环境：`bash scripts/collect_sysinfo.sh > results/sysinfo.txt`。
不同板子的 DDR 频率、散热、内核版本都不一样，脱离这些信息的数字没法比较。

### 3. 预热 + 重复

- **预热至少 1 轮。** 第一轮权重还没进 page cache，频率也没爬上来。
- **重复至少 3 轮**，取 decode 速率的**中位数轮次**作为代表。

代表值刻意取「某一整轮」而不是各字段分别取中位数——后者会拼出一个现实中
不存在的组合（TTFT 来自第 1 轮、p99 来自第 3 轮）。

### 4. 三个场景，打在不同瓶颈上

| 场景 | 形状 | 测的是 |
| --- | --- | --- |
| `short-prompt-long-gen` | 短输入 + 256 输出 | 纯 decode，内存带宽 |
| `long-prompt-short-gen` | 长输入 + 64 输出 | 纯 prefill，TTFT |
| `sustained-load` | 512 输出 × 5 轮无冷却 | 热降频，稳态性能 |

### 5. 贪心解码

`--temp 0`。采样引入的随机性会让不同 backend 生成不同长度的输出，
速度就没法直接比了。

## 两条硬规矩

### 原始时间戳必须落盘

`bench/runner.py` 把每个 token 的时间戳写进结果 JSON。
指标可以事后重算，时间戳丢了就永远补不回来了。

### 合成数据物理隔离

mock backend 的结果只能写进 `results/synthetic/`，写正式目录会直接抛 `ValueError`；
生成的报告顶部有警告横幅，表格里每行有 `⚠️合成` 标记。

这条约束由 `tests/test_runner.py::test_run_suite_refuses_synthetic_in_real_results_dir`
守着。理由很简单：一份 benchmark 仓库里混进假数字，整个仓库就失去价值了。

## 还没做但应该做的

- **精度评测。** 速度对比脱离精度是没意义的——w4a16 和 Q4_0 的精度损失不一样，
  更快可能只是因为模型更差了。需要同一套评测集（比如 CMMLU 子集）跑一遍。
- **功耗。** 端侧场景下 J/token 往往比 tok/s 更重要。需要外接功率计，
  或者用板子的 PMIC 采样（如果暴露了的话）。
- **首次加载时间。** 冷启动把 826 MiB 权重从 eMMC 读进内存要多久，
  直接影响交互体验。
