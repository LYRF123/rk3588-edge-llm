# NPU 路径：RKLLM

> **状态：骨架。** 还没有板子，本文全部来自公开资料，未经验证。
> `bench/backends/rkllm.py` 和 `scripts/convert_rkllm.py` 里的 `VERIFY:` /
> `TODO:` 标记是上板后的核对清单。

## 和 llama.cpp 的根本差别

| | llama.cpp | RKLLM |
| --- | --- | --- |
| 工具链 | 一体，板上直接量化 | **两段式**：x86 转换 + 板上推理 |
| 源码 | 全开源 | runtime 闭源（`librkllmrt.so`） |
| 模型格式 | GGUF | `.rkllm` |
| 量化 | Q4_0 / Q4_K / Q8_0 … | w4a16 / w8a8 / w4a16_g{32,64,128} |
| 模型支持 | 几乎所有主流模型 | 有限的支持列表 |
| 上下文长度 | 运行时可调 | **转换时固定**，板上改不了 |
| 算子级优化 | 可以 | 不可以 |

「上下文长度转换时固定」这条容易踩坑：选小了后面要重新转，而转换要在 x86 上
跑前向做量化校准，不是几分钟的事。

## 量化方案

| 方案 | 含义 | 与 llama.cpp 的关系 |
| --- | --- | --- |
| `w4a16` | 权重 4bit，激活 fp16 | 最接近 Q4_0，但激活不量化 |
| `w4a16_g{32,64,128}` | 分组 w4a16 | 类似 Q4_K，组越小精度越好体积越大 |
| `w8a8` | 权重和激活都 int8 | llama.cpp 没有直接对应 |

> **不要因为「都是 4bit」就直接对比 w4a16 和 Q4_0 的速度。**
> 两者的精度损失不一样。必须用同一套评测集单独测过精度再谈速度，
> 否则更快可能只是因为模型变差了——那不叫优化。

## 流程

**x86 主机上**（转换）：

```bash
pip install rknn-llm/rkllm-toolkit/rkllm_toolkit-*.whl
python3 scripts/convert_rkllm.py Qwen/Qwen2.5-1.5B-Instruct \
    -o models/qwen2.5-1.5b_w4a16.rkllm -q w4a16 --max-context 2048
```

⚠️ 该脚本目前抛 `NotImplementedError`。原因写在脚本的 docstring 里：
凭记忆拼一个调用序列比留一个明确的 TODO 更糟——它看起来能用，
实际会在某个参数上悄悄出错，而错误的量化配置不会报错，只会让模型胡言乱语。

上板前要用 `help(RKLLM)` 核对三件事：
1. `quantized_dtype` 的合法取值
2. `max_context` 到底叫什么名字、默认值多少
3. `target_platform` 是 `'rk3588'` 还是 `'RK3588'`（大小写敏感）

**板子上**（推理）：需要 `librkllmrt.so` + 编译好的示例程序。
`bench/backends/rkllm.py` 通过子进程驱动它。

## 前置检查

```bash
cat /sys/kernel/debug/rknpu/version     # 驱动版本，低于 runtime 要求会直接失败
ls /sys/class/devfreq/ | grep npu       # devfreq 节点，用于读频率
dmesg | grep -i rknpu                   # 三个核有没有都起来
```

## 待验证的假设

1. **prefill 上 NPU 明显快于 CPU**（roofline 推算约 8 倍，实际预期 3~5 倍）
2. **decode 上两者差距很小**（< 1.5 倍），因为共享 DRAM 带宽
3. **w4a16 的精度优于 Q4_0**（激活保 fp16），代价是体积略大
4. **NPU 路径的功耗低于 CPU 路径**，这在端侧可能比速度更重要

第 2 条是整个项目的核心论点，`tests/test_roofline.py` 里有对应的断言。
如果实测推翻了它，那是个更有意思的结果，要认真查为什么。

## 为什么算子优化做不进去

runtime 闭源，拿不到算子级 profile，也没法替换某个算子的实现。
能调的只有转换期的旋钮（量化方案、分组大小、上下文长度）和运行期的少量参数
（NPU 核数）——这是配置调优，不是算子优化。

所以本仓库的算子工作全部在 CPU 路径上，见 [`06-operator-notes.md`](06-operator-notes.md)。

## 后续：直接绑 so

子进程方案的时间戳里混了管道开销（微秒级，相对毫秒级的 ITL 可忽略）。
用 ctypes 直接绑 `librkllmrt.so` 能在 result callback 里打点，更干净，
还能拿到 runtime 自己统计的 prefill/decode 耗时。

前提是核对 `rkllm.h` 里 `RKLLMParam` / `RKLLMResult` 的完整字段布局——
这个结构体跨版本改过，抄错一个字段就是段错误，而且症状具有迷惑性。
在核对完成之前不要凭记忆写 `ctypes.Structure`。
