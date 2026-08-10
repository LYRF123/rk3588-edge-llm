# 上板环境准备

> **状态：未验证。** 手上还没有 RK3588 板子，这份清单是按公开资料整理的
> 待办事项，不是走通过的流程。每一步都标了「怎么确认这步真的成了」——
> 上板时按这个顺序做，遇到不符的地方直接改这个文件。

## 0. 先摸清楚这块板子

```bash
bash scripts/collect_sysinfo.sh > results/sysinfo-$(hostname)-$(date +%F).txt
```

重点看四项：

| 项 | 期望 | 不符怎么办 |
| --- | --- | --- |
| `/proc/cpuinfo` 里有 `asimddp` | ✅ 有 | 没有的话 `-march` 要改，dotprod 内核用不了 |
| `/proc/cpuinfo` 里有 `i8mm` | ❌ 应当没有 | 居然有的话说明不是 A76，整篇分析要重做 |
| `/sys/kernel/debug/rknpu/version` | 能读到版本号 | 读不到 → NPU 驱动没起来，NPU 路径免谈 |
| 内存总量 | 8GB 或 16GB | 4GB 的话模型选型要压到 1.5B 以下 |

## 1. 系统

板子出厂固件的内核版本往往偏老，NPU 驱动版本可能不够 RKLLM 用。
两个选择：

- **官方 Debian/Ubuntu 镜像**：省事，驱动一般配套好
- **Armbian**：内核新，但 NPU 驱动要自己确认

判断标准只有一个：`cat /sys/kernel/debug/rknpu/version` 读出来的版本
是否满足所用 RKLLM runtime 的要求。不满足就得刷机或换内核。

## 2. 调频器设成 performance

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq   # 应当接近 2400000
```

不做这一步，测出来的是调频策略的脾气而不是硬件能力。

## 3. 散热

被动散热的小盒子跑不了 `sustained-load` 场景——十分钟内会降到很低的频率。
最低要求是个小风扇。

确认方式：跑 `sustained-load` 场景，看降频比。低于 0.8 就说明散热压不住，
测出来的稳态数字只反映散热方案而不是芯片能力。

## 4. CPU 路径（llama.cpp）

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
bash scripts/build_llama_cpp.sh
export PATH="$HOME/llama.cpp/build-rk3588/bin:$PATH"
```

**怎么确认成了**：`llama-bench -m <model.gguf> -p 64 -n 32` 能跑出数字，
且 ggml 的启动日志里报告 `DOTPROD = 1`。

模型从 HuggingFace 拿现成的 GGUF，或者自己 `llama-quantize` 转。
先用 Qwen2.5-0.5B 这种小的验证链路通不通，再上 1.5B。

## 5. NPU 路径（RKLLM）

比 CPU 路径麻烦，因为工具链是两段式的：

**x86 主机上**（转换）：
```bash
git clone https://github.com/airockchip/rknn-llm
pip install rknn-llm/rkllm-toolkit/rkllm_toolkit-*.whl
python3 scripts/convert_rkllm.py <hf-model> -o model_w4a16.rkllm -q w4a16
```
⚠️ `scripts/convert_rkllm.py` 目前是骨架，需要先核对 toolkit 的 API 签名。

**板子上**（推理）：
- 把 `librkllmrt.so` 放好（`/usr/lib` 或设 `LD_LIBRARY_PATH`）
- 编译 rknn-llm 仓库里的示例程序
- `nm -D librkllmrt.so | grep rkllm_` 看导出的符号，核对 API

**怎么确认成了**：示例程序能加载 `.rkllm` 并输出**通顺的**文本。
注意「能输出」不等于「输出对」——量化配错了照样能跑，只是模型会胡言乱语。
一定要跟 CPU 路径用同样的 prompt 对比一下输出质量。

## 6. 跑通 benchmark

```bash
python3 -m bench sysinfo          # 确认能读到传感器
python3 -m bench roofline         # 确认上限计算合理
python3 -m bench run --mock --out results   # 确认流水线通
```

然后在 `bench/__main__.py` 的 `cmd_run` 里接上真实 backend，把结果写进
`results/`（不是 `results/synthetic/`）。

## 常见坑

| 症状 | 多半是 |
| --- | --- |
| 二进制在板子上 `SIGILL` | `-march` 给高了，A76 没那条指令 |
| 交叉编译的二进制跑不起来 | glibc 版本对不上，改用板上原生编译 |
| RKLLM 加载失败 | NPU 驱动版本低于 runtime 要求 |
| 速度只有预期的一半 | 线程漂到 A55 小核了，加 `taskset -c 4-7` |
| 跑一会儿就变慢 | 热降频，看 `sustained-load` 的降频比 |
| 模型输出乱码 | 量化配置错了，不是性能问题 |
