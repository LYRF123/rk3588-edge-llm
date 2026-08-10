"""屋顶线（roofline）模型：在跑任何 benchmark 之前先算出理论上限。

端侧 LLM 的 prefill 和 decode 是两种完全不同的负载：

- **prefill**（处理输入 prompt）：一次前向要过 N 个 token，权重被复用 N 次，
  算术强度高 → **compute-bound**，看 GOPS。
- **decode**（逐 token 生成）：每生成 1 个 token 就要把整份权重从 DRAM 读一遍，
  算术强度约等于 2 ops/byte → **memory-bound**，看内存带宽。

这个模块只做一件事：给定模型规格和硬件规格，算出两个上限。
实测值超过上限 = 测量方法有问题；实测值远低于上限 = 有优化空间。
两者都比"跑出来 X tok/s"这个孤零零的数字有信息量。

模块内所有硬件参数都是**标称值**，来源见 `docs/00-hardware.md`。
真实带宽必须实测（`scripts/collect_sysinfo.sh` + STREAM），标称值通常只能达到 40%~60%。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "ModelSpec",
    "HardwareSpec",
    "RooflineResult",
    "RK3588_CPU_A76x4",
    "RK3588_NPU",
    "KNOWN_HARDWARE",
    "roofline",
]

# 一个 token 走一遍权重，每个权重参与 1 次乘 + 1 次加。
OPS_PER_PARAM_PER_TOKEN = 2


@dataclass(frozen=True)
class ModelSpec:
    """描述一个待测模型。

    Attributes:
        name: 模型标识，例如 ``Qwen2.5-1.5B-Instruct``。
        params: 参数量（个，不是 B）。1.5B 就写 ``1.5e9``。
        weight_bits: 权重量化位宽。Q4 类写 4，w8a8 写 8，fp16 写 16。
        n_layers: 层数，用于估算 KV cache。
        n_kv_heads: KV head 数（GQA 之后的，不是 query head 数）。
        head_dim: 每个 head 的维度。
        kv_bits: KV cache 的位宽，默认 fp16。
        quant_overhead: 量化元数据开销系数。Q4_0 每 32 个权重带 1 个 fp16 scale，
            即 16 bit / 32 weight = 0.5 bit/weight，相对 4 bit 是 +12.5%。
    """

    name: str
    params: float
    weight_bits: int
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    kv_bits: int = 16
    quant_overhead: float = 0.125

    @property
    def weight_bytes(self) -> float:
        """权重在 DRAM 中占用的字节数（含量化元数据）。"""
        return self.params * self.weight_bits / 8 * (1 + self.quant_overhead)

    def kv_bytes_per_token(self) -> float:
        """每个 token 的 KV cache 字节数（K 和 V 各一份）。

        缺少 n_layers / n_kv_heads / head_dim 时返回 0，调用方需自行判断
        这只是说明 KV 项被忽略了，不代表 KV cache 真的不占带宽。
        """
        if not (self.n_layers and self.n_kv_heads and self.head_dim):
            return 0.0
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bits / 8


@dataclass(frozen=True)
class HardwareSpec:
    """描述一条执行路径的硬件能力。

    Attributes:
        name: 路径标识，例如 ``RK3588 CPU (A76 x4 @2.4GHz)``。
        peak_gops: 目标精度下的峰值算力（GOPS，10^9 ops/s）。
        peak_bandwidth_gbps: 峰值内存带宽（GB/s，按 10^9 字节算）。
        efficiency_compute: 算力实际可达比例。手写汇编 GEMM 在 ARM 上
            拿到 60%~80% 算不错，这里默认保守的 0.6。
        efficiency_bandwidth: 带宽实际可达比例。LPDDR4x 上纯读流式访问
            通常只有标称的 45%~60%，默认 0.5。
        note: 数据来源或注意事项。
    """

    name: str
    peak_gops: float
    peak_bandwidth_gbps: float
    efficiency_compute: float = 0.6
    efficiency_bandwidth: float = 0.5
    note: str = ""

    @property
    def effective_gops(self) -> float:
        return self.peak_gops * self.efficiency_compute

    @property
    def effective_bandwidth_gbps(self) -> float:
        return self.peak_bandwidth_gbps * self.efficiency_bandwidth


@dataclass
class RooflineResult:
    """屋顶线计算结果，单位统一为 token/s 和 ms。"""

    model: str
    hardware: str
    phase: Literal["prefill", "decode"]
    ceiling_tok_s: float
    bound_by: Literal["compute", "memory"]
    detail: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        bound_cn = {"compute": "算力", "memory": "带宽"}[self.bound_by]
        return (
            f"{self.model} @ {self.hardware} [{self.phase}] "
            f"上限 {self.ceiling_tok_s:.1f} tok/s（受{bound_cn}限制）"
        )


# --- 标称硬件参数 -----------------------------------------------------------
#
# 下面的数字是从公开规格书推算的**标称值**，用来当作"不可能超过"的天花板。
# 上板后请用 scripts/collect_sysinfo.sh 采集实际频率，用 STREAM 实测带宽，
# 然后覆盖这里的默认值 —— 不同板厂的内存颗粒和 DDR 频率差异很大。

# A76 支持 Armv8.2 的 ASIMD dotprod（SDOT/UDOT），不支持 i8mm（那是 Armv8.6）。
# 单条 SDOT 在 128-bit 向量上做 4 组 4-way 点积 = 16 次乘加 = 32 ops。
# A76 有 2 条 SIMD 流水线 → 64 ops/cycle/core。
_A76_OPS_PER_CYCLE_INT8 = 64
_A76_CLOCK_GHZ = 2.4
_A76_CORES = 4

RK3588_CPU_A76x4 = HardwareSpec(
    name="RK3588 CPU (A76 x4 @2.4GHz, int8 dotprod)",
    peak_gops=_A76_OPS_PER_CYCLE_INT8 * _A76_CLOCK_GHZ * _A76_CORES,  # 614.4 GOPS
    # LPDDR4x-4266，64-bit 总线：4266 MT/s * 8 B = 34.1 GB/s。
    # 装 LPDDR5 的板子会更高，必须实测。
    peak_bandwidth_gbps=34.1,
    efficiency_compute=0.6,
    efficiency_bandwidth=0.5,
    note="A76 有 dotprod 无 i8mm/SVE；A55 小核算力低且共享带宽，默认不计入。",
)

RK3588_NPU = HardwareSpec(
    name="RK3588 NPU (RKNPU2, 3 core, int8)",
    peak_gops=6000.0,  # 官方标称 6 TOPS int8（3 core x 2 TOPS）
    peak_bandwidth_gbps=34.1,  # NPU 与 CPU 共享同一套 DRAM，带宽不是额外的
    efficiency_compute=0.5,
    efficiency_bandwidth=0.5,
    note="6 TOPS 是 int8 稠密峰值；LLM 的 w4a16 走的是另一条路径，实际利用率待测。",
)

KNOWN_HARDWARE: dict[str, HardwareSpec] = {
    "cpu": RK3588_CPU_A76x4,
    "npu": RK3588_NPU,
}


def roofline(
    model: ModelSpec,
    hw: HardwareSpec,
    phase: Literal["prefill", "decode"],
    batch_tokens: int = 1,
    context_len: int = 0,
) -> RooflineResult:
    """算出某个模型在某条路径上某个阶段的 token/s 上限。

    Args:
        model: 模型规格。
        hw: 硬件规格。
        phase: ``"prefill"`` 或 ``"decode"``。
        batch_tokens: prefill 时一次前向处理的 token 数。decode 阶段固定为 1，
            传入其他值会被忽略。
        context_len: 当前上下文长度，用于估算 decode 时需要读的 KV cache。

    Returns:
        RooflineResult，其中 ``ceiling_tok_s`` 是 compute 和 memory 两个上限中
        较小的那个。

    Raises:
        ValueError: phase 不是 prefill/decode，或 batch_tokens < 1。
    """
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase 必须是 'prefill' 或 'decode'，收到 {phase!r}")
    if batch_tokens < 1:
        raise ValueError(f"batch_tokens 必须 >= 1，收到 {batch_tokens}")

    if phase == "decode":
        batch_tokens = 1

    # --- 算力上限 ---
    ops_per_token = model.params * OPS_PER_PARAM_PER_TOKEN
    compute_tok_s = hw.effective_gops * 1e9 / ops_per_token

    # --- 带宽上限 ---
    # 一次前向读一遍权重，代价被 batch_tokens 个 token 摊薄。
    bytes_per_forward = model.weight_bytes
    # decode 时还要把已有的 KV cache 全部读一遍。
    kv_bytes = model.kv_bytes_per_token() * context_len
    bytes_per_token = (bytes_per_forward + kv_bytes) / batch_tokens
    memory_tok_s = hw.effective_bandwidth_gbps * 1e9 / bytes_per_token

    if compute_tok_s <= memory_tok_s:
        ceiling, bound = compute_tok_s, "compute"
    else:
        ceiling, bound = memory_tok_s, "memory"

    return RooflineResult(
        model=model.name,
        hardware=hw.name,
        phase=phase,
        ceiling_tok_s=ceiling,
        bound_by=bound,  # type: ignore[arg-type]
        detail={
            "compute_ceiling_tok_s": compute_tok_s,
            "memory_ceiling_tok_s": memory_tok_s,
            "weight_mib": model.weight_bytes / 2**20,
            "kv_mib": kv_bytes / 2**20,
            "bytes_per_token": bytes_per_token,
            "arithmetic_intensity_ops_per_byte": ops_per_token
            * batch_tokens
            / max(bytes_per_forward + kv_bytes, 1.0),
        },
    )
