"""roofline 模型的测试。

这些测试锁的是**物理关系**而不是具体数字：decode 必须是 memory-bound、
prefill 的 batch 越大越偏向 compute-bound、模型越大越慢。
具体数字会随着实测参数更新而变，锁死它们只会带来无意义的测试失败。
"""

from __future__ import annotations

import pytest

from bench.roofline import (
    RK3588_CPU_A76x4,
    RK3588_NPU,
    HardwareSpec,
    ModelSpec,
    roofline,
)

QWEN_1_5B_Q4 = ModelSpec(
    name="Qwen2.5-1.5B-Q4",
    params=1.54e9,
    weight_bits=4,
    n_layers=28,
    n_kv_heads=2,
    head_dim=128,
)


def test_weight_bytes_includes_quant_overhead():
    # 1.54e9 个权重 * 4bit = 770 MB，加 12.5% 的 scale 开销
    assert QWEN_1_5B_Q4.weight_bytes == pytest.approx(1.54e9 / 2 * 1.125)


def test_decode_is_memory_bound_on_both_paths():
    """decode 阶段在 CPU 和 NPU 上都必须是带宽受限的。

    这是整个项目的核心论点：NPU 的 6 TOPS 在 decode 阶段派不上用场，
    因为瓶颈在把权重从 DRAM 搬进来，而两条路径共享同一套内存。
    如果这个断言挂了，要么模型参数错了，要么硬件参数错了。
    """
    for hw in (RK3588_CPU_A76x4, RK3588_NPU):
        r = roofline(QWEN_1_5B_Q4, hw, "decode", context_len=512)
        assert r.bound_by == "memory", f"{hw.name} 的 decode 居然不是带宽受限"


def test_npu_decode_advantage_is_small():
    """NPU 在 decode 上相对 CPU 的优势应当很有限（因为都卡带宽）。

    对比之下 prefill 的差距会大得多。这个对比是项目要用数据回答的核心问题。
    """
    cpu = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "decode", context_len=512)
    npu = roofline(QWEN_1_5B_Q4, RK3588_NPU, "decode", context_len=512)
    ratio = npu.ceiling_tok_s / cpu.ceiling_tok_s
    assert ratio < 1.5, "两条路径共享 DRAM，decode 上限不该差出 1.5 倍"


def test_prefill_batching_shifts_toward_compute_bound():
    """batch 越大，权重复用越多，越往算力受限那边走。"""
    small = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "prefill", batch_tokens=1)
    large = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "prefill", batch_tokens=512)
    assert small.bound_by == "memory"
    assert large.bound_by == "compute"
    assert large.ceiling_tok_s > small.ceiling_tok_s


def test_prefill_npu_beats_cpu_substantially():
    """prefill 是 compute-bound，NPU 的算力优势应当能体现出来。"""
    cpu = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "prefill", batch_tokens=512)
    npu = roofline(QWEN_1_5B_Q4, RK3588_NPU, "prefill", batch_tokens=512)
    assert npu.ceiling_tok_s > cpu.ceiling_tok_s * 2


def test_bigger_model_is_slower():
    small = ModelSpec(name="s", params=5e8, weight_bits=4)
    big = ModelSpec(name="b", params=7e9, weight_bits=4)
    rs = roofline(small, RK3588_CPU_A76x4, "decode")
    rb = roofline(big, RK3588_CPU_A76x4, "decode")
    assert rs.ceiling_tok_s > rb.ceiling_tok_s


def test_lower_bits_is_faster_in_decode():
    """decode 受带宽限制，位宽减半速度应当接近翻倍。"""
    q4 = ModelSpec(name="q4", params=1.5e9, weight_bits=4, quant_overhead=0.0)
    q8 = ModelSpec(name="q8", params=1.5e9, weight_bits=8, quant_overhead=0.0)
    r4 = roofline(q4, RK3588_CPU_A76x4, "decode")
    r8 = roofline(q8, RK3588_CPU_A76x4, "decode")
    assert r4.ceiling_tok_s == pytest.approx(r8.ceiling_tok_s * 2, rel=1e-6)


def test_kv_cache_costs_bandwidth_at_long_context():
    """长上下文时 KV cache 的读取开销必须被计入。"""
    short = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "decode", context_len=128)
    long = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "decode", context_len=32768)
    assert long.ceiling_tok_s < short.ceiling_tok_s
    assert long.detail["kv_mib"] > short.detail["kv_mib"]


def test_kv_bytes_zero_when_shape_unknown():
    m = ModelSpec(name="x", params=1e9, weight_bits=4)
    assert m.kv_bytes_per_token() == 0.0


def test_decode_ignores_batch_tokens():
    """decode 阶段一次就是 1 个 token，传别的值不该改变结果。"""
    a = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "decode", batch_tokens=1)
    b = roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, "decode", batch_tokens=64)
    assert a.ceiling_tok_s == pytest.approx(b.ceiling_tok_s)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"phase": "train"}, "phase"),
        ({"phase": "prefill", "batch_tokens": 0}, "batch_tokens"),
    ],
)
def test_invalid_args_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        roofline(QWEN_1_5B_Q4, RK3588_CPU_A76x4, **kwargs)


def test_efficiency_scales_ceiling_linearly():
    base = HardwareSpec(name="t", peak_gops=1000, peak_bandwidth_gbps=30,
                        efficiency_bandwidth=1.0)
    half = HardwareSpec(name="t", peak_gops=1000, peak_bandwidth_gbps=30,
                        efficiency_bandwidth=0.5)
    m = ModelSpec(name="m", params=1e9, weight_bits=4)
    assert (roofline(m, base, "decode").ceiling_tok_s
            == pytest.approx(roofline(m, half, "decode").ceiling_tok_s * 2))
