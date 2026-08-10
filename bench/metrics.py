"""生成过程的时间线记录与指标计算。

端侧场景下，"平均 tok/s"这一个数字会掩盖掉两件要命的事：

1. **prefill 和 decode 混在一起。** 用户感知的是 TTFT（首 token 延迟），
   而 TTFT 几乎全部由 prefill 决定。把两段平均掉等于把问题藏起来。
2. **尾延迟。** RK3588 会热降频，跑到第 200 个 token 时的速度可能只有
   第 10 个的 70%。只看均值看不出降频，要看 ITL 的 p90/p99 和分段速率。

所以这里只记录**原始时间戳**，所有指标都是从时间戳派生出来的。
原始数据落盘，指标随时可以重算。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GenerationTrace", "Metrics", "compute_metrics"]


@dataclass
class GenerationTrace:
    """一次生成的原始时间线。

    时间戳统一用 ``time.perf_counter()``，单位秒，只有相对差值有意义。

    Attributes:
        prompt_tokens: prompt 的 token 数（由 backend 报告，不要自己估）。
        t_start: 调用生成接口的时刻。
        t_first_token: 收到第 1 个输出 token 的时刻。
        token_times: 每个输出 token 到达的时刻，长度 == 输出 token 数，
            且 ``token_times[0] == t_first_token``。
        peak_rss_bytes: 进程峰值常驻内存，None 表示没采到。
        backend: backend 标识。
        model: 模型标识。
        synthetic: 是否为模拟数据。**任何非真机数据必须置 True**，
            report 会据此在报告里打标。
        extra: backend 自带的额外信息（线程数、量化方式等）。
    """

    prompt_tokens: int
    t_start: float
    t_first_token: float
    token_times: list[float] = field(default_factory=list)
    peak_rss_bytes: int | None = None
    backend: str = ""
    model: str = ""
    synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def output_tokens(self) -> int:
        return len(self.token_times)

    def validate(self) -> None:
        """检查时间线自洽性，不自洽就抛错而不是算出个假指标。

        Raises:
            ValueError: 时间戳不单调、首 token 时刻对不上、prompt_tokens 非正。
        """
        if self.prompt_tokens <= 0:
            raise ValueError(f"prompt_tokens 必须 > 0，收到 {self.prompt_tokens}")
        if not self.token_times:
            raise ValueError("token_times 为空，没有可用于计算的输出 token")
        if self.t_first_token < self.t_start:
            raise ValueError("t_first_token 早于 t_start")
        if abs(self.token_times[0] - self.t_first_token) > 1e-9:
            raise ValueError("token_times[0] 必须等于 t_first_token")
        for i in range(1, len(self.token_times)):
            if self.token_times[i] < self.token_times[i - 1]:
                raise ValueError(f"token_times 在第 {i} 项非单调递增")


@dataclass
class Metrics:
    """从 GenerationTrace 派生的指标。时间单位 ms，速率单位 tok/s。"""

    ttft_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    total_s: float
    prompt_tokens: int
    output_tokens: int
    peak_rss_mib: float | None
    # 前 25% 和后 25% 输出 token 的 decode 速率之比。
    # 明显小于 1 说明跑着跑着变慢了 —— 通常是热降频。
    throttle_ratio: float | None
    synthetic: bool

    def as_row(self) -> dict[str, Any]:
        """摊平成一行，供 CSV / markdown 表格使用。"""
        return {
            "ttft_ms": round(self.ttft_ms, 1),
            "prefill_tok_s": round(self.prefill_tok_s, 1),
            "decode_tok_s": round(self.decode_tok_s, 2),
            "itl_p50_ms": round(self.itl_p50_ms, 1),
            "itl_p90_ms": round(self.itl_p90_ms, 1),
            "itl_p99_ms": round(self.itl_p99_ms, 1),
            "peak_rss_mib": (
                round(self.peak_rss_mib, 1) if self.peak_rss_mib is not None else None
            ),
            "throttle_ratio": (
                round(self.throttle_ratio, 3) if self.throttle_ratio is not None else None
            ),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "synthetic": self.synthetic,
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数。``sorted_vals`` 必须已排序且非空，``q`` 取 [0, 1]。"""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def compute_metrics(trace: GenerationTrace) -> Metrics:
    """把一条时间线换算成指标。

    Args:
        trace: 生成时间线，函数内部会先调用 ``trace.validate()``。

    Returns:
        Metrics。

    Raises:
        ValueError: 时间线不自洽（见 ``GenerationTrace.validate``）。
    """
    trace.validate()

    ttft_s = trace.t_first_token - trace.t_start
    # prefill 速率：prompt 的 token 数除以首 token 延迟。
    # 注意这里把"生成第 1 个 token 的那次 decode"也算进了 prefill，
    # 这是行业惯例，且在 prompt 较长时误差可忽略。
    prefill_tok_s = trace.prompt_tokens / ttft_s if ttft_s > 0 else float("inf")

    # inter-token latency：相邻输出 token 的间隔。
    itls_ms = [
        (trace.token_times[i] - trace.token_times[i - 1]) * 1000
        for i in range(1, len(trace.token_times))
    ]

    if itls_ms:
        decode_span_s = trace.token_times[-1] - trace.token_times[0]
        decode_tok_s = (len(trace.token_times) - 1) / decode_span_s if decode_span_s > 0 else float("inf")
        srt = sorted(itls_ms)
        itl_mean = statistics.fmean(itls_ms)
        itl_p50 = _percentile(srt, 0.50)
        itl_p90 = _percentile(srt, 0.90)
        itl_p99 = _percentile(srt, 0.99)
    else:
        # 只生成了 1 个 token，没有间隔可算。
        decode_tok_s = float("nan")
        itl_mean = itl_p50 = itl_p90 = itl_p99 = float("nan")

    # 降频检测：需要足够的样本才有意义，少于 8 个间隔就不给结论。
    throttle_ratio: float | None = None
    if len(itls_ms) >= 8:
        q = len(itls_ms) // 4
        head = statistics.fmean(itls_ms[:q])
        tail = statistics.fmean(itls_ms[-q:])
        # 速率之比 = 间隔的倒数之比 = head / tail
        throttle_ratio = head / tail if tail > 0 else None

    return Metrics(
        ttft_ms=ttft_s * 1000,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        itl_mean_ms=itl_mean,
        itl_p50_ms=itl_p50,
        itl_p90_ms=itl_p90,
        itl_p99_ms=itl_p99,
        total_s=trace.token_times[-1] - trace.t_start,
        prompt_tokens=trace.prompt_tokens,
        output_tokens=trace.output_tokens,
        peak_rss_mib=(
            trace.peak_rss_bytes / 2**20 if trace.peak_rss_bytes is not None else None
        ),
        throttle_ratio=throttle_ratio,
        synthetic=trace.synthetic,
    )
