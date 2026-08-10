"""指标计算的测试。

重点测两类东西：**边界情况**（只有 1 个 token、时间戳非法）和**降频检测**
—— 后者是这个仓库的差异化指标，算错了整个"持续负载"场景就白测了。
"""

from __future__ import annotations

import pytest

from bench.metrics import GenerationTrace, compute_metrics


def make_trace(itls_s: list[float], *, prefill_s: float = 0.5, prompt_tokens: int = 100):
    """按给定的 token 间隔序列造一条时间线。"""
    t0 = 1000.0
    t_first = t0 + prefill_s
    times = [t_first]
    for dt in itls_s:
        times.append(times[-1] + dt)
    return GenerationTrace(
        prompt_tokens=prompt_tokens,
        t_start=t0,
        t_first_token=t_first,
        token_times=times,
    )


def test_ttft_and_prefill_rate():
    trace = make_trace([0.1] * 10, prefill_s=0.5, prompt_tokens=100)
    m = compute_metrics(trace)
    assert m.ttft_ms == pytest.approx(500.0)
    # 100 个 prompt token / 0.5s = 200 tok/s
    assert m.prefill_tok_s == pytest.approx(200.0)


def test_decode_rate_uniform():
    trace = make_trace([0.1] * 20)
    m = compute_metrics(trace)
    assert m.decode_tok_s == pytest.approx(10.0)
    assert m.itl_p50_ms == pytest.approx(100.0)
    assert m.output_tokens == 21


def test_itl_percentiles_catch_stall():
    # 19 个 100ms 的正常间隔 + 1 个 1s 的卡顿。
    # 均值只会被拉到 145ms，但 p99 应该明显暴露出这次卡顿。
    itls = [0.1] * 19 + [1.0]
    m = compute_metrics(make_trace(itls))
    assert m.itl_mean_ms == pytest.approx(145.0, rel=1e-3)
    assert m.itl_p50_ms == pytest.approx(100.0)
    assert m.itl_p99_ms > 500.0, "p99 必须能抓到卡顿，否则这个指标没用"


def test_throttle_ratio_detects_slowdown():
    # 前半段快（100ms），后半段慢一倍（200ms）—— 典型的热降频形状。
    itls = [0.1] * 20 + [0.2] * 20
    m = compute_metrics(make_trace(itls))
    assert m.throttle_ratio is not None
    # 前 25% 平均 100ms，后 25% 平均 200ms，比值 0.5
    assert m.throttle_ratio == pytest.approx(0.5, rel=1e-6)


def test_throttle_ratio_stable_run_is_one():
    m = compute_metrics(make_trace([0.1] * 40))
    assert m.throttle_ratio == pytest.approx(1.0, rel=1e-9)


def test_throttle_ratio_none_when_too_few_samples():
    # 少于 8 个间隔时不给结论，而不是给一个基于 1 个样本的假结论。
    m = compute_metrics(make_trace([0.1] * 5))
    assert m.throttle_ratio is None


def test_single_token_output():
    """只生成 1 个 token 时没有 ITL，不应崩溃。"""
    m = compute_metrics(make_trace([]))
    assert m.output_tokens == 1
    assert m.ttft_ms == pytest.approx(500.0)
    assert m.decode_tok_s != m.decode_tok_s  # nan


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda t: setattr(t, "prompt_tokens", 0), "prompt_tokens"),
        (lambda t: setattr(t, "token_times", []), "token_times"),
        (lambda t: setattr(t, "t_first_token", t.t_start - 1), "t_first_token"),
    ],
)
def test_validate_rejects_bad_traces(mutate, expect):
    trace = make_trace([0.1] * 5)
    mutate(trace)
    with pytest.raises(ValueError, match=expect):
        compute_metrics(trace)


def test_validate_rejects_non_monotonic():
    trace = make_trace([0.1] * 5)
    trace.token_times[3] = trace.token_times[1]  # 时间倒流
    with pytest.raises(ValueError, match="非单调"):
        compute_metrics(trace)


def test_as_row_is_json_friendly():
    m = compute_metrics(make_trace([0.1] * 20))
    row = m.as_row()
    assert set(row) >= {"ttft_ms", "decode_tok_s", "itl_p90_ms", "synthetic"}
    assert all(isinstance(v, (int, float, bool, str, type(None))) for v in row.values())
