"""runner / report 的测试，重点在**合成数据隔离**这条规矩上。

一份 benchmark 仓库里混进合成数字就失去全部价值，所以这条约束必须有测试守着，
而不是只写在文档里靠自觉。
"""

from __future__ import annotations

import json

import pytest

from bench.backends.mock import MockBackend
from bench.report import render_markdown
from bench.roofline import RK3588_CPU_A76x4, ModelSpec
from bench.runner import RunSpec, run_one, run_suite

MODEL = ModelSpec(name="test-1.5b", params=1.5e9, weight_bits=4,
                  n_layers=28, n_kv_heads=2, head_dim=128)


def make_backend():
    return MockBackend(MODEL, RK3588_CPU_A76x4, seed=42, sleep=False)


def test_mock_backend_marks_itself_synthetic():
    b = make_backend()
    assert b.synthetic is True
    b.load()
    trace = b.generate("hello world", 16)
    assert trace.synthetic is True


def test_run_one_produces_metrics():
    r = run_one(make_backend(), RunSpec(name="t", prompt="hi there", max_tokens=32,
                                        repeats=2, warmup=1))
    assert r.status == "ok"
    assert r.metrics is not None
    assert r.metrics["decode_tok_s"] > 0
    assert r.synthetic is True
    assert len(r.all_repeats) == 2


def test_run_one_is_reproducible_with_seed():
    spec = RunSpec(name="t", prompt="hi there", max_tokens=32, repeats=1, warmup=0)
    a = run_one(MockBackend(MODEL, RK3588_CPU_A76x4, seed=7), spec)
    b = run_one(MockBackend(MODEL, RK3588_CPU_A76x4, seed=7), spec)
    assert a.metrics["decode_tok_s"] == b.metrics["decode_tok_s"]


def test_mock_thermal_drift_shows_up_as_throttling():
    """mock 里注入了 15% 的衰减，降频比应当明显小于 1。

    这同时验证了 metrics 的降频检测和 mock 的注入是对得上的。
    """
    r = run_one(
        MockBackend(MODEL, RK3588_CPU_A76x4, seed=1, thermal_drift=0.30, jitter=0.0),
        RunSpec(name="t", prompt="hi there", max_tokens=128, repeats=1, warmup=0),
    )
    assert r.metrics["throttle_ratio"] < 0.95


def test_run_suite_refuses_synthetic_in_real_results_dir(tmp_path):
    """核心约束：合成结果不能写进正式结果目录。"""
    with pytest.raises(ValueError, match="synthetic"):
        run_suite(
            {"mock": make_backend()},
            [RunSpec(name="t", prompt="hi", max_tokens=8, repeats=1, warmup=0)],
            tmp_path / "results",
        )


def test_run_suite_accepts_synthetic_dir(tmp_path):
    out = run_suite(
        {"mock": make_backend()},
        [RunSpec(name="t", prompt="hi", max_tokens=8, repeats=1, warmup=0)],
        tmp_path / "results" / "synthetic",
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["synthetic"] is True
    assert data["results"][0]["status"] == "ok"
    assert "env" in data


def test_report_warns_loudly_about_synthetic_data(tmp_path):
    out = run_suite(
        {"mock": make_backend()},
        [RunSpec(name="t", prompt="hi", max_tokens=8, repeats=1, warmup=0)],
        tmp_path / "results" / "synthetic",
    )
    md = render_markdown(json.loads(out.read_text(encoding="utf-8")))
    assert "WARNING" in md
    assert "不是真机测量结果" in md
    assert "⚠️合成" in md


def test_unavailable_backend_is_recorded_not_swallowed():
    """backend 起不来要留下记录和原因，不能静默跳过。"""
    from bench.backends.llama_cpp import LlamaCppBackend

    r = run_one(
        LlamaCppBackend("/nonexistent/model.gguf", binary="definitely-not-a-real-binary"),
        RunSpec(name="t", prompt="hi", max_tokens=8, repeats=1, warmup=0),
    )
    assert r.status == "unavailable"
    assert r.error and "definitely-not-a-real-binary" in r.error


@pytest.mark.parametrize("field, value", [("repeats", 0), ("warmup", -1), ("max_tokens", 0)])
def test_runspec_validates(field, value):
    kwargs = {"name": "t", "prompt": "hi", field: value}
    with pytest.raises(ValueError, match=field):
        RunSpec(**kwargs)
