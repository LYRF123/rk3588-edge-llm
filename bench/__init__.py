"""RK3588 端侧 LLM benchmark 工具包。

模块划分：

- :mod:`bench.roofline` — 理论上限计算，跑测之前先知道天花板在哪
- :mod:`bench.metrics`  — 时间线记录与 TTFT / ITL / 降频指标
- :mod:`bench.sysmon`   — 温度、频率、内存的后台采样
- :mod:`bench.backends` — CPU（llama.cpp）/ NPU（RKLLM）/ mock 三条路径
- :mod:`bench.runner`   — 编排执行并落盘
- :mod:`bench.report`   — 结果 JSON → Markdown
- :mod:`bench.config`   — YAML 配置解析
"""

from __future__ import annotations

__version__ = "0.1.0"

from .metrics import GenerationTrace, Metrics, compute_metrics
from .roofline import HardwareSpec, ModelSpec, RooflineResult, roofline
from .runner import RunResult, RunSpec, run_one, run_suite

__all__ = [
    "__version__",
    "GenerationTrace",
    "Metrics",
    "compute_metrics",
    "HardwareSpec",
    "ModelSpec",
    "RooflineResult",
    "roofline",
    "RunSpec",
    "RunResult",
    "run_one",
    "run_suite",
]
