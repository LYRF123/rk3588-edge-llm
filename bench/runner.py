"""跑一组 benchmark 并把结果落盘。

设计上的三条硬规矩：

1. **原始时间戳一定落盘。** 指标可以事后重算，时间戳丢了就没了。
2. **合成数据必须隔离。** mock backend 的结果只能写进 ``results/synthetic/``，
   写正式目录会直接抛错。benchmark 仓库一旦混进假数字就失去全部意义。
3. **失败不静默。** backend 起不来就记 ``unavailable`` 并写明原因，
   不要跳过后假装这一格"没测"。
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import Backend, BackendUnavailable
from .metrics import Metrics, compute_metrics
from .sysmon import Sampler

__all__ = ["RunSpec", "RunResult", "run_one", "run_suite", "SYNTHETIC_DIR"]

#: 合成结果只能落在这里。
SYNTHETIC_DIR = "synthetic"


@dataclass
class RunSpec:
    """一次测量的完整描述。

    Attributes:
        name: 这次测量的标识，会成为结果文件名的一部分。
        prompt: 输入文本。
        max_tokens: 生成上限。
        repeats: 重复次数。取所有轮次的中位数作为代表值。
        warmup: 预热轮数，其结果被丢弃。第一轮总是偏慢（页缓存冷、
            权重还没被读进 page cache），不预热的数字没有可比性。
        cooldown_s: 每轮之间的等待秒数。想测"冷机峰值性能"就设大一点；
            想测"持续负载下的稳态性能"就设 0。两个都要测。
    """

    name: str
    prompt: str
    max_tokens: int = 128
    repeats: int = 3
    warmup: int = 1
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError(f"repeats 必须 >= 1，收到 {self.repeats}")
        if self.warmup < 0:
            raise ValueError(f"warmup 必须 >= 0，收到 {self.warmup}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {self.max_tokens}")


@dataclass
class RunResult:
    """一次测量的结果。``status`` 为 ``ok`` 时 metrics 才有值。"""

    spec_name: str
    backend: str
    path: str
    status: str  # ok | unavailable | error
    metrics: dict[str, Any] | None = None
    all_repeats: list[dict[str, Any]] = field(default_factory=list)
    sysmon: dict[str, Any] = field(default_factory=dict)
    backend_info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    synthetic: bool = False


def _median_metrics(metrics_list: list[Metrics]) -> dict[str, Any]:
    """按 decode_tok_s 取中位数的那一轮作为代表。

    刻意不对各个字段分别取中位数 —— 那样会拼出一个现实中不存在的组合
    （比如 TTFT 来自第 1 轮、p99 来自第 3 轮）。取"代表轮"更诚实。
    """
    ordered = sorted(metrics_list, key=lambda m: m.decode_tok_s)
    return ordered[len(ordered) // 2].as_row()


def collect_env() -> dict[str, Any]:
    """采集运行环境信息，用于日后复现。"""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # SoC 型号：RK 平台在设备树里有，x86 上没有。
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            info["soc_model"] = Path(p).read_text().strip("\x00").strip()
            break
        except OSError:
            continue
    try:
        info["kernel"] = subprocess.run(
            ["uname", "-r"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def run_one(backend: Backend, spec: RunSpec) -> RunResult:
    """在一个 backend 上执行一个 RunSpec。

    Args:
        backend: 已构造但**尚未 load** 的 backend。
        spec: 测量描述。

    Returns:
        RunResult。backend 不可用或抛异常时 status 不是 ``ok``，
        但函数本身不抛（除非是 RunSpec 自身非法）。
    """
    try:
        backend.load()
    except BackendUnavailable as e:
        return RunResult(
            spec_name=spec.name,
            backend=type(backend).__name__,
            path="?",
            status="unavailable",
            error=str(e),
        )

    info = backend.info()
    metrics_list: list[Metrics] = []
    all_rows: list[dict[str, Any]] = []
    sysmon_summary: dict[str, Any] = {}

    try:
        for i in range(spec.warmup):
            backend.generate(spec.prompt, spec.max_tokens)
            if spec.cooldown_s:
                time.sleep(spec.cooldown_s)

        with Sampler(interval_s=0.5) as sampler:
            for i in range(spec.repeats):
                trace = backend.generate(spec.prompt, spec.max_tokens)
                m = compute_metrics(trace)
                metrics_list.append(m)
                row = m.as_row()
                row["repeat"] = i
                all_rows.append(row)
                if spec.cooldown_s and i < spec.repeats - 1:
                    time.sleep(spec.cooldown_s)
        sysmon_summary = sampler.summary()
    except Exception as e:  # noqa: BLE001 - 一个组合失败不该终止整个 sweep
        return RunResult(
            spec_name=spec.name,
            backend=info.name,
            path=info.path,
            status="error",
            error=f"{type(e).__name__}: {e}",
            backend_info=asdict(info),
            synthetic=backend.synthetic,
        )
    finally:
        backend.unload()

    # sysmon 采到的峰值 RSS 比 backend 自报的更可信（覆盖子进程）。
    rep = _median_metrics(metrics_list)
    if sysmon_summary.get("peak_rss_bytes"):
        rep["peak_rss_mib"] = round(sysmon_summary["peak_rss_bytes"] / 2**20, 1)

    return RunResult(
        spec_name=spec.name,
        backend=info.name,
        path=info.path,
        status="ok",
        metrics=rep,
        all_repeats=all_rows,
        sysmon=sysmon_summary,
        backend_info=asdict(info),
        synthetic=backend.synthetic,
    )


def run_suite(
    backends: dict[str, Backend],
    specs: list[RunSpec],
    out_dir: str | Path,
    *,
    tag: str = "",
) -> Path:
    """跑完整个矩阵并写结果 JSON。

    Args:
        backends: ``{标识: backend 实例}``。
        specs: 要跑的测量列表。
        out_dir: 结果目录。含合成 backend 时必须指向 ``.../synthetic`` 之下。
        tag: 追加到文件名里的自定义标签。

    Returns:
        写出的 JSON 文件路径。

    Raises:
        ValueError: 试图把 synthetic 结果写进非 synthetic 目录。
    """
    out_dir = Path(out_dir)
    has_synthetic = any(b.synthetic for b in backends.values())
    if has_synthetic and SYNTHETIC_DIR not in out_dir.parts:
        raise ValueError(
            f"backends 里含合成 backend，结果目录必须在 '{SYNTHETIC_DIR}/' 之下，"
            f"收到 {out_dir}。合成数据和真实测量绝对不能混在一起。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for spec in specs:
        for key, backend in backends.items():
            print(f"[run] {spec.name} × {key} ...", flush=True)
            r = run_one(backend, spec)
            if r.status != "ok":
                print(f"       -> {r.status}: {r.error}", flush=True)
            elif r.metrics:
                print(
                    f"       -> decode {r.metrics['decode_tok_s']} tok/s, "
                    f"TTFT {r.metrics['ttft_ms']} ms",
                    flush=True,
                )
            results.append(r)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"bench-{stamp}{'-' + tag if tag else ''}.json"
    out_path = out_dir / fname
    out_path.write_text(
        json.dumps(
            {
                "env": collect_env(),
                "synthetic": has_synthetic,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] 结果写入 {out_path}", flush=True)
    return out_path
