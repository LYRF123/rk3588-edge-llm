"""从 YAML 配置构造模型规格和测量场景。

配置和代码分离的理由很实际：换一个模型、改一次线程数就要动 Python 文件的话，
sweep 的历史记录会全部混在 git diff 里，事后没法追溯"当时到底测的什么"。
配置文件跟结果 JSON 一起归档，才谈得上可复现。

YAML 解析走 ``yaml.safe_load``。没装 PyYAML 时退化为只支持 JSON，
这样 CI 的最小环境不需要额外依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .roofline import KNOWN_HARDWARE, HardwareSpec, ModelSpec
from .runner import RunSpec

__all__ = ["load_config", "models_from_config", "specs_from_config", "hardware_from_config"]


def _load_mapping(path: str | Path) -> dict[str, Any]:
    """读 YAML 或 JSON，返回顶层映射。

    Raises:
        FileNotFoundError: 文件不存在。
        RuntimeError: 是 YAML 但没装 PyYAML。
        ValueError: 顶层不是映射。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 - 可选依赖，按需导入
        except ImportError as e:
            raise RuntimeError(
                f"读取 {p} 需要 PyYAML：pip install pyyaml（或把配置改成 .json）"
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"{p} 的顶层必须是映射，实际是 {type(data).__name__}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """读取配置文件。"""
    return _load_mapping(path)


def models_from_config(cfg: dict[str, Any]) -> dict[str, ModelSpec]:
    """从配置里的 ``models`` 段构造 ModelSpec。

    期望的结构::

        models:
          qwen2.5-1.5b-q4:
            name: Qwen2.5-1.5B-Instruct
            params: 1.54e9
            weight_bits: 4
            n_layers: 28
            n_kv_heads: 2
            head_dim: 128

    Raises:
        ValueError: 缺少必填字段。
    """
    out: dict[str, ModelSpec] = {}
    for key, raw in (cfg.get("models") or {}).items():
        missing = {"name", "params", "weight_bits"} - raw.keys()
        if missing:
            raise ValueError(f"模型 {key!r} 缺少字段：{sorted(missing)}")
        out[key] = ModelSpec(
            name=str(raw["name"]),
            params=float(raw["params"]),
            weight_bits=int(raw["weight_bits"]),
            n_layers=int(raw.get("n_layers", 0)),
            n_kv_heads=int(raw.get("n_kv_heads", 0)),
            head_dim=int(raw.get("head_dim", 0)),
            kv_bits=int(raw.get("kv_bits", 16)),
            quant_overhead=float(raw.get("quant_overhead", 0.125)),
        )
    return out


def hardware_from_config(cfg: dict[str, Any]) -> dict[str, HardwareSpec]:
    """从 ``hardware`` 段构造 HardwareSpec，未声明的沿用内置标称值。

    上板实测出带宽之后，把实测值写进配置覆盖默认的 34.1 GB/s ——
    这是让 roofline 从"纸面推算"变成"可信上限"的关键一步。
    """
    out = dict(KNOWN_HARDWARE)
    for key, raw in (cfg.get("hardware") or {}).items():
        base = out.get(key)
        out[key] = HardwareSpec(
            name=str(raw.get("name", base.name if base else key)),
            peak_gops=float(raw.get("peak_gops", base.peak_gops if base else 0)),
            peak_bandwidth_gbps=float(
                raw.get("peak_bandwidth_gbps", base.peak_bandwidth_gbps if base else 0)
            ),
            efficiency_compute=float(
                raw.get("efficiency_compute", base.efficiency_compute if base else 0.6)
            ),
            efficiency_bandwidth=float(
                raw.get("efficiency_bandwidth", base.efficiency_bandwidth if base else 0.5)
            ),
            note=str(raw.get("note", base.note if base else "")),
        )
    return out


def specs_from_config(cfg: dict[str, Any]) -> list[RunSpec]:
    """从 ``scenarios`` 段构造 RunSpec 列表。

    期望的结构::

        scenarios:
          - name: short-prompt
            prompt: "用一句话解释什么是量化。"
            max_tokens: 128
            repeats: 3

    Raises:
        ValueError: 缺少必填字段。
    """
    out: list[RunSpec] = []
    for raw in cfg.get("scenarios") or []:
        missing = {"name", "prompt"} - raw.keys()
        if missing:
            raise ValueError(f"场景 {raw.get('name', '<未命名>')!r} 缺少字段：{sorted(missing)}")
        out.append(
            RunSpec(
                name=str(raw["name"]),
                prompt=str(raw["prompt"]),
                max_tokens=int(raw.get("max_tokens", 128)),
                repeats=int(raw.get("repeats", 3)),
                warmup=int(raw.get("warmup", 1)),
                cooldown_s=float(raw.get("cooldown_s", 0.0)),
            )
        )
    return out
