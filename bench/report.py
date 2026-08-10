"""把结果 JSON 渲染成 Markdown 表格。

合成数据在表格里会被显式标注，且整份报告顶部会挂一条警告横幅 ——
这样即使有人只截图了表格发出去，"这不是真机数据"也跟着一起走。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["render_markdown", "load_results"]

_COLUMNS = [
    ("spec_name", "场景"),
    ("path", "路径"),
    ("backend", "后端"),
    ("decode_tok_s", "decode tok/s"),
    ("ttft_ms", "TTFT (ms)"),
    ("prefill_tok_s", "prefill tok/s"),
    ("itl_p90_ms", "ITL p90 (ms)"),
    ("peak_rss_mib", "峰值内存 (MiB)"),
    ("throttle_ratio", "降频比"),
]


def load_results(path: str | Path) -> dict[str, Any]:
    """读取 runner 写出的结果 JSON。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 内容不是合法 JSON。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def render_markdown(data: dict[str, Any], *, title: str = "Benchmark 结果") -> str:
    """渲染 Markdown 报告。

    Args:
        data: ``load_results`` 的返回值。
        title: 报告标题。

    Returns:
        Markdown 文本。
    """
    lines: list[str] = [f"# {title}", ""]

    if data.get("synthetic"):
        lines += [
            "> [!WARNING]",
            "> **本报告含合成数据，不是真机测量结果。**",
            "> 由 mock backend 按 roofline 模型生成，仅用于验证流水线，"
            "任何情况下都不要引用其中的数字。",
            "",
        ]

    env = data.get("env", {})
    lines += [
        "## 运行环境",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for k in ("soc_model", "machine", "kernel", "platform", "hostname", "timestamp_utc"):
        if k in env:
            lines.append(f"| {k} | `{env[k]}` |")
    lines.append("")

    results = data.get("results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") != "ok"]

    lines += ["## 测量结果", ""]
    if not ok:
        lines += ["*（没有成功的测量）*", ""]
    else:
        lines.append("| " + " | ".join(h for _, h in _COLUMNS) + " |")
        lines.append("| " + " | ".join("---" for _ in _COLUMNS) + " |")
        for r in ok:
            m = r.get("metrics") or {}
            row = []
            for key, _ in _COLUMNS:
                v = r.get(key, m.get(key))
                row.append(_fmt(v))
            mark = " ⚠️合成" if r.get("synthetic") else ""
            row[0] = row[0] + mark
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines += [
            "**列的含义**（详见 `docs/04-benchmark-protocol.md`）：",
            "",
            "- `decode tok/s` — 逐 token 生成速率，端侧最关键的稳态指标，受内存带宽限制。",
            "- `TTFT` — 首 token 延迟，用户实际感知到的『卡不卡』，几乎全由 prefill 决定。",
            "- `ITL p90` — 相邻 token 间隔的 90 分位，比均值更能暴露卡顿。",
            "- `降频比` — 前 25% token 的速率 ÷ 后 25% 的速率。明显小于 1 说明跑热了在降频。",
            "",
        ]

    if bad:
        lines += ["## 未能完成的组合", "", "| 场景 | 后端 | 状态 | 原因 |", "| --- | --- | --- | --- |"]
        for r in bad:
            err = (r.get("error") or "").replace("\n", " ")[:180]
            lines.append(
                f"| {r.get('spec_name')} | {r.get('backend')} | {r.get('status')} | {err} |"
            )
        lines.append("")

    return "\n".join(lines)
