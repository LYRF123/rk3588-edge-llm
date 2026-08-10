"""命令行入口：``python -m bench <子命令>``。

四个子命令：

- ``roofline`` — 算理论上限，不需要板子也不需要模型
- ``run``      — 跑 benchmark
- ``report``   — 结果 JSON → Markdown
- ``sysinfo``  — 打印当前机器的传感器和环境信息
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.mock import MockBackend
from .config import hardware_from_config, load_config, models_from_config, specs_from_config
from .report import load_results, render_markdown
from .roofline import KNOWN_HARDWARE, roofline
from .runner import SYNTHETIC_DIR, collect_env, run_suite
from .sysmon import Sampler, discover_sensors

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def cmd_roofline(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    models = models_from_config(cfg)
    hardware = hardware_from_config(cfg)

    if not models:
        print("配置里没有 models 段", file=sys.stderr)
        return 1

    for mkey, model in models.items():
        if args.model and mkey != args.model:
            continue
        print(f"\n=== {mkey}  ({model.name}, {model.weight_bits}bit, "
              f"权重 {model.weight_bytes / 2**20:.0f} MiB) ===")
        for hkey, hw in hardware.items():
            if args.path and hkey != args.path:
                continue
            pf = roofline(model, hw, "prefill", batch_tokens=args.prompt_tokens)
            dc = roofline(model, hw, "decode", context_len=args.prompt_tokens)
            print(f"  [{hkey}] {hw.name}")
            print(f"    prefill ({args.prompt_tokens} tok): "
                  f"{pf.ceiling_tok_s:8.1f} tok/s  受限于{'算力' if pf.bound_by == 'compute' else '带宽'}")
            print(f"    decode              : "
                  f"{dc.ceiling_tok_s:8.1f} tok/s  受限于{'算力' if dc.bound_by == 'compute' else '带宽'}"
                  f"   (每 token 读 {dc.detail['bytes_per_token'] / 2**20:.0f} MiB)")
    print("\n注：以上是**标称参数**推算的上限，不是实测值。"
          "实测带宽通常只有标称的 45%~60%，上板后请用实测值覆盖 configs/models.yaml 的 hardware 段。")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    models = models_from_config(cfg)
    hardware = hardware_from_config(cfg)
    specs = specs_from_config(cfg)

    if not specs:
        print("配置里没有 scenarios 段", file=sys.stderr)
        return 1

    backends = {}
    if args.mock:
        mkey = args.model or next(iter(models))
        if mkey not in models:
            print(f"配置里没有模型 {mkey!r}，可选：{sorted(models)}", file=sys.stderr)
            return 1
        for hkey in ("cpu", "npu"):
            backends[f"mock-{hkey}"] = MockBackend(
                models[mkey], hardware.get(hkey, KNOWN_HARDWARE[hkey])
            )
        out_dir = Path(args.out) / SYNTHETIC_DIR
    else:
        print(
            "真实 backend 需要板子和模型文件。当前仓库还没上板，\n"
            "先用 --mock 验证流水线，或者按 docs/01-setup-board.md 配好环境后\n"
            "在这里接入 LlamaCppBackend / RKLLMBackend。",
            file=sys.stderr,
        )
        return 2

    run_suite(backends, specs, out_dir, tag=args.tag)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data = load_results(args.input)
    md = render_markdown(data, title=args.title)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"报告写入 {args.output}")
    else:
        print(md)
    return 0


def cmd_sysinfo(args: argparse.Namespace) -> int:
    env = collect_env()
    sensors = discover_sensors()
    sampler = Sampler(interval_s=0.1)
    sampler.start()
    import time as _t

    _t.sleep(0.5)
    sampler.stop()

    print(json.dumps(
        {"env": env, "sensors": sensors, "sample": sampler.summary()},
        indent=2, ensure_ascii=False,
    ))
    if not sensors["npu"]:
        print("\n注：没找到 NPU devfreq 节点 —— 这台机器不是 RK 平台，"
              "或者 NPU 驱动没加载。", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="RK3588 端侧 LLM benchmark",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("roofline", help="计算理论上限（不需要板子）")
    pr.add_argument("--model", help="只算这一个模型")
    pr.add_argument("--path", choices=["cpu", "npu"], help="只算这一条路径")
    pr.add_argument("--prompt-tokens", type=int, default=512, help="prefill 的 token 数")
    pr.set_defaults(func=cmd_roofline)

    prun = sub.add_parser("run", help="跑 benchmark")
    prun.add_argument("--mock", action="store_true", help="用合成 backend 验证流水线")
    prun.add_argument("--model", help="模型 key")
    prun.add_argument("--out", default="results", help="结果目录")
    prun.add_argument("--tag", default="", help="结果文件名标签")
    prun.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="结果 JSON → Markdown")
    prep.add_argument("input", help="结果 JSON 路径")
    prep.add_argument("-o", "--output", help="输出 Markdown 路径，省略则打到 stdout")
    prep.add_argument("--title", default="Benchmark 结果", help="报告标题")
    prep.set_defaults(func=cmd_report)

    psi = sub.add_parser("sysinfo", help="打印本机传感器与环境信息")
    psi.set_defaults(func=cmd_sysinfo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
