#!/usr/bin/env python3
"""HuggingFace 模型 → ``.rkllm``，在 **x86 主机**上跑，不是在板子上。

.. warning::
   **这是骨架。** RKLLM-Toolkit 的 API 在版本之间改过，手上又还没有板子来
   闭环验证转换产物，所以这里不写死调用细节 —— 写一个凭记忆拼出来的调用序列，
   比留一个明确的 TODO 更糟：它看起来能用，实际会在某个参数上悄悄出错。

   上板前要做的事：装好对应版本的 RKLLM-Toolkit，用 ``help(RKLLM)`` 核对
   ``load_huggingface`` / ``build`` / ``export_rkllm`` 三个方法的**实际签名**，
   然后把下面 ``convert()`` 里的 TODO 补上。

RKLLM 的工具链是两段式的，这一点和 llama.cpp 很不一样：

- **转换**（本脚本）只能在 x86 Linux 上做，需要 GPU 或至少大内存，
  因为量化校准要跑前向；
- **推理** 只能在板子上做，靠 ``librkllmrt.so``。

量化方式的选择直接决定了后面 benchmark 的可比性：

===========  ==========================  ===============================
方案         含义                        与 llama.cpp 的对应关系
===========  ==========================  ===============================
``w4a16``    权重 4bit，激活 fp16        最接近 Q4_0，但激活不量化
``w8a8``     权重和激活都 int8           llama.cpp 没有直接对应
``w4a16_g``  分组 w4a16（g32/g64/g128）  类似 Q4_K 的分组量化
===========  ==========================  ===============================

**不要**因为"都是 4bit"就把 w4a16 和 Q4_0 的速度直接对比而不提精度。
两者的精度损失不一样，必须用同一套评测集单独测过精度再谈速度，
否则就是拿一个更糟的模型换来的更快 —— 那不叫优化。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_QUANT = ("w4a16", "w4a16_g32", "w4a16_g64", "w4a16_g128", "w8a8")


def check_toolkit() -> str:
    """确认 RKLLM-Toolkit 装好了，返回版本号。

    Raises:
        SystemExit: 没装或装的版本不对。
    """
    try:
        from rkllm.api import RKLLM  # noqa: F401,PLC0415
    except ImportError:
        sys.exit(
            "没装 RKLLM-Toolkit。\n"
            "它不在 PyPI 上，要从 Rockchip 的 rknn-llm 仓库里拿 wheel 装：\n"
            "  git clone https://github.com/airockchip/rknn-llm\n"
            "  pip install rknn-llm/rkllm-toolkit/rkllm_toolkit-*.whl\n"
            "注意：只支持 x86_64 Linux + 特定的 Python 版本，看仓库 README 的要求。"
        )
    try:
        import rkllm  # noqa: PLC0415

        return getattr(rkllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def convert(
    model_path: str,
    out_path: Path,
    quant: str,
    *,
    max_context: int,
    calib_data: str | None,
) -> None:
    """把 HF 模型转成 .rkllm。

    Args:
        model_path: HF 模型目录或 repo id。
        out_path: 输出的 .rkllm 路径。
        quant: 量化方案，取值见 ``SUPPORTED_QUANT``。
        max_context: 最大上下文长度。**这个值在转换时就固定了**，
            板子上没法再调大，选小了后面要重新转。
        calib_data: 量化校准数据集路径。None 表示用 toolkit 的默认数据。
            用目标场景的真实语料做校准，精度会明显好于默认的通用语料。

    Raises:
        NotImplementedError: 目前总是抛 —— 见模块文档的说明。
    """
    version = check_toolkit()
    print(f"RKLLM-Toolkit 版本：{version}")
    print(f"模型：{model_path}")
    print(f"量化：{quant}   最大上下文：{max_context}")
    print(f"输出：{out_path}")

    raise NotImplementedError(
        "转换逻辑还没实现 —— 需要先在装好 toolkit 的环境里核对 API 签名。\n"
        "\n"
        "大致流程（**必须先用 help() 核对，不要照抄**）：\n"
        "  1. llm = RKLLM()\n"
        "  2. llm.load_huggingface(model=...)         # 参数名待核对\n"
        "  3. llm.build(quantized_dtype=..., target_platform='rk3588', ...)\n"
        "  4. llm.export_rkllm(str(out_path))\n"
        "\n"
        "核对时重点确认三件事：\n"
        f"  - quantized_dtype 的合法取值里有没有 {quant!r}；\n"
        "  - max_context / max_context_len 到底叫什么，以及默认值是多少；\n"
        "  - target_platform 是 'rk3588' 还是 'RK3588'（大小写敏感）。\n"
        "\n"
        "转完之后一定要做的事：在板子上跑一遍，用同一批 prompt 和 CPU 路径\n"
        "对比输出质量。转换成功 ≠ 转换正确 —— 量化配错了照样能导出文件，\n"
        "只是模型会开始胡言乱语。"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="HuggingFace → .rkllm 转换（在 x86 主机上跑）",
        epilog="注意：这个脚本目前是骨架，见 --help 之外的模块文档。",
    )
    p.add_argument("model", help="HF 模型目录或 repo id")
    p.add_argument("-o", "--output", required=True, help="输出的 .rkllm 路径")
    p.add_argument(
        "-q", "--quant", default="w4a16", choices=SUPPORTED_QUANT, help="量化方案"
    )
    p.add_argument(
        "--max-context", type=int, default=2048,
        help="最大上下文长度。转换时固定，板子上改不了。",
    )
    p.add_argument("--calib-data", help="量化校准数据集路径")
    args = p.parse_args(argv)

    out = Path(args.output)
    if out.exists():
        print(f"输出文件已存在，不覆盖：{out}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)

    convert(
        args.model, out, args.quant,
        max_context=args.max_context, calib_data=args.calib_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
