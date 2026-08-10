"""模拟 backend：在没有板子的机器上把整条流水线跑通。

它按 roofline 模型合成时间线，**不代表任何真实性能**。存在的意义只有两个：

1. 在 x86 开发机上验证 runner / metrics / report 的逻辑是否正确；
2. 给 CI 一个不依赖硬件的端到端测试对象。

所有由它产生的 trace 都带 ``synthetic=True``，report 会在表格里显式标注，
且 `runner` 拒绝把 synthetic 结果写进 `results/` 的正式目录。
这条约束是刻意的：一份 benchmark 仓库里混进合成数字，整个仓库就没有价值了。
"""

from __future__ import annotations

import random
import time

from ..metrics import GenerationTrace
from ..roofline import HardwareSpec, ModelSpec, roofline
from .base import Backend, BackendInfo

__all__ = ["MockBackend"]


class MockBackend(Backend):
    """按 roofline 上限 × 一个效率折扣合成时间线。

    Args:
        model: 模型规格。
        hw: 硬件规格。
        efficiency: 实测相对 roofline 上限的比例，0.55 是端侧运行时比较常见的水平。
        jitter: 每个 token 间隔的相对抖动幅度（正负比例）。
        thermal_drift: 生成到最后一个 token 时，速率相对第一个 token 的衰减比例。
            0.15 表示末尾比开头慢 15%，用来模拟热降频。
        seed: 随机种子，保证可复现。
        sleep: 是否真的 sleep。默认 False —— CI 里合成 512 个 token 不该真等半分钟。
            置 True 时时间戳来自真实时钟，可用来验证计时链路本身。
    """

    synthetic = True

    def __init__(
        self,
        model: ModelSpec,
        hw: HardwareSpec,
        *,
        efficiency: float = 0.55,
        jitter: float = 0.05,
        thermal_drift: float = 0.15,
        seed: int = 0,
        sleep: bool = False,
    ) -> None:
        self.model = model
        self.hw = hw
        self.efficiency = efficiency
        self.jitter = jitter
        self.thermal_drift = thermal_drift
        self.sleep = sleep
        self._rng = random.Random(seed)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str, max_tokens: int) -> GenerationTrace:
        if not self._loaded:
            raise RuntimeError("generate() 调用前必须先 load()")
        if max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {max_tokens}")

        # 粗略估 token 数：这是 mock，不需要真 tokenizer。
        prompt_tokens = max(1, len(prompt.split()) + len(prompt) // 8)

        prefill = roofline(self.model, self.hw, "prefill", batch_tokens=prompt_tokens)
        decode = roofline(self.model, self.hw, "decode", context_len=prompt_tokens)

        prefill_s = prompt_tokens / (prefill.ceiling_tok_s * self.efficiency)
        base_itl_s = 1.0 / (decode.ceiling_tok_s * self.efficiency)

        t0 = time.perf_counter()
        if self.sleep:
            time.sleep(prefill_s)
            t_first = time.perf_counter()
        else:
            t_first = t0 + prefill_s

        times = [t_first]
        t = t_first
        for i in range(1, max_tokens):
            drift = 1.0 + self.thermal_drift * (i / max(max_tokens - 1, 1))
            noise = 1.0 + self._rng.uniform(-self.jitter, self.jitter)
            step = base_itl_s * drift * noise
            if self.sleep:
                time.sleep(step)
                t = time.perf_counter()
            else:
                t += step
            times.append(t)

        return GenerationTrace(
            prompt_tokens=prompt_tokens,
            t_start=t0,
            t_first_token=t_first,
            token_times=times,
            peak_rss_bytes=int(self.model.weight_bytes * 1.12),
            backend="mock",
            model=self.model.name,
            synthetic=True,
            extra={
                "efficiency": self.efficiency,
                "roofline_prefill_tok_s": prefill.ceiling_tok_s,
                "roofline_decode_tok_s": decode.ceiling_tok_s,
                "decode_bound_by": decode.bound_by,
            },
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock",
            path="mock",
            version="synthetic",
            model=self.model.name,
            quantization=f"{self.model.weight_bits}bit",
            config={
                "hardware": self.hw.name,
                "efficiency": self.efficiency,
                "thermal_drift": self.thermal_drift,
            },
        )
