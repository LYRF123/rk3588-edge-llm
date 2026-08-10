"""NPU 路径：RKLLM（Rockchip rknn-llm）。

.. warning::
   **这是骨架，不是已验证的实现。** 手上还没有 RK3588 板子，
   下面对 RKLLM 的一切描述都来自公开文档，必须上板核对后才能当真。
   每一处需要核对的地方都标了 ``VERIFY:``。

RKLLM 的形态和 llama.cpp 差别很大，值得先说清楚，否则容易踩坑：

- **两段式工具链。** x86 上用 RKLLM-Toolkit（Python）把 HuggingFace 模型转成
  ``.rkllm``；板子上用 RKLLM Runtime（``librkllmrt.so``，C API）加载运行。
  转换只能在 x86 上做，板子上做不了。
- **运行时闭源。** 拿不到算子级的 profile，也没法替换某个算子的实现。
  这直接决定了本仓库"算子优化"那部分只能落在 CPU 路径上 —— 见
  ``docs/06-operator-notes.md`` 里对这个边界的说明。
- **量化方式不同。** 走的是 w4a16 / w8a8 这类 activation 也量化的方案，
  和 llama.cpp 的 Q4_0（activation 动态量化成 int8）不是一回事，
  精度对比必须用同一套评测集单独做，不能想当然认为 "都是 4bit 所以差不多"。
- **依赖 NPU 驱动版本。** 板子上的 rknpu 驱动版本低于 runtime 要求会直接失败。

接入方式与 CPU 路径保持一致：拉起官方 demo 可执行文件，流式读 stdout 打时间戳。
直接用 ctypes 绑 ``librkllmrt.so`` 能拿到更准的回调时间戳，但结构体布局会随版本
变动，在没有板子核对头文件之前写死等于埋雷，所以留作 TODO（见文件末尾）。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from ..metrics import GenerationTrace
from .base import Backend, BackendInfo, BackendUnavailable

__all__ = ["RKLLMBackend", "check_npu_driver"]

#: NPU 驱动版本文件。VERIFY: 路径在不同内核版本上可能是
#: /sys/kernel/debug/rknpu/version（需 root）或 dmesg 里的 "RKNPU driver" 行。
_NPU_VERSION_PATHS = (
    "/sys/kernel/debug/rknpu/version",
    "/proc/rknpu/version",
)

#: NPU devfreq 节点，用于读当前频率。VERIFY: 地址前缀 fdab0000 来自 RK3588 设备树，
#: 换 SoC 或换内核版本都可能变，用 `ls /sys/class/devfreq/` 确认。
NPU_DEVFREQ = "/sys/class/devfreq/fdab0000.npu"


def check_npu_driver() -> str | None:
    """读 NPU 驱动版本，读不到返回 None。

    Returns:
        版本字符串，或 None（文件不存在 / 没权限 / 不是 RK 平台）。
    """
    for p in _NPU_VERSION_PATHS:
        try:
            return Path(p).read_text().strip()
        except (OSError, UnicodeDecodeError):
            continue
    return None


class RKLLMBackend(Backend):
    """通过官方 demo 可执行文件驱动 RKLLM runtime。

    Args:
        model_path: ``.rkllm`` 模型路径。
        binary: demo 可执行文件。rknn-llm 仓库里的示例编译产物通常叫
            ``llm_demo``；VERIFY: 不同版本的示例名字和参数顺序都变过。
        num_npu_core: 使用的 NPU 核数，RK3588 有 3 个。
        lib_dir: ``librkllmrt.so`` 所在目录，会加到 LD_LIBRARY_PATH。
        max_context_len: 上下文长度上限，需与转换模型时的设置一致。
        timeout_s: 单次生成超时秒数。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        binary: str = "llm_demo",
        num_npu_core: int = 3,
        lib_dir: str | Path | None = None,
        max_context_len: int = 2048,
        timeout_s: float = 600.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.binary = binary
        self.num_npu_core = num_npu_core
        self.lib_dir = Path(lib_dir) if lib_dir else None
        self.max_context_len = max_context_len
        self.timeout_s = timeout_s
        self._resolved_binary = ""
        self._driver_version = "unknown"

    def load(self) -> None:
        if platform.machine() not in ("aarch64", "arm64"):
            raise BackendUnavailable(
                f"RKLLM runtime 只有 aarch64 版本，当前架构是 {platform.machine()}。"
                "模型转换（RKLLM-Toolkit）才是在 x86 上做的，见 scripts/convert_rkllm.py"
            )

        driver = check_npu_driver()
        if driver is None:
            raise BackendUnavailable(
                "读不到 rknpu 驱动版本，NPU 可能没有启用。"
                f"检查 {NPU_DEVFREQ} 是否存在，以及 dmesg | grep -i rknpu"
            )
        self._driver_version = driver

        resolved = shutil.which(self.binary)
        if resolved is None:
            raise BackendUnavailable(
                f"找不到 {self.binary}。需要先编译 rknn-llm 的示例程序，"
                "见 docs/03-npu-path.md"
            )
        self._resolved_binary = resolved

        if not self.model_path.is_file():
            raise BackendUnavailable(f".rkllm 模型不存在：{self.model_path}")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ, LC_ALL="C")
        if self.lib_dir:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{self.lib_dir}:{prev}" if prev else str(self.lib_dir)
        return env

    def generate(self, prompt: str, max_tokens: int) -> GenerationTrace:
        if not self._resolved_binary:
            raise RuntimeError("generate() 调用前必须先 load()")
        if max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {max_tokens}")

        # VERIFY: 参数顺序按 rknn-llm 示例的 `llm_demo <model> <max_new_tokens>
        # <max_context_len>` 写的，上板后用 --help 核对。
        cmd = [
            self._resolved_binary,
            str(self.model_path),
            str(max_tokens),
            str(self.max_context_len),
        ]

        t_start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env(),
            bufsize=0,
        )
        assert proc.stdin is not None and proc.stdout is not None

        proc.stdin.write((prompt + "\n").encode())
        proc.stdin.flush()

        token_times: list[float] = []
        out = bytearray()
        deadline = t_start + self.timeout_s
        try:
            while True:
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                now = time.perf_counter()
                if now > deadline:
                    proc.kill()
                    raise TimeoutError(f"{self.binary} 超过 {self.timeout_s}s 未结束")
                out += chunk
                if (chunk[0] & 0xC0) != 0x80:
                    token_times.append(now)
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            proc.wait(timeout=30)

        if not token_times:
            raise RuntimeError(f"没有收到输出\nstderr 末尾：\n{stderr[-2000:]}")

        return GenerationTrace(
            # VERIFY: demo 不一定打印 prompt token 数。真拿不到的话，
            # 用转换时同一个 tokenizer 在 x86 上离线数好，通过配置传进来，
            # 绝对不要用 len(split()) 糊弄 —— prefill 速率会整个跑偏。
            prompt_tokens=max(1, len(prompt.split())),
            t_start=t_start,
            t_first_token=token_times[0],
            token_times=token_times,
            peak_rss_bytes=None,
            backend="rkllm",
            model=self.model_path.name,
            synthetic=False,
            extra={
                "num_npu_core": self.num_npu_core,
                "driver_version": self._driver_version,
                "max_context_len": self.max_context_len,
                "prompt_tokens_are_estimated": True,
            },
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="rkllm",
            path="npu",
            version=self._driver_version,
            model=self.model_path.name,
            # 量化方式在转换时决定，从文件名里带出来，见 scripts/convert_rkllm.py
            quantization=self.model_path.stem.split("_")[-1],
            config={
                "num_npu_core": self.num_npu_core,
                "max_context_len": self.max_context_len,
            },
        )


# TODO(上板后): 换成 ctypes 直接绑 librkllmrt.so。
#
# 子进程方案的时间戳里混了管道传输的开销，虽然量级上远小于 ITL（微秒 vs 毫秒），
# 但直接绑 so 能在 result callback 里打点，更干净，还能拿到 runtime 自己统计的
# prefill/decode 耗时。要做这件事需要在板子上核对三样东西：
#
#   1. rkllm.h 里 RKLLMParam / RKLLMResult 的**完整字段顺序和类型**
#      —— 这个结构体跨版本改过，抄错一个字段就是段错误，且症状具有迷惑性；
#   2. 回调函数签名和 RKLLMCallState 的枚举值；
#   3. rkllm_init / rkllm_run / rkllm_destroy 的实际符号名（nm -D librkllmrt.so）。
#
# 在核对完成之前不要凭记忆写 ctypes.Structure —— 写错了不会报错，只会给出
# 看起来合理但完全错误的数字，那比跑不起来更糟。
