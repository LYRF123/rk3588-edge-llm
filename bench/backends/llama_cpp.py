"""CPU 路径：llama.cpp（GGUF）。

实现方式是拉起 ``llama-cli`` 子进程并**流式**读取 stdout，每读到一块输出就打一次
时间戳。之所以不用 ``llama-bench``：llama-bench 只给聚合后的 pp/tg 均值，
拿不到 ITL 分布，也就看不出热降频 —— 而热降频恰恰是端侧最需要暴露的问题。

.. warning::
   **本文件尚未在真实 RK3588 上验证过。** 已知的不确定点集中在两处，
   都在代码里用 ``VERIFY:`` 标出了。上板第一件事就是核对这两处。

关于绑核：RK3588 是 big.LITTLE，Linux 下 ``cpu0-3`` 是 A55 小核，
``cpu4-7`` 是 A76 大核。让线程漂到小核上会拖垮整体，所以默认
``taskset -c 4-7`` + ``--threads 4``。混用 8 核通常**更慢**，
因为 GEMM 是同步的，快核要等慢核。这个结论需要实测确认，见
``configs/sweeps/default.yaml`` 里的 ``thread_affinity`` 扫描项。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..metrics import GenerationTrace
from .base import Backend, BackendInfo, BackendUnavailable

__all__ = ["LlamaCppBackend"]

# llama-cli 会把加载日志和统计信息写到 stderr，token 写到 stdout。
# VERIFY: 不同版本的 llama.cpp 对 stdout/stderr 的划分变过，上板后先手跑一次
# `llama-cli -m model.gguf -p hi -n 4 2>/dev/null` 确认 stdout 里只有 token。
_TOKENS_RE = re.compile(r"prompt eval count:\s+(\d+)")


class LlamaCppBackend(Backend):
    """通过子进程驱动 llama.cpp。

    Args:
        model_path: GGUF 文件路径。
        binary: ``llama-cli`` 可执行文件路径，默认从 PATH 里找。
        threads: 计算线程数，默认 4（对应 4 个 A76 大核）。
        cpu_list: 传给 ``taskset -c`` 的核列表，None 表示不绑核。
        n_ctx: 上下文长度。
        extra_args: 追加给 llama-cli 的参数。
        timeout_s: 单次生成的超时秒数。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        binary: str = "llama-cli",
        threads: int = 4,
        cpu_list: str | None = "4-7",
        n_ctx: int = 2048,
        extra_args: list[str] | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.binary = binary
        self.threads = threads
        self.cpu_list = cpu_list
        self.n_ctx = n_ctx
        self.extra_args = extra_args or []
        self.timeout_s = timeout_s
        self._resolved_binary: str = ""
        self._version = "unknown"

    def load(self) -> None:
        resolved = shutil.which(self.binary)
        if resolved is None:
            raise BackendUnavailable(
                f"找不到 {self.binary}，先跑 scripts/build_llama_cpp.sh 编译，"
                "或用 binary= 指定完整路径"
            )
        self._resolved_binary = resolved

        if not self.model_path.is_file():
            raise BackendUnavailable(f"GGUF 模型不存在：{self.model_path}")

        if self.cpu_list and shutil.which("taskset") is None:
            raise BackendUnavailable("指定了 cpu_list 但系统里没有 taskset（util-linux）")

        self._version = self._probe_version()

    def _probe_version(self) -> str:
        """从 ``llama-cli --version`` 里取版本号，失败则返回 unknown。"""
        try:
            out = subprocess.run(
                [self._resolved_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        blob = (out.stderr or "") + (out.stdout or "")
        for line in blob.splitlines():
            if "version" in line.lower():
                return line.strip()
        return "unknown"

    def _build_cmd(self, prompt: str, max_tokens: int) -> list[str]:
        cmd: list[str] = []
        if self.cpu_list:
            cmd += ["taskset", "-c", self.cpu_list]
        cmd += [
            self._resolved_binary,
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.threads),
            "-c", str(self.n_ctx),
            "--no-display-prompt",  # 只输出生成的部分，否则时间戳全乱
            "--simple-io",          # 关掉 ANSI 控制符，避免污染 stdout
            "-no-cnv",              # 单轮补全，不进交互式对话循环
            "--temp", "0",          # 贪心解码，保证不同 backend 之间可比
            "-s", "0",
        ]
        # 注意：**不要**加 `-st/--single-turn`。它的语义是"对话模式只跑一轮"，
        # 会把 `-no-cnv` 重新翻回对话模式，于是套上 chat template；
        # 对 Qwen3 这类默认开思考模式的模型，套上模板后会先吐一大段
        # `[Start thinking]...`，TTFT 变成"第一个思考 token 的时间"、
        # decode tok/s 里混进思考 token，整组指标失去意义。
        # 2026-08-10 在 x86 上实测踩过这个坑，见 results/x86-baseline/README.md。
        cmd += self.extra_args
        return cmd

    def generate(self, prompt: str, max_tokens: int) -> GenerationTrace:
        if not self._resolved_binary:
            raise RuntimeError("generate() 调用前必须先 load()")
        if max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {max_tokens}")

        cmd = self._build_cmd(prompt, max_tokens)
        env = dict(os.environ, LC_ALL="C")

        t_start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,  # 无缓冲，否则时间戳会被 libc 的行缓冲对齐掉
        )
        assert proc.stdout is not None

        token_times: list[float] = []
        chunks: list[bytes] = []
        deadline = t_start + self.timeout_s
        try:
            while True:
                # VERIFY: 这里假设"一次 read 返回的一块 == 一个 token"。
                # llama-cli 每生成一个 token 就 write+flush 一次，在无缓冲管道上
                # 通常成立，但多字节 UTF-8（中文）会被拆成多次 write。
                # 上板后用 ASCII prompt 校准：token_times 的长度应该正好等于
                # stderr 里报告的 eval count。对不上就改用 llama-server 的 SSE 流。
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                now = time.perf_counter()
                if now > deadline:
                    proc.kill()
                    raise TimeoutError(f"llama-cli 超过 {self.timeout_s}s 未结束")
                chunks.append(chunk)
                # 只在非续接字节（不是 UTF-8 continuation byte 0b10xxxxxx）上打点，
                # 这样一个多字节汉字算作 1 个事件而不是 3 个。
                if (chunk[0] & 0xC0) != 0x80:
                    token_times.append(now)
        finally:
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            proc.wait(timeout=30)

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-cli 退出码 {proc.returncode}\n"
                f"命令：{' '.join(cmd)}\nstderr 末尾：\n{stderr[-2000:]}"
            )
        if not token_times:
            raise RuntimeError(f"没有从 stdout 收到任何输出\nstderr 末尾：\n{stderr[-2000:]}")

        m = _TOKENS_RE.search(stderr)
        prompt_tokens = int(m.group(1)) if m else max(1, len(prompt.split()))

        return GenerationTrace(
            prompt_tokens=prompt_tokens,
            t_start=t_start,
            t_first_token=token_times[0],
            token_times=token_times,
            peak_rss_bytes=None,  # 由 runner 通过 sysmon 采集
            backend="llama.cpp",
            model=self.model_path.name,
            synthetic=False,
            extra={
                "threads": self.threads,
                "cpu_list": self.cpu_list,
                "n_ctx": self.n_ctx,
                "raw_output_bytes": len(b"".join(chunks)),
                "prompt_tokens_parsed": m is not None,
            },
        )

    def info(self) -> BackendInfo:
        # GGUF 文件名里一般带量化方式，如 xxx-Q4_0.gguf
        quant = ""
        for part in self.model_path.stem.split("-"):
            if part.upper().startswith("Q") and any(c.isdigit() for c in part):
                quant = part.upper()
        return BackendInfo(
            name="llama.cpp",
            path="cpu",
            version=self._version,
            model=self.model_path.name,
            quantization=quant,
            config={
                "threads": self.threads,
                "cpu_list": self.cpu_list,
                "n_ctx": self.n_ctx,
                "extra_args": self.extra_args,
            },
        )
