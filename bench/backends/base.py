"""Backend 抽象：一条"把 prompt 变成 token 流"的执行路径。

两条真实路径（llama.cpp / RKLLM）和一条模拟路径（mock）都实现这个接口，
这样 runner、metrics、report 完全不需要知道底下跑的是 CPU 还是 NPU。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..metrics import GenerationTrace

__all__ = ["BackendInfo", "Backend", "BackendUnavailable"]


class BackendUnavailable(RuntimeError):
    """backend 在当前机器上跑不起来（缺二进制、缺驱动、架构不对）。

    runner 捕获这个异常后会跳过该 backend 并在报告里记为 "unavailable"，
    而不是让整个 sweep 挂掉。
    """


@dataclass
class BackendInfo:
    """backend 的自我描述，会原样写进结果文件，用于复现。

    Attributes:
        name: 短标识，如 ``llama.cpp`` / ``rkllm``。
        path: 执行路径归类，``cpu`` / ``npu`` / ``gpu`` / ``mock``。
        version: 运行时版本（llama.cpp 的 commit、librkllmrt 的版本号）。
        model: 模型文件标识。
        quantization: 量化方式，如 ``Q4_0`` / ``w4a16_g128``。
        config: 影响性能的配置项（线程数、绑核掩码、NPU 核数……）。
    """

    name: str
    path: str
    version: str = "unknown"
    model: str = ""
    quantization: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class Backend(abc.ABC):
    """一条执行路径。

    生命周期：``__init__`` → ``load()`` → 多次 ``generate()`` → ``unload()``。
    实现类应当在 ``load()`` 里做所有昂贵的初始化（读权重、初始化 NPU 上下文），
    这样 ``generate()`` 的计时才不会被加载时间污染。
    """

    #: 是否为模拟 backend。基类默认 False，mock 覆盖为 True。
    synthetic: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """加载模型，准备好接受生成请求。

        Raises:
            BackendUnavailable: 当前环境跑不了这个 backend。
        """

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> GenerationTrace:
        """跑一次生成，返回带时间戳的原始时间线。

        实现必须做到：

        - ``t_start`` 在真正提交请求之前打点；
        - 每收到一个输出 token 就立刻打点，**不要**等全部生成完再补时间戳，
          否则 ITL 和降频分析全部失效；
        - ``prompt_tokens`` 用 backend 自己的 tokenizer 报告的数字。

        Args:
            prompt: 输入文本。
            max_tokens: 最多生成多少个 token。

        Returns:
            GenerationTrace。
        """

    @abc.abstractmethod
    def info(self) -> BackendInfo:
        """返回自我描述，用于结果复现。"""

    def unload(self) -> None:
        """释放资源。默认什么都不做。"""

    def __enter__(self) -> Backend:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()
