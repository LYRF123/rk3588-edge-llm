"""三条执行路径的 backend 实现。

``llama_cpp`` 和 ``rkllm`` 按需导入 —— 它们在 import 期不碰硬件，
但把 mock 的可用性和它们绑在一起没有意义。
"""

from __future__ import annotations

from .base import Backend, BackendInfo, BackendUnavailable
from .mock import MockBackend

__all__ = ["Backend", "BackendInfo", "BackendUnavailable", "MockBackend"]
