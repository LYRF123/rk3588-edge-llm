"""后台采样板子的温度、频率和内存占用。

端侧 benchmark 不带温度曲线基本没法看：同一个模型同一条路径，冷机跑和连续跑
半小时之后跑，能差 30%。RK3588 在被动散热的小盒子里尤其明显。

采样在独立线程里做，读的都是 sysfs 里的伪文件，开销可以忽略（每次采样几十微秒）。
在 x86 开发机上这些路径都不存在，此时 ``Sampler`` 会安静地退化成只采 RSS。
"""

from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Sample", "Sampler", "discover_sensors"]

#: RK3588 的 CPU 核编号：0-3 是 A55 小核，4-7 是 A76 大核。
BIG_CORES = "4-7"
LITTLE_CORES = "0-3"


@dataclass
class Sample:
    """一个采样点。缺失的传感器留 None，不要填 0 —— 0 度和"没采到"是两回事。"""

    t: float
    temps_c: dict[str, float] = field(default_factory=dict)
    cpu_freq_mhz: dict[str, float] = field(default_factory=dict)
    npu_freq_mhz: float | None = None
    rss_bytes: int | None = None


def discover_sensors() -> dict[str, list[str]]:
    """枚举当前机器上可用的 sysfs 传感器路径。

    Returns:
        形如 ``{"thermal": [...], "cpufreq": [...], "npu": [...]}`` 的字典。
        在非 RK 平台上对应的列表会是空的。
    """
    return {
        "thermal": sorted(glob.glob("/sys/class/thermal/thermal_zone*")),
        "cpufreq": sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")),
        "npu": sorted(glob.glob("/sys/class/devfreq/*npu*")),
    }


def _read_int(path: str | Path) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _read_str(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _read_self_rss() -> int | None:
    """从 /proc/self/status 读 VmHWM（峰值 RSS），单位字节。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_pid_rss(pid: int) -> int | None:
    """读指定进程的当前 RSS，单位字节。进程已退出返回 None。"""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class Sampler:
    """后台线程周期采样。

    用法::

        with Sampler(interval_s=0.5) as s:
            ...跑 benchmark...
        summary = s.summary()

    Args:
        interval_s: 采样间隔。0.5s 对温度曲线足够，再密只是浪费。
        pid: 要跟踪 RSS 的进程号。None 表示跟踪当前进程。
    """

    def __init__(self, interval_s: float = 0.5, pid: int | None = None) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s 必须 > 0，收到 {interval_s}")
        self.interval_s = interval_s
        self.pid = pid
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sensors = discover_sensors()
        self._zone_names = {
            z: (_read_str(f"{z}/type") or Path(z).name) for z in self._sensors["thermal"]
        }

    def _sample_once(self) -> Sample:
        temps: dict[str, float] = {}
        for zone in self._sensors["thermal"]:
            milli = _read_int(f"{zone}/temp")
            if milli is not None:
                temps[self._zone_names[zone]] = milli / 1000.0

        freqs: dict[str, float] = {}
        for cf in self._sensors["cpufreq"]:
            khz = _read_int(f"{cf}/scaling_cur_freq")
            if khz is not None:
                cpu = Path(cf).parent.name  # cpu0, cpu4, ...
                freqs[cpu] = khz / 1000.0

        npu_mhz: float | None = None
        for npu in self._sensors["npu"]:
            hz = _read_int(f"{npu}/cur_freq")
            if hz is not None:
                npu_mhz = hz / 1e6
                break

        rss = _read_pid_rss(self.pid) if self.pid is not None else _read_self_rss()

        return Sample(
            t=time.perf_counter(),
            temps_c=temps,
            cpu_freq_mhz=freqs,
            npu_freq_mhz=npu_mhz,
            rss_bytes=rss,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._sample_once())
            except Exception:  # noqa: BLE001 - 采样失败不该拖垮 benchmark
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Sampler 已经启动过了")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sysmon")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> Sampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        """把采样序列压成几个标量，写进结果文件。

        Returns:
            含 ``n_samples`` / ``temp_max_c`` / ``temp_start_c`` / ``temp_end_c``
            / ``cpu_freq_mean_mhz`` / ``npu_freq_mean_mhz`` / ``peak_rss_bytes``
            的字典。没采到的项为 None。
        """
        if not self.samples:
            return {"n_samples": 0}

        all_temps = [t for s in self.samples for t in s.temps_c.values()]
        first_temps = list(self.samples[0].temps_c.values())
        last_temps = list(self.samples[-1].temps_c.values())

        big_freqs = [
            f
            for s in self.samples
            for cpu, f in s.cpu_freq_mhz.items()
            if cpu in {f"cpu{i}" for i in range(4, 8)}
        ]
        npu_freqs = [s.npu_freq_mhz for s in self.samples if s.npu_freq_mhz is not None]
        rss = [s.rss_bytes for s in self.samples if s.rss_bytes is not None]

        return {
            "n_samples": len(self.samples),
            "duration_s": self.samples[-1].t - self.samples[0].t,
            "temp_max_c": max(all_temps) if all_temps else None,
            "temp_start_c": max(first_temps) if first_temps else None,
            "temp_end_c": max(last_temps) if last_temps else None,
            "cpu_big_freq_mean_mhz": sum(big_freqs) / len(big_freqs) if big_freqs else None,
            "cpu_big_freq_min_mhz": min(big_freqs) if big_freqs else None,
            "npu_freq_mean_mhz": sum(npu_freqs) / len(npu_freqs) if npu_freqs else None,
            "peak_rss_bytes": max(rss) if rss else None,
        }
