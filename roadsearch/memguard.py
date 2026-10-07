"""Memory guard: a run must never push the machine into swapping, and memory growth must show.

- snapshot(): available memory, this process's memory, and (on Windows) the kernel's
  paged and non-paged pools. The pools belong to drivers, not to us, but on this machine the
  non-paged pool grew to 22 GB during heavy runs and fell back after them, so it is watched.
- pool_tags(): which driver tags hold the pools (what poolmon shows), so a spike can be traced.
- Watchdog: a background thread sampling every few seconds. It logs the pool's top tags when
  the pool passes POOL_WARN_GB. If available memory falls below FLOOR_GB, or the pool passes
  POOL_CEILING_GB, it interrupts the run (KeyboardInterrupt in the main thread, which run()
  turns into MemoryGuardError). If the run does not stop within a grace period (stuck inside
  compiled code), it records the reason and ends the process.
- require(): before a big allocation, refuse to start if the estimate does not fit.

The thresholds can be set with the environment variables ROADSEARCH_MEM_FLOOR_GB,
ROADSEARCH_POOL_WARN_GB and ROADSEARCH_POOL_CEILING_GB.
"""

from __future__ import annotations

import _thread
import ctypes
import os
import sys
import threading
import time

import psutil

GB = 1024 ** 3
FLOOR_GB = float(os.environ.get("ROADSEARCH_MEM_FLOOR_GB", 2.0))
POOL_WARN_GB = float(os.environ.get("ROADSEARCH_POOL_WARN_GB", 4.0))
POOL_CEILING_GB = float(os.environ.get("ROADSEARCH_POOL_CEILING_GB", 8.0))
EVERY_S = 5.0                # seconds between samples


class MemoryGuardError(MemoryError):
    pass


def _kernel_pools() -> tuple[float, float] | tuple[None, None]:
    """(paged, non-paged) kernel pool in GB, from GetPerformanceInfo; (None, None) off Windows."""
    if sys.platform != "win32":
        return None, None

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return None, None
    return pi.KernelPaged * pi.PageSize / GB, pi.KernelNonpaged * pi.PageSize / GB


def pool_tags(top: int = 6) -> list[dict]:
    """The kernel pool tags holding the most non-paged memory (Windows only; [] elsewhere)."""
    if sys.platform != "win32":
        return []
    import struct
    size = 1 << 20
    for _ in range(6):
        buf = ctypes.create_string_buffer(size)
        ret = ctypes.c_ulong()
        st = ctypes.windll.ntdll.NtQuerySystemInformation(22, buf, size, ctypes.byref(ret)) & 0xFFFFFFFF
        if st == 0xC0000004:               # buffer too small
            size = max(2 * size, ret.value + 4096)
            continue
        if st != 0:
            return []
        n = struct.unpack_from("<I", buf, 0)[0]
        rows = []
        for i in range(n):                 # SYSTEM_POOLTAG, 40 bytes on 64-bit Windows
            off = 8 + 40 * i
            _, _, paged, _, _, nonpaged = struct.unpack_from("<IIxxxxQIIQ", buf, off + 4)
            rows.append({"tag": bytes(buf[off:off + 4]).decode("latin1"),
                         "nonpaged_mb": round(nonpaged / 2**20, 1), "paged_mb": round(paged / 2**20, 1)})
        return sorted(rows, key=lambda r: -r["nonpaged_mb"])[:top]
    return []


def describe_tags(tags: list[dict]) -> str:
    return ", ".join(f"{t['tag'].strip()!r} {t['nonpaged_mb']:,.0f} MB" for t in tags)


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    mi = psutil.Process(os.getpid()).memory_info()
    paged, nonpaged = _kernel_pools()
    out = {"avail_gb": round(vm.available / GB, 2), "rss_gb": round(mi.rss / GB, 2),
           "private_gb": round(getattr(mi, "private", mi.rss) / GB, 2)}
    if nonpaged is not None:
        out["kernel_nonpaged_gb"] = round(nonpaged, 2)
        out["kernel_paged_gb"] = round(paged, 2)
    return out


def require(need_gb: float, what: str, floor_gb: float) -> None:
    """Refuse an allocation that would leave less than floor_gb available."""
    s = snapshot()
    if s["avail_gb"] - need_gb < floor_gb:
        raise MemoryGuardError(
            f"{what} needs about {need_gb:.1f} GB but only {s['avail_gb']:.1f} GB is available "
            f"(keeping {floor_gb:g} GB free); use a coarser resolution or a tighter clip"
            + (f"; the kernel's non-paged pool is {s['kernel_nonpaged_gb']:.1f} GB (top tags: "
               f"{describe_tags(pool_tags(3))}), which a restart clears"
               if s.get("kernel_nonpaged_gb", 0) > POOL_WARN_GB else ""))


class Watchdog:
    """Samples memory every `every` seconds while a run is in progress."""

    def __init__(self, floor_gb: float = None, pool_warn_gb: float = None, pool_ceiling_gb: float = None,
                 every: float = None, grace: float = 120.0, log=None, on_hard_stop=None):
        self.floor_gb = FLOOR_GB if floor_gb is None else floor_gb
        self.pool_warn_gb = POOL_WARN_GB if pool_warn_gb is None else pool_warn_gb
        self.pool_ceiling_gb = POOL_CEILING_GB if pool_ceiling_gb is None else pool_ceiling_gb
        self.every, self.grace = EVERY_S if every is None else every, grace
        self.log = log
        self.on_hard_stop = on_hard_stop        # called with the reason before the process is ended
        self.reason: str | None = None
        self.peak: dict = {}                    # min_avail_gb, max_private_gb, max_kernel_nonpaged_gb, ...
        self.pool_events: list[dict] = []       # pool size and its top tags each time it passes a new level
        self._next_pool_level = self.pool_warn_gb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="memguard", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2 * self.every)
        return False

    def _breach(self, s: dict) -> str | None:
        if s["avail_gb"] < self.floor_gb:
            return f"available memory fell to {s['avail_gb']:.2f} GB (floor {self.floor_gb:g} GB)"
        if s.get("kernel_nonpaged_gb", 0.0) > self.pool_ceiling_gb:
            return (f"the kernel's non-paged pool reached {s['kernel_nonpaged_gb']:.1f} GB (ceiling "
                    f"{self.pool_ceiling_gb:g} GB; top tags: {describe_tags(pool_tags(3))})")
        return None

    def _watch_pool(self, s: dict):
        pool = s.get("kernel_nonpaged_gb")
        if pool is None or pool < self._next_pool_level:
            return
        tags = pool_tags()
        self.pool_events.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **s, "tags": tags})
        self._next_pool_level = pool + 2.0
        if self.log:
            self.log(f"memory guard: kernel non-paged pool {pool:.1f} GB (available {s['avail_gb']:.1f} GB); "
                     f"top tags {describe_tags(tags)}")

    def _loop(self):
        while not self._stop.wait(self.every):
            s = snapshot()
            self._watch_pool(s)
            for k, v in s.items():
                key = "min_avail_gb" if k == "avail_gb" else f"max_{k}"
                self.peak[key] = min(self.peak.get(key, v), v) if k == "avail_gb" else max(self.peak.get(key, v), v)
            if self.reason is None:
                why = self._breach(s)
                if why:
                    self.reason = why
                    _thread.interrupt_main()
                    t0 = time.monotonic()
                    while not self._stop.wait(1.0):
                        if time.monotonic() - t0 > self.grace:
                            if self.on_hard_stop:
                                try:
                                    self.on_hard_stop(why)
                                except Exception:
                                    pass
                            os._exit(3)
                    return
