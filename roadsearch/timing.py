"""Stage timing for a run: where the time goes, and how big each search is.

run() starts a profile; code anywhere in the pipeline wraps a stage in
`with stage("name") as row:` and may add facts to the row (states settled,
candidates tried). Stages nest; each row keeps its depth, and the process's
memory (private_gb) and the machine's available memory (avail_gb) when it ended.
With no profile started, stage() does nothing, so library calls outside a run pay nothing.
"""

from __future__ import annotations

import contextlib
import time

_current = None
_MEM_KEYS = ("private_gb", "avail_gb")


def _memory() -> dict:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"private_gb": round(getattr(mi, "private", mi.rss) / 2**30, 2),
                "avail_gb": round(psutil.virtual_memory().available / 2**30, 2)}
    except Exception:
        return {}


class Profile:
    def __init__(self):
        self.rows: list[dict] = []
        self.depth = 0

    @contextlib.contextmanager
    def stage(self, name: str, **info):
        row = {"stage": name, "depth": self.depth, **info}
        self.rows.append(row)
        self.depth += 1
        t0 = time.perf_counter()
        try:
            yield row
        finally:
            row["seconds"] = round(time.perf_counter() - t0, 3)
            row.update(_memory())
            self.depth -= 1

    def add(self, name: str, seconds: float, **info):
        """A row for time accumulated elsewhere (e.g. many small calls in a loop)."""
        self.rows.append({"stage": name, "depth": self.depth, "seconds": round(seconds, 3), **info})

    def table(self) -> str:
        total = next((r["seconds"] for r in self.rows if r["depth"] == 0 and r["stage"] == "run"), None)
        lines = [f"{'stage':<44} {'time':>11} {'share':>6} {'ours GB':>8} {'free GB':>8}"]
        for r in self.rows:
            extra = ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                              for k, v in r.items() if k not in ("stage", "depth", "seconds", *_MEM_KEYS))
            share = f"{100 * r['seconds'] / total:5.1f}%" if total else ""
            mem = f" {r['private_gb']:>8.2f} {r['avail_gb']:>8.2f}" if "avail_gb" in r else " " * 18
            lines.append(f"{'  ' * r['depth']}{r['stage']:<{44 - 2 * r['depth']}} {r['seconds']:>9.1f} s {share:>6}"
                         + mem + (f"  ({extra})" if extra else ""))
        return "\n".join(lines)


def start() -> Profile:
    global _current
    _current = Profile()
    return _current


def stop() -> Profile | None:
    global _current
    p, _current = _current, None
    return p


def stage(name: str, **info):
    return _current.stage(name, **info) if _current is not None else contextlib.nullcontext({})


def add(name: str, seconds: float, **info) -> None:
    if _current is not None:
        _current.add(name, seconds, **info)
