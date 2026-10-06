"""Stage timing for a run: where the time goes, and how big each search is.

run() starts a profile; code anywhere in the pipeline wraps a stage in
`with stage("name") as row:` and may add facts to the row (states settled,
candidates tried). Stages nest; each row keeps its depth. With no profile
started, stage() does nothing, so library calls outside a run pay nothing.
"""

from __future__ import annotations

import contextlib
import time

_current = None


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
            self.depth -= 1

    def add(self, name: str, seconds: float, **info):
        """A row for time accumulated elsewhere (e.g. many small calls in a loop)."""
        self.rows.append({"stage": name, "depth": self.depth, "seconds": round(seconds, 3), **info})

    def table(self) -> str:
        total = next((r["seconds"] for r in self.rows if r["depth"] == 0 and r["stage"] == "run"), None)
        lines = []
        for r in self.rows:
            extra = ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                              for k, v in r.items() if k not in ("stage", "depth", "seconds"))
            share = f"{100 * r['seconds'] / total:5.1f}%" if total else ""
            lines.append(f"{'  ' * r['depth']}{r['stage']:<{44 - 2 * r['depth']}} {r['seconds']:>9.1f} s {share}"
                         + (f"  ({extra})" if extra else ""))
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
