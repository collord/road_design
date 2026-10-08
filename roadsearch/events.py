"""The run's event stream: one JSON object per line in <run folder>/events.jsonl.

Every event has "t" (Unix time, seconds) and "kind":

    run_start    run_id, inputs
    stage_start  stage, depth
    stage_end    stage, depth, seconds, and the stage's facts (states settled, memory...)
    progress     stage, settled, key, queue, rate (states per second): from inside a search
    memory       avail_gb, private_gb, kernel pool sizes: the memory guard's samples
    log          message
    run_end      status ("finished", "failed" or "stopped"), error

`roadsearch status` reads it. With no stream started, emit() does nothing.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

_current: "EventLog | None" = None


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class EventLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)      # line-buffered
        self._lock = threading.Lock()                                       # the memory guard writes too

    def emit(self, kind: str, **fields) -> None:
        line = json.dumps({"t": round(time.time(), 3), "kind": kind, **_clean(fields)}, default=str)
        with self._lock:
            if not self._f.closed:
                self._f.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


def start(path) -> EventLog:
    global _current
    if _current is not None:
        _current.close()
    _current = EventLog(path)
    return _current


def stop() -> None:
    global _current
    if _current is not None:
        _current.close()
    _current = None


def emit(kind: str, **fields) -> None:
    if _current is not None:
        _current.emit(kind, **fields)


def read(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                                    # a line being written
    return out
