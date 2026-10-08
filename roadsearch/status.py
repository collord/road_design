"""`roadsearch status`: where a run is, from its event stream (events.jsonl)."""

from __future__ import annotations

import time
from pathlib import Path

from .events import read

STALE_S = 60.0      # the memory guard samples every 5 s, so a minute of silence means trouble


def find_events(path: str | Path) -> Path | None:
    """The events.jsonl of a run folder, or the most recently written one under an output root
    (or a folder of output roots)."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "events.jsonl").exists():
        return path / "events.jsonl"
    found = list(path.glob("runs/*/events.jsonl")) + list(path.glob("*/runs/*/events.jsonl"))
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def summarize(path: str | Path, now: float | None = None) -> str:
    ev_path = find_events(path)
    if ev_path is None:
        return f"no events.jsonl under {path}"
    evs = read(ev_path)
    starts = [i for i, e in enumerate(evs) if e["kind"] == "run_start"]
    evs = evs[starts[-1]:] if starts else evs
    if not evs:
        return f"{ev_path}: empty"
    now = time.time() if now is None else now
    t0 = evs[0]["t"]
    run_id = evs[0].get("run_id", ev_path.parent.name)
    end = next((e for e in reversed(evs) if e["kind"] == "run_end"), None)
    lines = [f"run {run_id}  ({ev_path.parent})",
             f"started {time.strftime('%Y-%m-%d %H:%M', time.localtime(t0))}"]
    if end:
        lines[-1] += f", {end['status']} after {_dur(end['t'] - t0)}"
        if end.get("error"):
            lines.append(f"error: {end['error']}")
    else:
        age = now - evs[-1]["t"]
        lines[-1] += f", running for {_dur(now - t0)}"
        if age > STALE_S:
            lines.append(f"no events for {_dur(age)}: the process may have stopped")
    # Stages under way: replay starts and ends.
    stack = []
    for e in evs:
        if e["kind"] == "stage_start":
            stack.append(e)
        elif e["kind"] == "stage_end" and stack:
            stack.pop()
    if stack and not end:
        lines.append("stage: " + " > ".join(e["stage"] for e in stack)
                     + f"  ({_dur(now - stack[-1]['t'])} in)")
        prog = next((e for e in reversed(evs) if e["kind"] == "progress" and e["t"] >= stack[-1]["t"]), None)
        if prog:
            lines.append(f"search: {prog['settled'] / 1e6:,.1f} M states settled, {prog['rate'] / 1e6:.2f} M/s, "
                         f"frontier {prog['key']:,.0f}, queue {prog['queue'] / 1e6:.1f} M  ({_dur(now - prog['t'])} ago)")
    else:
        done = [e for e in evs if e["kind"] == "stage_end" and e.get("depth") == 1]
        if done:
            lines.append("stages: " + ", ".join(f"{e['stage']} {_dur(e['seconds'])}" for e in done))
    mem = [e for e in evs if e["kind"] == "memory"]
    if mem:
        last = mem[-1]
        lines.append(f"memory: this run {last['private_gb']:.2f} GB (peak {max(e['private_gb'] for e in mem):.2f}), "
                     f"free {last['avail_gb']:.1f} GB (lowest {min(e['avail_gb'] for e in mem):.1f})")
    logs = [e["message"] for e in evs if e["kind"] == "log"]
    if logs:
        lines.append("last log lines:")
        lines += ["  " + l for m in logs[-3:] for l in str(m).splitlines()[:6]]
    return "\n".join(lines)
