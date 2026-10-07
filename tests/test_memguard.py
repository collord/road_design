"""Memory guard: refusing grids that do not fit, and stopping a run when memory runs short."""

import json
import sys
import time

import pytest

from roadsearch import memguard, timing
from roadsearch.cli import main

from .test_run import args, dem_file


def test_snapshot_reports_memory():
    s = memguard.snapshot()
    assert s["avail_gb"] > 0 and s["private_gb"] > 0
    if sys.platform == "win32":
        assert s["kernel_nonpaged_gb"] > 0
        tags = memguard.pool_tags(3)
        assert len(tags) == 3 and all(len(t["tag"]) == 4 for t in tags)
        assert tags[0]["nonpaged_mb"] >= tags[-1]["nonpaged_mb"]


def test_require_refuses_what_does_not_fit():
    memguard.require(0.01, "small grid", 0.0)
    with pytest.raises(memguard.MemoryGuardError, match="huge grid needs"):
        memguard.require(1e6, "huge grid", 0.0)


def test_watchdog_interrupts_main_thread():
    wd = memguard.Watchdog(floor_gb=1e6, every=0.02)
    with pytest.raises(KeyboardInterrupt):
        with wd:
            t0 = time.time()
            while time.time() - t0 < 10:
                time.sleep(0.01)
    assert "available memory fell" in wd.reason
    assert wd.peak["min_avail_gb"] > 0


def test_stage_rows_record_memory():
    prof = timing.start()
    try:
        with timing.stage("work"):
            pass
    finally:
        timing.stop()
    assert prof.rows[0]["avail_gb"] > 0 and prof.rows[0]["private_gb"] > 0
    assert "free GB" in prof.table()


def test_run_refuses_grid_larger_than_free_memory(tmp_path, monkeypatch):
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(memguard, "FLOOR_GB", 1e6)
    assert main(args(path, s, g, out)) == 2
    rec = json.loads(next((out / "runs").glob("*/params.json")).read_text())
    assert rec["error"].startswith("MemoryGuardError") and "working grid needs" in rec["error"]


def test_run_stopped_by_watchdog_records_why(tmp_path, monkeypatch):
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(memguard, "require", lambda *a, **k: None)     # past the preflight check
    monkeypatch.setattr(memguard, "FLOOR_GB", 1e6)                      # then short of memory at once
    monkeypatch.setattr(memguard, "EVERY_S", 0.05)
    assert main(args(path, s, g, out)) == 3
    rec = json.loads(next((out / "runs").glob("*/params.json")).read_text())
    assert rec["error"].startswith("MemoryGuardError: available memory fell")
    assert rec["memory"]["min_avail_gb"] > 0 and rec["profile"][0]["stage"] == "run"
    # With memory to spare, the same run finishes.
    monkeypatch.setattr(memguard, "FLOOR_GB", 0.0)
    assert main(args(path, s, g, out)) == 0
