"""The event stream (events.jsonl), search progress from the Rust core, and `roadsearch status`."""

import numpy as np
import pytest

from roadsearch import search
from roadsearch.cli import main
from roadsearch.events import read
from roadsearch.status import summarize

from .test_run import args, dem_file


def test_run_writes_events_and_status_reads_them(tmp_path, monkeypatch, capsys):
    pytest.importorskip("roadsearch_core")
    monkeypatch.setattr(search, "ENGINE", "rust")
    monkeypatch.setattr(search, "PROGRESS_EVERY", 2000)          # small searches report too
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    assert main(args(path, s, g, out)) == 0
    run_dir = next((out / "runs").iterdir())
    evs = read(run_dir / "events.jsonl")
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and evs[-1]["status"] == "finished"
    starts = [e["stage"] for e in evs if e["kind"] == "stage_start"]
    ends = [e["stage"] for e in evs if e["kind"] == "stage_end"]
    assert sorted(starts) == sorted(ends) and "corridor fields (two full Dijkstra)" in starts
    prog = [e for e in evs if e["kind"] == "progress"]
    assert prog and all(e["settled"] % 2000 == 0 and e["stage"] for e in prog)
    assert any(e["kind"] == "log" and "rank 1" in e["message"] for e in evs)
    text = summarize(out)
    assert run_dir.name in text and "finished after" in text and "stages:" in text


def test_status_of_a_run_under_way(tmp_path):
    from roadsearch.events import EventLog

    log = EventLog(tmp_path / "events.jsonl")
    t = 1_000_000.0
    for e in [dict(kind="run_start", run_id="abc"), dict(kind="stage_start", stage="run", depth=0),
              dict(kind="stage_start", stage="A* search 1", depth=1),
              dict(kind="progress", stage="A* search 1", settled=3_000_000, key=1.5e7, queue=2_000_000, rate=250_000),
              dict(kind="memory", avail_gb=5.0, private_gb=2.0), dict(kind="log", message="working grid ...")]:
        log.emit(**e)
    log.close()
    evs = read(tmp_path / "events.jsonl")
    text = summarize(tmp_path, now=evs[-1]["t"] + 5)
    assert "running for" in text and "stage: run > A* search 1" in text and "3.0 M states settled" in text
    assert "may have stopped" in summarize(tmp_path, now=evs[-1]["t"] + 600)


def test_progress_callback_can_stop_the_rust_search():
    roadsearch_core = pytest.importorskip("roadsearch_core")
    from roadsearch.params import load_params
    from roadsearch.preprocess import preprocess
    from roadsearch.search import make_problem, rust_args
    from .conftest import make_grid
    from .test_search import random_terrain

    p = load_params()
    z = random_terrain(4, n=60, relief=5.0)
    g = make_grid(z, 5.0)
    prob = make_problem(preprocess(g, g.xy(5, 5), g.xy(54, 50), p), p, hairpins=False)
    lat, terr = rust_args(prob)
    src = np.array([(5 * prob.nc + 5) * prob.n_theta], dtype=np.int64)
    calls = []

    def stop_after_three(n, key, queue):
        calls.append(n)
        return len(calls) < 3

    out = roadsearch_core.search(prob.n_theta, lat, terr, src, -1, -1, False, np.inf, stop_after_three, 500)
    assert out[6] is False and out[4] == 1500 and calls == [500, 1000, 1500]
    with pytest.raises(ZeroDivisionError):
        roadsearch_core.search(prob.n_theta, lat, terr, src, -1, -1, False, np.inf, lambda *a: 1 / 0, 500)
