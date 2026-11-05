"""An OptKing start that asv stops at the class timeout is searched once."""

import json
import os
import signal

import pytest

from benchmarks import bench_optking
from optbench.runners import optking_ts

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="asv-runner forks each benchmark")

TRACKS = (
    "track_success",
    "track_failure_crash",
    "track_failure_divergence",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_force_calls",
)


def _asv_call(method):
    """Run one benchmark method in a forked child, as asv-runner does. ``None`` when killed."""
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader)
            benchmark = bench_optking.OptkingLj38PRfo()
            benchmark.setup("0", "0000")
            try:
                outcome = {"value": getattr(benchmark, method)("0", "0000")}
            except Exception as exc:
                outcome = {"error": f"{type(exc).__name__}: {exc}"}
            os.write(writer, json.dumps(outcome).encode())
        finally:
            os._exit(0)
    os.close(writer)
    with os.fdopen(reader, "rb") as stream:
        data = stream.read()
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        assert os.WTERMSIG(status) == signal.SIGKILL
        return None
    return json.loads(data)


@pytest.fixture
def record_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(bench_optking, "ensure_optking", lambda: None)
    monkeypatch.setattr(optking_ts, "load_jobs", lambda *args, **kwargs: [{"start": "0000"}])
    return bench_optking._cache_path("optking/lj38-p-rfo", "0", "0000")


def _killed_at_the_timeout(searches):
    def search(job):
        with searches.open("a", encoding="utf-8") as handle:
            handle.write(job["start"] + "\n")
        # asv-runner sends SIGTERM and then SIGKILL; the search process runs nothing after that.
        os.kill(os.getpid(), signal.SIGKILL)

    return search


def test_a_search_killed_at_the_timeout_is_recorded_once_for_every_track(tmp_path, monkeypatch, record_path):
    searches = tmp_path / "searches.txt"
    monkeypatch.setattr(optking_ts, "search_start", _killed_at_the_timeout(searches))

    assert _asv_call("time_search") is None
    results = {}
    inodes = []
    for method in TRACKS:
        results[method] = _asv_call(method)
        inodes.append(record_path.stat().st_ino if record_path.exists() else None)

    assert searches.read_text(encoding="utf-8").split() == ["0000"]
    assert inodes[0] is not None
    assert set(inodes) == {inodes[0]}
    assert results == {
        "track_success": {"value": 0.0},
        "track_failure_crash": {"value": 0.0},
        "track_failure_divergence": {"value": 1.0},
        "track_failure_cap": {"value": 0.0},
        "track_failure_wrong_stationary": {"value": 0.0},
        "track_force_calls": {"error": "RuntimeError: start 0000 timed out, so it has no force count"},
    }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["success"] is False
    assert record["force_calls"] is None
    assert record["reasons"] == ["timeout"]
    assert not record_path.with_suffix(".started").exists()


def test_a_killed_search_does_not_leave_an_older_record_in_place(tmp_path, monkeypatch, record_path):
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps({"force_calls": 301, "success": True, "failure": None}), encoding="utf-8")
    monkeypatch.setattr(optking_ts, "search_start", _killed_at_the_timeout(tmp_path / "searches.txt"))

    assert _asv_call("time_search") is None
    assert _asv_call("track_success") == {"value": 0.0}


def test_a_finished_search_is_read_back_without_a_marker(monkeypatch, record_path):
    def finished(job):
        return {"start": job["start"], "force_calls": 301, "success": True, "failure": None, "reasons": []}

    def searched_again(job):
        raise AssertionError("the record was not reused")

    monkeypatch.setattr(optking_ts, "search_start", finished)
    assert _asv_call("time_search") == {"value": None}
    monkeypatch.setattr(optking_ts, "search_start", searched_again)
    assert _asv_call("track_force_calls") == {"value": 301.0}
    assert json.loads(record_path.read_text(encoding="utf-8"))["seed"] == "0"
    assert not record_path.with_suffix(".started").exists()
