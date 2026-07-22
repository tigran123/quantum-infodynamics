"""End-to-end streaming tests via the starlette TestClient (headless)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import protocol
from core.quantize import dequantize
from main import app

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
IC = {"type": "mixture",
      "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707, "sigma_p": 0.707}]}


def _mk(client, variants=("qn",), **over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC,
           "variants": list(variants), "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _recv_frames(ws, n, max_msgs=500):
    frames = []
    for _ in range(max_msgs):
        m = ws.receive()
        if m.get("bytes"):
            frames.append(protocol.unpack_frame(m["bytes"]))
            if len(frames) == n:
                return frames
    raise AssertionError("only %d/%d frames received" % (len(frames), n))


def test_stream_play_pause_seek():
    with TestClient(app) as client:
        info = _mk(client)
        with client.websocket_connect(info["ws_url"]) as ws:
            # record 0 (the Cauchy data) arrives even before play
            (rec0,) = _recv_frames(ws, 1)
            assert rec0.record == 0 and rec0.t == 0.0
            assert (rec0.geom.x1, rec0.geom.x2) == (-6.0, 6.0)
            assert (rec0.geom.p1, rec0.geom.p2) == (-7.0, 7.0)

            ws.send_text(json.dumps({"type": "play"}))
            frames = _recv_frames(ws, 6)
            recs = [f.record for f in frames]
            ts = {f.record: f.t for f in frames}
            assert recs == sorted(recs) and len(set(recs)) == len(recs)
            for n, t in ts.items():
                assert t == pytest.approx(n*0.05, abs=1e-12)

            v = frames[-1].variants[0]
            W = dequantize(v.wq, v.wmin, v.wmax)
            assert abs(W.sum()*(12./64)*(14./64) - 1.0) < 1e-3
            assert v.E == pytest.approx(2.5, abs=2e-3)
            # coherent state: purity == 1, conserved by the unitary flow
            assert v.purity == pytest.approx(1.0, abs=1e-3)

            ws.send_text(json.dumps({"type": "pause"}))
            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.record == 0:
                        assert f.t == 0.0
                        break
            else:
                raise AssertionError("seek(0) frame never arrived")
        assert client.delete("/api/sessions/%s" % info["session_id"]).json()["ok"]


def test_interactive_playback_stops_at_frontier():
    """Play pressed behind the frontier is playback-only: it must replay
    history, auto-pause AT the frontier without computing a single new
    record, and leave resumption of computation to an explicit play at the
    frontier (the transport button's "Solve")."""
    import time as _time
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 6)
            ws.send_text(json.dumps({"type": "pause"}))
            _time.sleep(0.3)                    # let in-flight records land
            r = client.get("/api/sessions/%s" % sid).json()
            frontier = r["record_extent"][1]
            assert frontier >= 5
            # a paused solve settles ATTACHED at the final frontier — the
            # transport must come to rest on "Solve", not a phantom "Play"
            # over the in-flight records that landed after the pause
            assert r["cursor"] == pytest.approx(frontier)

            # workers run flat out now, so the frontier can be large —
            # replay at zero delay (max speed) to keep the test fast
            ws.send_text(json.dumps({"type": "delay", "seconds": 0}))
            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            ws.send_text(json.dumps({"type": "play"}))
            paused = None
            for _ in range(300):
                _time.sleep(0.05)
                r = client.get("/api/sessions/%s" % sid).json()
                if not r["running"]:
                    paused = r
                    break
            assert paused is not None, "playback never auto-paused"
            assert paused["record_extent"][1] == frontier, \
                "playback rolled into computation"
            assert paused["cursor"] == pytest.approx(frontier)

            # the replay arrived in exact sequence and ends on the frontier
            seen = []
            for _ in range(20*frontier + 200):
                m = ws.receive()
                if m.get("bytes"):
                    k = protocol.unpack_frame(m["bytes"]).record
                    seen.append(k)
                    if k == frontier and 0 in seen:
                        break
            tail = seen[seen.index(0):]
            assert tail == sorted(tail) and tail[-1] == frontier

            # play AT the frontier (= Solve) resumes computation
            ws.send_text(json.dumps({"type": "play"}))
            for _ in range(100):
                _time.sleep(0.05)
                r = client.get("/api/sessions/%s" % sid).json()
                if r["record_extent"][1] > frontier:
                    break
            else:
                raise AssertionError("Solve at the frontier did not compute")
        client.delete("/api/sessions/%s" % sid)


def test_solve_speed_independent_of_delay():
    """The delay dial paces the DISPLAY only: solving at the frontier must
    run flat out even at the dial's slowest setting (1 s per frame)."""
    import time as _time
    with TestClient(app) as client:
        info = _mk(client, delay=1.0)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _time.sleep(1.0)
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["record_extent"][1] > 10, \
                "computation was throttled by the display delay"
        client.delete("/api/sessions/%s" % sid)


def test_pause_of_a_solve_keeps_the_display_attached():
    """Pausing a solve leaves in-flight records landing AFTER the pause;
    the attached cursor must follow the settling frontier so the transport
    comes to rest AT the frontier (button: "Solve") — never a phantom
    "Play" over a few records the user did not rewind to."""
    from core.session import SessionClock
    clock = SessionClock(0.0, 0.05, "interactive", 0.0, None)
    clock.set_running(True, 0)             # Solve at the frontier
    clock.advance_cursor(0.05, 10, 10)     # solving: pinned to the frontier
    assert clock.cursor == 10
    clock.set_running(False)               # pause; 2 in-flight records land
    assert clock.advance_cursor(0.05, 12, 10) == 12, \
        "paused attached cursor must follow the settling frontier"
    clock.set_cursor(5, 12)                # rewind while paused: detached
    assert clock.advance_cursor(0.05, 12, 5) == 5, \
        "a rewound (detached) cursor must stay put while paused"


def test_clock_pause_is_delivery_aware():
    """A playback-only run pauses only once the frontier record was
    actually DELIVERED: at delay 0 the cursor reaches the frontier
    instantly, and time spent blocked in a send to a slow client must not
    end the run over records the client never saw."""
    from core.session import SessionClock
    clock = SessionClock(0.0, 0.05, "interactive", 0.0, None)
    clock.set_cursor(92, 613)
    clock.set_running(True, 613)
    assert clock.stop_at_frontier
    clock.advance_cursor(30.0, 613, 150)     # cursor jumps to the frontier
    assert clock.running, "paused while records 151..613 were still unsent"
    clock.advance_cursor(0.1, 613, 613)      # frontier record delivered
    assert not clock.running and clock.cursor == 613


def test_playback_zero_delay_never_skips():
    """Delay 0 (the default) sends the cursor to the frontier immediately.
    Replay must still deliver every record in exact order and then
    auto-pause — never coalesce over the unsent gap (that used to teleport
    playback straight to the end)."""
    import time as _time
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 6)
            ws.send_text(json.dumps({"type": "pause"}))
            _time.sleep(0.3)                    # let in-flight records land
            frontier = client.get("/api/sessions/%s" % sid).json()["record_extent"][1]
            assert frontier >= 5

            ws.send_text(json.dumps({"type": "delay", "seconds": 0}))
            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            for _ in range(200):                # wait out the seek echo
                m = ws.receive()
                if m.get("bytes") and protocol.unpack_frame(m["bytes"]).record == 0:
                    break
            else:
                raise AssertionError("seek(0) frame never arrived")
            ws.send_text(json.dumps({"type": "play"}))
            seen = []
            for _ in range(20*frontier + 400):
                m = ws.receive()
                if m.get("bytes"):
                    seen.append(protocol.unpack_frame(m["bytes"]).record)
                    if seen[-1] == frontier:
                        break
            assert seen == list(range(1, frontier + 1)), \
                "replay skipped records: %r" % seen

            # ... and auto-paused at the frontier without computing
            r = None
            for _ in range(100):
                _time.sleep(0.05)
                r = client.get("/api/sessions/%s" % sid).json()
                if not r["running"]:
                    break
            assert r is not None and not r["running"]
            assert r["record_extent"][1] == frontier, \
                "zero-delay playback rolled into computation"
        client.delete("/api/sessions/%s" % sid)


def test_two_variant_lockstep():
    with TestClient(app) as client:
        info = _mk(client, variants=("qn", "cn"))
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            frames = _recv_frames(ws, 5)
            for f in frames:
                assert len(f.variants) == 2    # one bundle, both variants, same t
                vids = {v.vid for v in f.variants}
                assert vids == {protocol.variant_id(True, False),
                                protocol.variant_id(False, False)}
            # harmonic oscillator: quantum == classical -> identical scalars
            v1, v2 = frames[-1].variants
            assert v1.E == pytest.approx(v2.E, abs=1e-6)
            assert v1.x_mean == pytest.approx(v2.x_mean, abs=1e-6)
        client.delete("/api/sessions/%s" % info["session_id"])


def test_set_params_live_and_rejected():
    with TestClient(app) as client:
        info = _mk(client)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 2)
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"U": "x^4/4"}}))
            saw_applied = False
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "params_applied":
                        assert d["applied"]["U"] == "x^4/4"
                        saw_applied = True
                        break
            assert saw_applied
            # a quantum-invalid potential must be rejected with an error msg
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"U": "1/x"}}))
            saw_error = False
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "error":
                        saw_error = True
                        break
            assert saw_error
        client.delete("/api/sessions/%s" % info["session_id"])


def test_no_op_params_are_dropped():
    """A change that changes nothing must leave no trace. The UI sends whole
    fields — PotentialEditor's "Apply live" always carries the U(x) draft,
    edited or not — and a param_log full of U changes that never happened
    makes an exported video's "how to reproduce this" block a lie."""
    from core.session import SESSIONS
    with TestClient(app) as client:
        info = _mk(client)
        s = SESSIONS[info["session_id"]]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 2)
            ws.send_text(json.dumps({"type": "pause"}))
            # every field identical to what is live
            ws.send_text(json.dumps({"type": "set_params", "params": {
                "U": s.cfg.potential, "mass": s.cfg.mass, "c": s.cfg.c,
                "hbar_eff": s.cfg.hbar_eff, "tol": s.cfg.tol,
                "dt_sign": s.clock.sign, "auto_expand": s.auto_expand}}))
            # ...then one real change, whose arrival proves the first was seen
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"U": s.cfg.potential,
                                                "hbar_eff": 0.5}}))
            for _ in range(400):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "params_applied":
                        break
            else:
                raise AssertionError("params_applied never arrived")
            assert d["applied"] == {"hbar_eff": 0.5}     # U dropped: unchanged
            assert d["before"] == {"hbar_eff": 1.0}
            assert [e["applied"] for e in s.param_log] == [{"hbar_eff": 0.5}]
        client.delete("/api/sessions/%s" % info["session_id"])


def test_hbar_change_revalidates_potential():
    """Raising hbar_eff widens the extended Bopp range; the CURRENT U must
    stay valid there or the change is rejected up front — worker-side
    rollback cannot save a pending regrid (its non-finite check is fatal,
    lockstep geometry must stay uniform)."""
    with TestClient(app) as client:
        # valid at hbar=1 (extended range ~[-13.2, 13.2], singular at -20)
        info = _mk(client, potential="log(x+20)")
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 2)
            # hbar=4 -> extended range ~[-34.7, 34.7]: crosses the pole
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"hbar_eff": 4.0}}))
            saw_error = False
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "error":
                        saw_error = True
                        break
                    assert d["type"] != "params_applied", d
            assert saw_error, "invalid hbar_eff change was not rejected"
            r = client.get("/api/sessions/%s" % info["session_id"]).json()
            assert r["hbar_eff"] == 1.0            # unchanged
            # shrinking the range is always fine
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"hbar_eff": 0.5}}))
            for _ in range(200):
                m = ws.receive()
                if m.get("text") and \
                   json.loads(m["text"])["type"] == "params_applied":
                    break
            else:
                raise AssertionError("valid hbar_eff change not applied")
        client.delete("/api/sessions/%s" % info["session_id"])


def test_series_backfill():
    with TestClient(app) as client:
        info = _mk(client)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 5)
            sid = info["session_id"]
            d = client.get("/api/sessions/%s/series" % sid).json()
            ns = [r["n"] for r in d["records"]]
            assert ns == list(range(len(ns)))          # gapless
            assert all(len(r["variants"]) == 1 for r in d["records"])
            es = [r["variants"][0]["E"] for r in d["records"]]
            assert max(es) - min(es) < 1e-3            # harmonic: E flat
        client.delete("/api/sessions/%s" % info["session_id"])


def test_runahead_starts_paused_and_stops_at_t2():
    with TestClient(app) as client:
        info = _mk(client, mode="runahead", t2=0.5)   # 10 records at 0.05
        with client.websocket_connect(info["ws_url"]) as ws:
            # sessions start PAUSED in both modes: without play, only the
            # Cauchy record exists
            import time as _time
            _time.sleep(0.4)
            r = client.get("/api/sessions/%s" % info["session_id"]).json()
            assert r["record_extent"][1] == 0, "computed without being asked"
            assert r["t2"] == 0.5

            # play: workers run flat-out; the preview stream carries the
            # newest lockstep-complete record
            ws.send_text(json.dumps({"type": "play"}))
            saw_preview = False
            for _ in range(400):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.flags & protocol.FLAG_LIVE_PREVIEW:
                        saw_preview = True
                    if f.record == 10:
                        assert f.t == pytest.approx(0.5, abs=1e-9)
                        break
            else:
                raise AssertionError("run-ahead never reached t2")
            assert saw_preview
            # ... and STOPS at t2: the frontier must not advance past it,
            # and the RUN must end — the workers idle from here on, so a
            # still-"running" clock would freeze the transport button on
            # "Pause" forever and lock out every paused-only action
            _time.sleep(0.5)
            r = client.get("/api/sessions/%s" % info["session_id"]).json()
            assert r["record_extent"][1] == 10, "computed past t2"
            assert not r["running"], "finished run-ahead still reports running"
            # the whole timeline is scrubbable immediately
            ws.send_text(json.dumps({"type": "seek", "record": 4}))
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes") and protocol.unpack_frame(m["bytes"]).record == 4:
                    break
            else:
                raise AssertionError("scrub into computed run-ahead failed")
        client.delete("/api/sessions/%s" % info["session_id"])


def test_runahead_rewind_plays_back_without_computing():
    """Pausing a run-ahead mid-run and rewinding must offer pure playback:
    play behind the frontier replays history gaplessly and auto-pauses AT
    the frontier — it must never jump to the end nor resume computing
    toward t2. Only play AT the frontier resumes the run-ahead."""
    import time as _time
    with TestClient(app) as client:
        info = _mk(client, mode="runahead", t2=100.0)   # far away: never done
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 4)
            ws.send_text(json.dumps({"type": "pause"}))
            _time.sleep(0.3)                    # let in-flight records land
            frontier = client.get("/api/sessions/%s" % sid).json()["record_extent"][1]
            assert 3 <= frontier < 1999, "expected a mid-run pause"

            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            for _ in range(200):                # wait out the seek echo
                m = ws.receive()
                if m.get("bytes") and protocol.unpack_frame(m["bytes"]).record == 0:
                    break
            else:
                raise AssertionError("seek(0) frame never arrived")
            ws.send_text(json.dumps({"type": "play"}))
            seen = []
            for _ in range(20*frontier + 400):
                m = ws.receive()
                if m.get("bytes"):
                    seen.append(protocol.unpack_frame(m["bytes"]).record)
                    if seen[-1] == frontier:
                        break
            assert seen == list(range(1, frontier + 1)), \
                "runahead playback skipped records: %r" % seen[:20]
            r = None
            for _ in range(100):
                _time.sleep(0.05)
                r = client.get("/api/sessions/%s" % sid).json()
                if not r["running"]:
                    break
            assert r is not None and not r["running"]
            assert r["record_extent"][1] == frontier, \
                "rewound runahead playback resumed computing toward t2"

            # play AT the frontier resumes the run-ahead
            ws.send_text(json.dumps({"type": "play"}))
            for _ in range(100):
                _time.sleep(0.05)
                r = client.get("/api/sessions/%s" % sid).json()
                if r["record_extent"][1] > frontier:
                    break
            else:
                raise AssertionError("Solve at the frontier did not resume")
        client.delete("/api/sessions/%s" % sid)


def test_time_reversal_over_the_wire():
    with TestClient(app) as client:
        info = _mk(client)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            frames = _recv_frames(ws, 4)
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"dt_sign": -1}}))
            # records keep increasing append-only, but t must start falling
            newest = frames[-1]
            for _ in range(500):
                m = ws.receive()
                if not m.get("bytes"):
                    continue
                f = protocol.unpack_frame(m["bytes"])
                if f.record > newest.record + 2 and f.t < newest.t:
                    break   # t decreased on a later record
            else:
                raise AssertionError("time never reversed")
        client.delete("/api/sessions/%s" % info["session_id"])


def test_session_validation():
    with TestClient(app) as client:
        # quantum-invalid potential rejected at creation
        cfg = {"grid": GRID, "potential": "1/x", "ic": IC, "variants": ["qn"]}
        assert client.post("/api/sessions", json=cfg).status_code == 422
        # empty variant list rejected by schema
        cfg = {"grid": GRID, "potential": "x^2", "ic": IC, "variants": []}
        assert client.post("/api/sessions", json=cfg).status_code == 422
        # mass = 0 with a non-relativistic variant would stream NaN frames
        # (T = p^2/2m); only exclusively relativistic runs may be massless
        cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC,
               "variants": ["qn"], "mass": 0.0}
        assert client.post("/api/sessions", json=cfg).status_code == 422
        cfg["variants"] = ["qr"]
        r = client.post("/api/sessions", json=cfg)
        assert r.status_code == 200, r.text
        client.delete("/api/sessions/%s" % r.json()["session_id"])
        # unknown session
        assert client.get("/api/sessions/nope").status_code == 404


def test_grid_cap_enforced(monkeypatch):
    """The per-axis ceiling is WIGNERF_MAX_GRID (env-tunable BOTH ways);
    the pydantic le=16384 is only a sanity rail behind it."""
    import config as appconfig
    monkeypatch.setattr(appconfig, "MAX_GRID", 512)
    with TestClient(app) as client:
        cfg = {"grid": dict(GRID, Nx=1024), "potential": "x^2/2", "ic": IC,
               "variants": ["qn"]}
        r = client.post("/api/sessions", json=cfg)
        assert r.status_code == 422 and "WIGNERF_MAX_GRID" in r.text
        cfg["grid"] = dict(GRID, Nx=512, Np=512)
        r = client.post("/api/sessions", json=cfg)
        assert r.status_code == 200, r.text
        assert client.get("/api/sessions/%s"
                          % r.json()["session_id"]).json()["max_grid"] == 512
        client.delete("/api/sessions/%s" % r.json()["session_id"])
        # raising the env cap unlocks sizes past the old 4096 limit...
        monkeypatch.setattr(appconfig, "MAX_GRID", 16384)
        cfg["grid"] = dict(GRID, Nx=8192)
        r = client.post("/api/sessions", json=cfg)
        assert r.status_code == 200, r.text
        client.delete("/api/sessions/%s" % r.json()["session_id"])
        # ...but the schema rail still holds
        cfg["grid"] = dict(GRID, Nx=32768)
        assert client.post("/api/sessions", json=cfg).status_code == 422


def test_lockstep_skew_gate():
    """A worker may run at most SKEW_MARGIN records past the record the
    lockstep frontier is waiting on; the slowest worker (frontier ==
    latest_complete) is never gated, so the gate cannot deadlock."""
    from core.session import SessionClock, SKEW_MARGIN
    clock = SessionClock(0.0, 0.05, "interactive", 0.0, None)
    clock.set_running(True, 0)
    # slowest worker: always gets its next target
    assert clock.next_target(5, 5) == (6, pytest.approx(0.3))
    # fast worker at the gate edge: allowed...
    assert clock.next_target(5 + SKEW_MARGIN, 5) is not None
    # ...one past it: idles until the frontier advances
    assert clock.next_target(5 + SKEW_MARGIN + 1, 5) is None
    assert clock.next_target(5 + SKEW_MARGIN + 1, 6) is not None


def test_worker_skew_is_bounded(monkeypatch):
    """The history byte cap is only enforceable if incomplete records above
    the lockstep frontier stay bounded: with one variant artificially slow,
    the fast one must not run away."""
    import time as _time
    from core.session import SESSIONS, SKEW_MARGIN
    from core.worker import SolverWorker
    orig = SolverWorker._advance

    def slowed(self, prop, W, t, t_tgt):
        if self.key == "qn":
            _time.sleep(0.05)
        return orig(self, prop, W, t, t_tgt)

    monkeypatch.setattr(SolverWorker, "_advance", slowed)
    with TestClient(app) as client:
        info = _mk(client, variants=("qn", "cn"))
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 3)                  # frontier is moving
            s = SESSIONS[sid]
            for _ in range(10):
                fr = [s.history.variant_frontier(i) for i in range(2)]
                assert max(fr) - min(fr) <= SKEW_MARGIN + 1, \
                    "fast variant ran away: frontiers %r" % (fr,)
                _time.sleep(0.05)
            assert s.history.latest_complete() >= 3   # and progress was made
        client.delete("/api/sessions/%s" % sid)


def test_live_params_tracked_in_status():
    """Live parameter changes must be visible in status() and used for
    later U validations (the extended Bopp range depends on hbar_eff)."""
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            _recv_frames(ws, 2)
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"hbar_eff": 0.5, "tol": 0.02}}))
            for _ in range(200):
                m = ws.receive()
                if m.get("text") and \
                   json.loads(m["text"])["type"] == "params_applied":
                    break
            else:
                raise AssertionError("params_applied never arrived")
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["hbar_eff"] == 0.5 and r["tol"] == 0.02
            assert r["mass"] == 1.0        # untouched params keep their value
        client.delete("/api/sessions/%s" % sid)
