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
           "variants": list(variants), "record_dt": 0.05, "rate": 4.0}
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
            assert rec0[0] == 0 and rec0[1] == 0.0

            ws.send_text(json.dumps({"type": "play"}))
            frames = _recv_frames(ws, 6)
            recs = [f[0] for f in frames]
            ts = {f[0]: f[1] for f in frames}
            assert recs == sorted(recs) and len(set(recs)) == len(recs)
            for n, t in ts.items():
                assert t == pytest.approx(n*0.05, abs=1e-12)

            v = frames[-1][5][0]
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
                    if f[0] == 0:
                        assert f[1] == 0.0
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
            frontier = client.get("/api/sessions/%s" % sid).json()["record_extent"][1]
            assert frontier >= 5

            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            ws.send_text(json.dumps({"type": "play"}))
            paused = None
            for _ in range(100):                # replay runs at 80 records/s
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
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    k = protocol.unpack_frame(m["bytes"])[0]
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


def test_two_variant_lockstep():
    with TestClient(app) as client:
        info = _mk(client, variants=("qn", "cn"))
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            frames = _recv_frames(ws, 5)
            for rec, t, Nx, Np, flags, variants in frames:
                assert len(variants) == 2      # one bundle, both variants, same t
                vids = {v.vid for v in variants}
                assert vids == {protocol.variant_id(True, False),
                                protocol.variant_id(False, False)}
            # harmonic oscillator: quantum == classical -> identical scalars
            v1, v2 = frames[-1][5]
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
                    if f[4] & protocol.FLAG_LIVE_PREVIEW:
                        saw_preview = True
                    if f[0] == 10:
                        assert f[1] == pytest.approx(0.5, abs=1e-9)
                        break
            else:
                raise AssertionError("run-ahead never reached t2")
            assert saw_preview
            # ... and STOPS at t2: the frontier must not advance past it
            _time.sleep(0.5)
            r = client.get("/api/sessions/%s" % info["session_id"]).json()
            assert r["record_extent"][1] == 10, "computed past t2"
            # the whole timeline is scrubbable immediately
            ws.send_text(json.dumps({"type": "seek", "record": 4}))
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes") and protocol.unpack_frame(m["bytes"])[0] == 4:
                    break
            else:
                raise AssertionError("scrub into computed run-ahead failed")
        client.delete("/api/sessions/%s" % info["session_id"])


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
                if f[0] > newest[0] + 2 and f[1] < newest[1]:
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
        # unknown session
        assert client.get("/api/sessions/nope").status_code == 404
