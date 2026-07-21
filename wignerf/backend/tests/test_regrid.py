"""Auto-expand regrid: exactness of the fixed-lattice primitive, the
integer planner, and end-to-end expansion during live runs (including
mixed-geometry replay across the regrid boundary)."""

import json
import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import boundary, protocol
from core.grid import GridState, embed_window
from core.xp import ArrayBackend
from main import app

SPEC = protocol.GridSpec(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)


def test_regrid_exactness_move_and_double():
    b = ArrayBackend(device="cpu")
    gs = GridState.from_spec(SPEC)
    g = gs.make_grid(b)
    xv, pv = g.xv[:, None], g.pv[None, :]
    W = np.exp(-((xv - 1.0)**2/(2.*0.64) + (pv + 0.5)**2/(2.*0.81))) \
        / (2.*np.pi*0.8*0.9)

    # move by whole cells (+7 in x, -3 in p): lattice and values bitwise
    mv = replace(gs, ox=gs.ox + 7, op=gs.op - 3)
    gm = mv.make_grid(b)
    assert gm.dx == gs.dx and gm.dp == gs.dp        # frozen lattice, bitwise
    assert np.array_equal(gm.xv[:-7], g.xv[7:])     # same global cell, same float
    assert np.array_equal(gm.pv[3:], g.pv[:-3])
    Wm = embed_window(W, gs, mv, np)
    assert np.array_equal(Wm[:-7, 3:], W[7:, :-3])  # overlap copied bitwise
    assert not Wm[-7:, :].any() and not Wm[:, :3].any()   # entering cells zero
    # the state is well contained, so the dropped strips carry ~nothing
    dxdp = gs.dx*gs.dp
    assert abs(Wm.sum() - W.sum())*dxdp < 1e-10
    assert abs((Wm*Wm).sum() - (W*W).sum())*dxdp < 1e-12   # purity survives

    # double both axes, old window centered: pure zero-padding
    db = replace(gs, ox=gs.ox - 32, Nx=128, op=gs.op - 32, Np=128)
    gd = db.make_grid(b)
    assert gd.dx == gs.dx and gd.dp == gs.dp
    assert np.array_equal(gd.xv[32:96], g.xv)
    assert np.array_equal(gd.pv[32:96], g.pv)
    Wd = embed_window(W, gs, db, np)
    assert np.array_equal(Wd[32:96, 32:96], W)
    Wd[32:96, 32:96] = 0.0
    assert not Wd.any()                              # everything else is zero
    Wd = embed_window(W, gs, db, np)
    assert Wd.sum()*dxdp == pytest.approx(W.sum()*dxdp, rel=1e-13)
    assert (Wd*Wd).sum()*dxdp == pytest.approx((W*W).sum()*dxdp, rel=1e-13)


def test_support_cells_and_plan_axis():
    xv = np.linspace(-6, 6, 64, endpoint=False)
    dx = xv[1] - xv[0]
    rho = np.exp(-((xv - 2.0)**2)/(2.*0.25))
    rho /= rho.sum()*dx
    lo, hi = boundary.support_cells(rho, dx)
    assert xv[lo] < 2.0 - 2.0 and 2.0 + 2.0 < xv[hi - 1]   # covers +-4 sigma
    assert lo > 24 and hi < 62                             # but stays local
    # degenerate marginals fall back to the full window
    assert boundary.support_cells(np.zeros(64), dx) == (0, 64)
    nanrho = rho.copy()
    nanrho[0] = np.nan
    assert boundary.support_cells(nanrho, dx) == (0, 64)

    # recentring move: support fits in half the axis
    ap = boundary.plan_axis(0, 64, 30, 50, 4096)
    assert (ap.kind, ap.n, ap.offset) == ("move", 64, 8)   # center 40 kept
    # wide support: double and center (a combined move+double)
    ap = boundary.plan_axis(0, 64, 0, 60, 4096)
    assert (ap.kind, ap.n, ap.offset) == ("double", 128, -34)
    # at the cap a pure move still gives relief
    ap = boundary.plan_axis(0, 64, 20, 60, 64)
    assert (ap.kind, ap.n, ap.offset) == ("move", 64, 8)
    # hopeless at the cap (no margin left even recentered): capped
    ap = boundary.plan_axis(0, 64, 2, 62, 64)
    assert ap.kind == "capped" and (ap.offset, ap.n) == (0, 64)


def _mk(client, cfg_over=None, **over):
    cfg = {"grid": dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64),
           "potential": "0", "ic": {"type": "mixture", "components": [
               {"x0": -2.0, "p0": 2.0, "sigma_x": 0.707, "sigma_p": 0.707}]},
           "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0,
           "auto_expand": True}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def test_e2e_free_particle_expands():
    """A free packet (p0 = 2) drifts and spreads toward the +x edge; with
    auto_expand on the session must double the x-axis at a lockstep k_star,
    keep E/purity healthy across the switch (kinetic-only evolution is
    exact), and serve mixed-geometry history on both sides of it."""
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            regrid = None
            for _ in range(5000):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "regrid":
                        regrid = d
                        break
                    assert d["type"] != "error", d
            assert regrid is not None, "regrid never scheduled"
            assert regrid["kind"] == {"x": "double"}
            ng = regrid["grid"]
            assert ng["Nx"] == 128 and ng["Np"] == 64
            assert (ng["x2"] - ng["x1"])/ng["Nx"] == 12.0/64   # dx frozen
            assert ng["x1"] <= -10.0 and ng["x2"] >= 9.0
            k_star = regrid["at_record"]

            post = None
            for _ in range(5000):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.record >= k_star:
                        post = f
                        break
                elif m.get("text"):
                    assert json.loads(m["text"])["type"] != "error", m["text"]
            assert post is not None, "no post-regrid frame arrived"
            assert (post.geom.Nx, post.geom.x1, post.geom.x2) == \
                (128, ng["x1"], ng["x2"])
            assert len(post.variants) == 2      # lockstep held across the switch

            ws.send_text(json.dumps({"type": "pause"}))
            # replay across the boundary: exact per-record geometry both ways
            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            pre = None
            for _ in range(500):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])   # strict size check
                    if f.record == 0:
                        assert (f.geom.Nx, f.geom.x1) == (64, -6.0)
                        pre = f
                        break
            else:
                raise AssertionError("seek(0) frame never arrived")

            # E and purity healthy across the regrid (kinetic-only: exact)
            for v_pre, v_post in zip(pre.variants, post.variants):
                assert v_post.E == pytest.approx(v_pre.E, rel=1e-4)
                assert v_post.purity == pytest.approx(1.0, abs=1e-3)
            ws.send_text(json.dumps({"type": "seek", "record": post.record}))
            for _ in range(500):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.record == post.record:
                        assert f.geom.Nx == 128
                        break
            else:
                raise AssertionError("post-regrid seek frame never arrived")

            r = client.get("/api/sessions/%s" % sid).json()
            assert r["grid"]["Nx"] == 128       # status reports the live grid
        client.delete("/api/sessions/%s" % sid)


def test_combined_toggle_and_hbar_message_is_safe():
    """One set_params carrying BOTH auto_expand:true AND a larger hbar_eff:
    the immediate schedule must validate under the hbar that arrived WITH
    it. log(x+20) at hbar=1.6 is valid on the current window (extended
    range down to -17.5) but the planned leftward move (x1' ~ -10.1)
    pushes the range past the pole at -20 — the regrid must be refused
    with invalid_potential, never committed under the stale hbar=1 (which
    would fatally kill the worker at k_star)."""
    ic = {"type": "mixture", "components": [
        {"x0": -5.2, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0}]}
    with TestClient(app) as client:
        info = _mk(client, ic=ic, potential="log(x+20)", variants=["qn"],
                   auto_expand=False)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            for _ in range(50):
                m = ws.receive()
                if m.get("text") and json.loads(m["text"])["type"] == "boundary":
                    break
            else:
                raise AssertionError("boundary warning never arrived")
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True,
                                                "hbar_eff": 1.6}}))
            refused = applied = False
            for _ in range(50):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                assert d["type"] != "regrid", "regrid committed under stale hbar"
                if d["type"] == "boundary" and d["action"] == "invalid_potential":
                    refused = True
                if d["type"] == "params_applied":
                    assert d["applied"]["hbar_eff"] == 1.6
                    applied = True
                if refused and applied:
                    break
            assert refused and applied
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["hbar_eff"] == 1.6            # physics landed...
            assert r["grid"]["x1"] == -6.0         # ...the unsafe move did not
        client.delete("/api/sessions/%s" % sid)


def test_apply_params_guards_a_pending_plan_directly():
    """The session-level guard, exercised WITHOUT the streamer: a plan can
    commit during the streamer's ~ms validation compile, so apply_params
    itself must revalidate a pending plan's union under the incoming
    physics before any worker sees them (plan commits take the same lock,
    so nothing can slip in mid-check). Hitting that interleaving through
    the WS API is not deterministic — call apply_params directly instead."""
    from core.protocol import ParamChange
    from core.session import SESSIONS
    ic = {"type": "mixture", "components": [
        {"x0": 5.2, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0}]}
    with TestClient(app) as client:
        info = _mk(client, ic=ic, potential="log(20-x)", variants=["qn"],
                   auto_expand=True)
        sid = info["session_id"]
        s = SESSIONS[sid]
        with client.websocket_connect(info["ws_url"]):
            for _ in range(200):
                if s._regrid_plan is not None:
                    break
                time.sleep(0.02)
            assert s._regrid_plan is not None, "no plan committed"
            assert s._plan_pending(), "plan should still be in flight"
            before = s.cfg.hbar_eff

            # invalid on the union (extended range past the pole at x=20)
            s.apply_params(ParamChange(hbar_eff=1.6))
            assert s.cfg.hbar_eff == before, "unsafe hbar reached the workers"
            assert any(m.get("code") == "bad_potential" for m in s.msgs)
            # ...and a value that fits the union still applies
            s.apply_params(ParamChange(hbar_eff=1.1))
            assert s.cfg.hbar_eff == 1.1
        client.delete("/api/sessions/%s" % sid)


def test_hbar_rejected_against_pending_plan_union():
    """The streamer path for the same hazard: while a plan is pending,
    validation_grid() returns the old-new union, so an hbar change must be
    valid there. log(20-x) at hbar=1.6 is fine on the current window
    (extended up to 17.5) but not on the committed rightward move
    (x2' ~ 10.1 -> range past the pole at 20): reject; hbar=1.1 fits."""
    ic = {"type": "mixture", "components": [
        {"x0": 5.2, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0}]}
    with TestClient(app) as client:
        info = _mk(client, ic=ic, potential="log(20-x)", variants=["qn"],
                   auto_expand=False)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            for _ in range(50):
                m = ws.receive()
                if m.get("text") and json.loads(m["text"])["type"] == "boundary":
                    break
            else:
                raise AssertionError("boundary warning never arrived")
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            saw_regrid = False
            for _ in range(50):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "regrid":
                    saw_regrid = True
                # drain this message's own echo too, so the scan below sees
                # only what the hbar change produces
                if d["type"] == "params_applied" and saw_regrid:
                    break
            else:
                raise AssertionError("no regrid scheduled")
            assert saw_regrid, "no regrid scheduled"
            # plan stays pending (paused, workers idle): union is authoritative
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"hbar_eff": 1.6}}))
            for _ in range(50):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    assert d["type"] != "params_applied", d
                    if d["type"] == "error":
                        break
            else:
                raise AssertionError("invalid hbar_eff was not rejected")
            assert client.get("/api/sessions/%s" % sid).json()["hbar_eff"] == 1.0
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"hbar_eff": 1.1}}))
            for _ in range(50):
                m = ws.receive()
                if m.get("text") and \
                   json.loads(m["text"])["type"] == "params_applied":
                    break
            else:
                raise AssertionError("valid hbar_eff change not applied")
        client.delete("/api/sessions/%s" % sid)


def test_toggle_while_paused_schedules_immediately():
    """The 'warning fired -> pause -> enable auto-expand' flow must commit
    a regrid plan from the already-recorded edge state: the regrid event
    arrives while still paused, before a single record is computed on the
    old torus (waiting for the next report_edge would never fire here)."""
    ic = {"type": "mixture", "components": [
        {"x0": 5.2, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0}]}
    with TestClient(app) as client:
        info = _mk(client, ic=ic, auto_expand=False)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            # record 0 (the Cauchy data) trips the boundary without play
            for _ in range(50):
                m = ws.receive()
                if m.get("text") and json.loads(m["text"])["type"] == "boundary":
                    break
            else:
                raise AssertionError("boundary warning never arrived")
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            regrid = None
            for _ in range(50):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "regrid":
                        regrid = d
                        break
                    assert d["type"] != "error", d
            assert regrid is not None, "no regrid scheduled while paused"
            assert regrid["kind"] == {"x": "move"}
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["record_extent"][1] == 0      # nothing was computed
            assert r["grid"]["Nx"] == 64 and r["grid"]["x1"] > -6.0
        client.delete("/api/sessions/%s" % sid)


def test_e2e_warn_only_then_live_toggle_moves():
    """With auto_expand off a near-edge state only warns and the geometry
    stays put; flipping the toggle live (the key UX flow: warning fires,
    user opts in, the run is NOT lost) must produce a pure move — the
    support fits in half the axis, so Nx stays 64."""
    ic = {"type": "mixture", "components": [
        {"x0": 5.2, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0}]}
    with TestClient(app) as client:
        info = _mk(client, ic=ic, auto_expand=False)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            warned = False
            for _ in range(500):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "boundary":
                        assert d["axes"] == ["x"] and d["action"] == "warn"
                        warned = True
                        break
            assert warned, "boundary warning never arrived"
            ws.send_text(json.dumps({"type": "play"}))

            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            regrid = None
            for _ in range(2000):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "regrid":
                        regrid = d
                        break
                    assert d["type"] != "error", d
            assert regrid is not None, "toggle did not lead to a regrid"
            assert regrid["kind"] == {"x": "move"}
            ng = regrid["grid"]
            assert ng["Nx"] == 64 and ng["Np"] == 64
            assert ng["x1"] > -6.0 and ng["x2"] > 6.0     # window slid right
            assert (ng["x2"] - ng["x1"])/64 == 12.0/64    # dx frozen
        client.delete("/api/sessions/%s" % sid)
