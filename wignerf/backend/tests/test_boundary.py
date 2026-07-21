"""Boundary watch (Phase: detection + events): edge-band unit checks, the
'boundary' WS event, status fields, the live auto_expand toggle, and the
IC-preview measure-based edge warning."""

import json
from urllib.parse import unquote

import numpy as np
from fastapi.testclient import TestClient

from core import boundary
from main import app

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
EDGE_IC = {"type": "mixture",
           "components": [{"x0": 5.2, "p0": 0.0, "sigma_x": 0.5,
                           "sigma_p": 0.707}]}


def _mk(client, **over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": EDGE_IC,
           "variants": ["qn"], "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _gauss(v, x0, s):
    g = np.exp(-((v - x0)**2)/(2.*s*s))
    return g/(g.sum()*(v[1] - v[0]))


def test_edge_report_degenerate_axes_stay_quiet():
    """On axes shorter than 8 bands (N < 32) the band pair covers over a
    quarter of the axis — at N <= 8 the slices overlap and a uniform
    marginal would read as edge mass 2.0. Such axes must never trigger
    (they would otherwise warn always and auto-expand-storm to the cap)."""
    for n in (4, 8, 16):
        m = np.full(n, 1.0/(n*0.1))      # normalized uniform density, d=0.1
        es = boundary.edge_report(m, m, 0.1, 0.1)
        assert (es.x_mass, es.p_mass) == (0.0, 0.0) and not es.triggered
    # 32 cells is the smallest meaningful axis: centered stays clear...
    xv = np.linspace(-6, 6, 32, endpoint=False)
    dx = xv[1] - xv[0]
    centered = _gauss(xv, 0.0, 0.7)
    assert not boundary.edge_report(centered, centered, dx, dx).triggered
    # ...and a genuine edge state still trips it
    assert boundary.edge_report(_gauss(xv, 5.5, 0.7), centered,
                                dx, dx).axes == ["x"]


def test_edge_report_unit():
    xv = np.linspace(-6, 6, 256, endpoint=False)
    pv = np.linspace(-7, 7, 256, endpoint=False)
    dx, dp = xv[1] - xv[0], pv[1] - pv[0]
    rho_c, phi_c = _gauss(xv, 0.0, 0.7), _gauss(pv, 0.0, 0.7)
    assert not boundary.edge_report(rho_c, phi_c, dx, dp).triggered
    es = boundary.edge_report(_gauss(xv, 5.5, 0.7), phi_c, dx, dp)
    assert es.axes == ["x"] and es.x_mass > boundary.EDGE_THRESHOLD
    # a diverged run (non-finite marginals) must never trigger
    rho_n = rho_c.copy()
    rho_n[0] = np.nan
    assert not boundary.edge_report(rho_n, phi_c, dx, dp).triggered


def test_boundary_event_and_status():
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            saw = None
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "boundary":
                        saw = d
                        break
            assert saw is not None, "boundary event never arrived"
            assert saw["axes"] == ["x"] and saw["action"] == "warn"
            assert saw["x_mass"] > boundary.EDGE_THRESHOLD
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["grid"] == {"x1": -6.0, "x2": 6.0, "Nx": 64,
                                 "p1": -7.0, "p2": 7.0, "Np": 64}
            assert r["auto_expand"] is False
            assert r["max_grid"] >= 64
            assert r["boundary"]["axes"] == ["x"]
        client.delete("/api/sessions/%s" % sid)


def test_auto_expand_live_toggle():
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "params_applied":
                        assert d["applied"]["auto_expand"] is True
                        break
            else:
                raise AssertionError("params_applied never arrived")
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["auto_expand"] is True
        client.delete("/api/sessions/%s" % sid)


def test_wigner_preview_edge_warning():
    client = TestClient(app)
    r = client.post("/api/preview/wigner",
                    json={"grid": GRID, **EDGE_IC})
    assert r.status_code == 200
    warns = unquote(r.headers["X-Wignerf-Warnings"])
    assert "edge band" in warns          # the measure-based total-W check
    # a centered state stays clean
    r = client.post("/api/preview/wigner", json={
        "grid": GRID, "type": "mixture",
        "components": [{"x0": 0.0, "p0": 0.0, "sigma_x": 0.5,
                        "sigma_p": 0.707}]})
    assert "edge band" not in unquote(r.headers["X-Wignerf-Warnings"])
