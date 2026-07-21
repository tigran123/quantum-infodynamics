"""REST API tests for the preview/meta endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import protocol
from core.quantize import dequantize
from main import app

client = TestClient(app)

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_device():
    r = client.get("/api/device")
    assert r.status_code == 200
    assert "device" in r.json()


def test_potential_preview_valid():
    r = client.post("/api/preview/potential",
                    json={"expr": "x^2/2", "x1": -6, "x2": 6, "grid": GRID})
    d = r.json()
    assert d["ok"] and d["validity"]["quantum"] and d["validity"]["classical"]
    assert len(d["samples"]["x"]) == 400
    assert d["extended_range"][0] < -6


def test_potential_preview_heaviside():
    r = client.post("/api/preview/potential",
                    json={"expr": "Heaviside(x)", "x1": -6, "x2": 6, "grid": GRID})
    d = r.json()
    assert d["ok"] and d["validity"]["quantum"] and not d["validity"]["classical"]


def test_potential_preview_rejected():
    r = client.post("/api/preview/potential",
                    json={"expr": "__import__('os')", "x1": -6, "x2": 6})
    d = r.json()
    assert not d["ok"] and "error" in d


def _decode(resp):
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")
    return protocol.unpack_frame(resp.content)


def test_wigner_preview_mixture_roundtrip():
    r = client.post("/api/preview/wigner", json={
        "type": "mixture", "grid": GRID,
        "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707, "sigma_p": 0.707}],
    })
    f = _decode(r)
    assert (f.geom.Nx, f.geom.Np) == (64, 64)
    assert (f.geom.x1, f.geom.x2, f.geom.p1, f.geom.p2) == (-6.0, 6.0, -7.0, 7.0)
    assert f.flags & protocol.FLAG_LIVE_PREVIEW
    v = f.variants[0]
    W = dequantize(v.wq, v.wmin, v.wmax)
    # dequantized norm: 16-bit quantization of a 64x64 frame keeps the grid
    # integral within ~1e-3
    assert abs(W.sum()*(12./64)*(14./64) - 1.0) < 1e-3
    assert v.x_mean == pytest.approx(2.0, abs=1e-3)
    assert float(r.headers["X-Wignerf-Norm-Deficit"]) < 1e-4
    # marginals are float32 natural-order arrays of the right size
    assert v.rho.shape == (64,) and v.phi.shape == (64,)
    assert np.argmax(v.rho) == np.abs(np.linspace(-6, 6, 64, endpoint=False) - 2.0).argmin()


def test_wigner_preview_cat_has_negativity():
    r = client.post("/api/preview/wigner", json={
        "type": "cat", "grid": GRID, "hbar_eff": 1.0,
        "components": [{"x0": -2.0, "p0": 0.0, "sigma_x": 0.5},
                       {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5}],
    })
    assert _decode(r).variants[0].wmin < 0


def test_wigner_preview_mixture_requires_sigma_p():
    r = client.post("/api/preview/wigner", json={
        "type": "mixture", "grid": GRID,
        "components": [{"x0": 0.0, "p0": 0.0, "sigma_x": 0.5}],
    })
    assert r.status_code == 422
