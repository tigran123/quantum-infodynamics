"""
Preview endpoints:
- POST /api/preview/potential — compile a U(x) expression, report per-family
  validity + warnings and return plot samples (debounced client-side).
- POST /api/preview/wigner — build the initial Wigner function (mixture or
  cat) and return it as the SAME binary frame bundle the WebSocket streams,
  so the frontend reuses its decoder and WebGL panel for IC editing.
  Diagnostics travel in X-Wignerf-* headers.

Previews always run on the CPU backend: they are cheap, deterministic, and
keep the GPU free for running sessions.
"""

from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from core import initial, observables, protocol
from core.potential import PotentialError, compile_potential, sample_potential
from core.protocol import GridSpec, ICSpec
from core.quantize import quantize
from core.xp import ArrayBackend

router = APIRouter()

_cpu_backend = None


def _backend():
    global _cpu_backend
    if _cpu_backend is None:
        _cpu_backend = ArrayBackend(device="cpu")
    return _cpu_backend


class PotentialPreviewIn(BaseModel):
    expr: str
    x1: float
    x2: float
    n: int = Field(default=400, ge=16, le=4096)
    grid: Optional[GridSpec] = None   # enables the extended-range quantum probe
    hbar_eff: float = Field(default=1.0, gt=0)


@router.post("/preview/potential")
def preview_potential(req: PotentialPreviewIn):
    x_ext = None
    x_range = (req.x1, req.x2)
    if req.grid is not None:
        x_ext = req.grid.x_extended(req.hbar_eff)
        x_range = (req.grid.x1, req.grid.x2)
    try:
        cp = compile_potential(req.expr, x_range=x_range, x_extended=x_ext)
    except PotentialError as e:
        return {"ok": False, "error": str(e)}
    xs, us = sample_potential(cp, req.x1, req.x2, req.n)
    return {
        "ok": True,
        "validity": {"quantum": cp.quantum_valid, "classical": cp.classical_valid},
        "reasons": cp.reasons,
        "warnings": cp.warnings,
        "latex": cp.latex,
        "dudx_latex": cp.dUdx_latex,
        "samples": {"x": xs, "U": us},
        "extended_range": list(x_ext) if x_ext else None,
    }


class WignerPreviewIn(ICSpec):
    grid: GridSpec
    hbar_eff: float = Field(default=1.0, gt=0)


@router.post("/preview/wigner")
def preview_wigner(req: WignerPreviewIn):
    b = _backend()
    try:
        g, W, warns = initial.from_spec(req.grid, req, req.hbar_eff, b)
    except ValueError as e:
        raise HTTPException(422, str(e))
    deficit = abs(1.0 - float(W.sum())*g.dx*g.dp)
    Ws = g.shift2d(W)
    wq, wmin, wmax = quantize(Ws, b)
    obs = observables.compute_basic(Ws, g, req.hbar_eff)
    vf = protocol.VariantFrame(
        vid=0, wq=wq, wmin=wmin, wmax=wmax, E=0.0,
        x_mean=obs.x_mean, x_std=obs.x_std, p_mean=obs.p_mean, p_std=obs.p_std,
        purity=obs.purity, dt=0.0, rho=obs.rho, phi=obs.phi)
    payload = protocol.pack_frame(0, 0.0, g.geom(), [vf],
                                  flags=protocol.FLAG_LIVE_PREVIEW)
    # HTTP headers are latin-1 only: percent-encode so warnings can carry
    # Unicode (sigma, hbar, rho...); the client decodeURIComponent()s it.
    return Response(content=payload, media_type="application/octet-stream",
                    headers={"X-Wignerf-Norm-Deficit": "%.3e" % deficit,
                             "X-Wignerf-Warnings": quote(" | ".join(warns))})
