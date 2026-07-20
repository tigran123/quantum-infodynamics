"""
Session lifecycle: create (validates + compiles the potential, spawns the
solver workers), inspect, delete, and the scalar time-series backfill.
"""

import os
from functools import partial

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request

import config
from core import session as sessions
from core.potential import PotentialError, compile_potential
from core.protocol import VARIANTS, SessionCreate

router = APIRouter()


def _fft_threads(n_variants):
    if config.FFT_THREADS > 0:
        return config.FFT_THREADS
    return max(1, min(4, (os.cpu_count() or 4)//(2*n_variants)))


async def compile_for(cfg_grid, expr, hbar_eff, variants):
    """Compile U(x) off the event loop and enforce per-family validity for
    the requested variants. Raises HTTPException(422) on failure."""
    try:
        cp = await to_thread.run_sync(partial(
            compile_potential, expr,
            x_range=(cfg_grid.x1, cfg_grid.x2),
            x_extended=cfg_grid.x_extended(hbar_eff)))
    except PotentialError as e:
        raise HTTPException(422, "potential: %s" % e)
    needs_q = any(VARIANTS[v]["quantum"] for v in variants)
    needs_c = any(not VARIANTS[v]["quantum"] for v in variants)
    if needs_q and not cp.quantum_valid:
        raise HTTPException(422, "potential is not quantum-valid: "
                            + "; ".join(cp.reasons))
    if needs_c and not cp.classical_valid:
        raise HTTPException(422, "potential is not classical-valid: "
                            + "; ".join(cp.reasons))
    return cp


@router.post("/sessions")
async def create_session(cfg: SessionCreate, request: Request):
    if cfg.ic.type == "mixture" and any(c.sigma_p is None for c in cfg.ic.components):
        raise HTTPException(422, "sigma_p is required for mixture components")
    cp = await compile_for(cfg.grid, cfg.potential, cfg.hbar_eff, cfg.variants)
    s = sessions.create_session(
        cfg, cp, device=config.DEVICE,
        fft_threads=_fft_threads(len(cfg.variants)),
        history_bytes=config.HISTORY_MB*1024*1024)
    # Prefix the WS path with the app's root_path so it inherits the nginx
    # prefix (uvicorn --root-path /wignerf, from APP_ROOT_PATH). Empty in dev.
    root_path = request.scope.get("root_path", "").rstrip("/")
    return {"session_id": s.id, "ws_url": "%s/api/ws/%s" % (root_path, s.id),
            "variants": cfg.variants, "record_dt": cfg.record_dt,
            "warnings": cp.warnings}


@router.get("/sessions/{sid}")
def get_session(sid: str):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    return s.status()


@router.delete("/sessions/{sid}")
def delete_session(sid: str):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    s.close()
    return {"ok": True}


@router.get("/sessions/{sid}/series")
def series(sid: str, start: int = 0, end: int = 1 << 62):
    """Per-record scalars (gapless even when live streaming skipped frames)."""
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    first, last = s.history.extent()
    if last < 0:
        return {"records": [], "extent": [first, last]}
    lo, hi = max(start, first), min(end, last)
    out = []
    for k in range(lo, min(hi, lo + 2000) + 1):
        rec = s.history.get(k)
        if rec is None:
            continue
        t, variants = rec
        out.append({"n": k, "t": t,
                    "variants": [{"vid": v.vid, "E": v.E,
                                  "x_mean": v.x_mean, "x_std": v.x_std,
                                  "p_mean": v.p_mean, "p_std": v.p_std,
                                  "purity": v.purity,
                                  "dt": v.dt} for v in variants]})
    return {"records": out, "extent": [first, last]}
