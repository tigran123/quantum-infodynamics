"""Health and device-info endpoints."""

from functools import lru_cache

from fastapi import APIRouter

import config
from core.xp import ArrayBackend, resolve_devices

router = APIRouter()


@lru_cache(maxsize=1)
def _probe_backend():
    """Probe the whole device pool; top-level device/is_gpu/fft_provider
    describe the fastest (first) device for backward compatibility."""
    try:
        infos = []
        for d in resolve_devices(config.DEVICE):
            b = ArrayBackend(device=d)
            infos.append({"device": b.name, "is_gpu": b.is_gpu,
                          "fft_provider": b.fft_provider})
        return {**infos[0], "devices": infos}
    except Exception as e:
        return {"device": "unavailable", "is_gpu": False,
                "devices": [], "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/device")
def device():
    return _probe_backend()
