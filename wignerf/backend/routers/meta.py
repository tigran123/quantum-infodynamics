"""Health and device-info endpoints."""

from functools import lru_cache

from fastapi import APIRouter

import config
from core.xp import ArrayBackend

router = APIRouter()


@lru_cache(maxsize=1)
def _probe_backend():
    try:
        b = ArrayBackend(device=config.DEVICE)
        return {"device": b.name, "is_gpu": b.is_gpu,
                "fft_provider": b.fft_provider}
    except Exception as e:
        return {"device": "unavailable", "is_gpu": False, "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/device")
def device():
    return _probe_backend()
