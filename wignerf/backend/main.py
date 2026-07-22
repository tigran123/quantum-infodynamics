"""
wignerf backend — pure wiring (urantia-library style): the FastAPI app,
lifespan, routers and (in production) the static SPA mount. All logic lives
in core/ and routers/.

Run:  uvicorn main:app --port 8010     (or via ../start.sh)
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import session as sessions
from core import videoexport
from routers import export, meta, preview, sessions as sessions_router, stream


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    sweeper = asyncio.create_task(sessions.ttl_sweeper())
    try:
        yield
    finally:
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper
        sessions.close_all()
        videoexport.close_all()      # no orphaned mp4s in the export dir


app = FastAPI(title="wignerf", lifespan=_lifespan)
app.include_router(meta.router, prefix="/api")
app.include_router(preview.router, prefix="/api")
app.include_router(sessions_router.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(stream.router, prefix="/api")

# Serve the built SPA in production; absent in dev, where Vite proxies /api.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True))
