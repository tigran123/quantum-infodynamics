"""
mp4 export of a computed record range: create a job, watch it (REST poll or
the session's WebSocket 'export' events), download the file, delete it.

Export is PAUSED-only: a running session evicts old records once the history
cap is reached, and it is also the interaction the feature is for — you film
a range you have already played back and judged interesting.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import config
from core import session as sessions
from core import videoexport
from core.protocol import ExportSpec

router = APIRouter()


@router.post("/sessions/{sid}/export", status_code=202)
def create_export(sid: str, spec: ExportSpec):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    if videoexport.ffmpeg_path() is None:
        raise HTTPException(503, "ffmpeg is not installed on the server")
    if s.clock.running:
        raise HTTPException(409, "pause the session before exporting "
                                 "(export renders already-computed records)")
    if videoexport.active_for(sid) is not None:
        raise HTTPException(409, "an export is already running for this session")
    first, last = s.history.extent()
    if last < 0:
        raise HTTPException(422, "no computed records to export")
    k0 = first if spec.k0 is None else max(spec.k0, first)
    k1 = last if spec.k1 is None else min(spec.k1, last)
    if k1 < k0:
        raise HTTPException(422, "empty record range after clamping to the "
                                 "retained history [%d, %d]" % (first, last))
    unknown = set(spec.variants or ()) - set(s.cfg.variants)
    if unknown:
        raise HTTPException(422, "variants not in this session: %s"
                            % ", ".join(sorted(unknown)))
    job = videoexport.start(s, spec, k0, k1, config.EXPORT_DIR)
    return job.status()


@router.get("/exports/{jid}")
def export_status(jid: str):
    job = videoexport.get(jid)
    if job is None:
        raise HTTPException(404, "no such export job")
    return job.status()


@router.get("/exports/{jid}/file")
def export_file(jid: str):
    job = videoexport.get(jid)
    if job is None:
        raise HTTPException(404, "no such export job")
    if job.state != "done" or not os.path.exists(job.path):
        raise HTTPException(409, "export is %s" % job.state)
    return FileResponse(job.path, media_type="video/mp4",
                        filename=job.download_name)


@router.delete("/exports/{jid}")
def export_delete(jid: str):
    if videoexport.drop(jid) is None:
        raise HTTPException(404, "no such export job")
    return {"ok": True}
