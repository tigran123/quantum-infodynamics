"""
WebSocket streamer: binary frame bundles out, JSON control in.

Backpressure by design: the sender never queues binary frames. In the live
region (cursor at the lockstep frontier) it always sends the NEWEST complete
record — a slow client skips frames, never buffers them. In the replay
region (cursor behind the frontier) records are sent in exact sequence and
playback slips in wall time instead. Seek sends the exact requested record,
paused or not.
"""

import asyncio
import json
import logging
import time
from time import monotonic

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from core import protocol
from core import session as sessions
from routers.sessions import compile_for

log = logging.getLogger(__name__)

router = APIRouter()

_client_msg = TypeAdapter(protocol.ClientMsg)

STATUS_PERIOD = 1.0


async def _handle(msg, s, ws):
    if msg.type == "play":
        # the frontier at play time decides playback-only vs solving
        s.clock.set_running(True, s.history.latest_complete())
    elif msg.type == "pause":
        s.clock.set_running(False)
    elif msg.type == "rate":
        s.clock.set_rate(msg.au_per_second)
    elif msg.type == "seek":
        s.pending_seek = msg.record
        s.frame_evt.set()
    elif msg.type == "ping":
        s.post_msg({"type": "pong"})
    elif msg.type == "set_params":
        cp = None
        if msg.params.U is not None:
            try:
                hbar = msg.params.hbar_eff or s.cfg.hbar_eff
                cp = await compile_for(s.cfg.grid, msg.params.U, hbar,
                                       s.cfg.variants)
            except Exception as e:
                detail = getattr(e, "detail", str(e))
                s.post_msg({"type": "error", "code": "bad_potential",
                            "message": str(detail)})
                return
        s.apply_params(msg.params, cp)


async def _receiver(ws, s):
    while True:
        text = await ws.receive_text()
        try:
            msg = _client_msg.validate_json(text)
        except ValidationError as e:
            s.post_msg({"type": "error", "code": "bad_message",
                        "message": e.errors()[0].get("msg", "invalid message")})
            continue
        await _handle(msg, s, ws)


def _pack_record(s, k, live):
    rec = s.history.get(k)
    if rec is None:
        return None
    t, variants = rec
    flags = 0 if live else protocol.FLAG_REPLAY
    if s.cfg.mode == "runahead" and live:
        flags |= protocol.FLAG_LIVE_PREVIEW
    return protocol.pack_frame(k, t, s.cfg.grid.Nx, s.cfg.grid.Np,
                               variants, flags=flags)


async def _sender(ws, s, recv_task):
    last_sent = -1
    last_wall = monotonic()
    last_status = 0.0
    last_running = s.clock.running
    await ws.send_text(json.dumps(s.status()))
    while not recv_task.done():
        now = monotonic()
        lc = s.history.latest_complete()
        cursor = s.clock.advance_cursor(now - last_wall, lc)
        last_wall = now

        k = None
        live = True
        seek = getattr(s, "pending_seek", None)
        if seek is not None:
            s.pending_seek = None
            first, last = s.history.extent()
            if last >= 0:
                k = min(max(seek, first), last)
                s.clock.set_cursor(k)
                live = k >= lc
                last_sent = -1          # force resend even of the same index
        elif lc >= 0:
            target = int(cursor)
            if target >= lc:
                k = lc                   # live: coalesce to newest
                # (runahead keeps the cursor pinned here until the user
                # seeks, so the newest frame previews while computing;
                # after a seek both modes replay from history identically)
            else:
                # Replay: exact sequential records from history, paced by
                # the cursor. Batch the sends — the sender loop runs at
                # ~20 Hz, so one-record-per-iteration would cap replay at
                # 20 records/s and any faster rate would overtake the
                # frontier and needlessly resume computation. If the
                # client can't keep up, pull the cursor back so playback
                # slips in wall time rather than skipping records.
                first, _ = s.history.extent()
                nxt = max(last_sent + 1, first)
                sent = 0
                while nxt <= min(target, lc) and sent < 64:
                    payload = _pack_record(s, nxt, live=False)
                    if payload is None:
                        break
                    await ws.send_bytes(payload)
                    last_sent = nxt
                    nxt += 1
                    sent += 1
                if last_sent < target:
                    s.clock.set_cursor(last_sent)

        if k is not None and k != last_sent:
            payload = _pack_record(s, k, live)
            if payload is not None:
                await ws.send_bytes(payload)
                last_sent = k

        while s.msgs:
            await ws.send_text(json.dumps(s.msgs.popleft()))
        if s.history.take_evicted_flag():
            first, last = s.history.extent()
            await ws.send_text(json.dumps({"type": "eviction",
                                           "new_extent": [first, last]}))
        # push status immediately on a running-state flip (auto-pause at the
        # frontier, play/pause echo) — the 1 s cadence covers the rest
        if s.clock.running != last_running or now - last_status > STATUS_PERIOD:
            last_running = s.clock.running
            last_status = now
            await ws.send_text(json.dumps(s.status()))

        s.frame_evt.clear()
        try:
            await asyncio.wait_for(s.frame_evt.wait(),
                                   timeout=0.05 if s.clock.running else 0.2)
        except asyncio.TimeoutError:
            pass


@router.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    s = sessions.get_session(sid)
    if s is None:
        await ws.close(code=4404)
        return
    if s.ws_attached:
        await ws.accept()
        await ws.close(code=4409)
        return
    await ws.accept()
    s.ws_attached = True
    s.pending_seek = None
    recv_task = asyncio.create_task(_receiver(ws, s))
    try:
        await _sender(ws, s, recv_task)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("streamer for session %s failed", s.id)
    finally:
        recv_task.cancel()
        s.ws_attached = False
        s.clock.set_running(False)     # pause on disconnect; TTL takes over
        s.last_seen = time.monotonic()
