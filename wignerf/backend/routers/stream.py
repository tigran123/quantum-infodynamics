"""
WebSocket streamer: binary frame bundles out, JSON control in.

Backpressure by design: the sender never queues binary frames. In the live
region (cursor at the lockstep frontier) it always sends the NEWEST complete
record — a slow client skips frames, never buffers them. In the replay
region (cursor behind the frontier) records are sent in exact sequence and
playback slips in wall time instead: `delay` (seconds injected between
played-back frames) paces the display, and its default 0 simply means "as
fast as this client renders". A playback-only run never skips a record —
it must not coalesce to the frontier while sequential records remain
unsent, and it auto-pauses only once the frontier record was actually
delivered. Seek sends the exact requested record, paused or not.
"""

import asyncio
import json
import logging
import time
from contextlib import suppress
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
        s.post_msg(s.status())      # echo the flip ahead of any frame burst
    elif msg.type == "pause":
        s.clock.set_running(False)
        s.post_msg(s.status())
    elif msg.type == "delay":
        s.clock.set_delay(msg.seconds)
    elif msg.type == "seek":
        # move the cursor NOW, not on the next sender tick: a play arriving
        # right behind the seek must classify playback-vs-solve against the
        # seeked position, never the stale cursor
        first, last = s.history.extent()
        if last >= 0:
            k = min(max(msg.record, first), last)
            s.clock.set_cursor(k, s.history.latest_complete())
            s.pending_seek = k
            s.frame_evt.set()
    elif msg.type == "ping":
        s.post_msg({"type": "pong"})
    elif msg.type == "set_params":
        cp = None
        if msg.params.U is not None or msg.params.hbar_eff is not None:
            # validate against the LIVE window (auto-expand may have moved
            # it; unions in the pre-regrid window while a plan is pending).
            # hbar-only changes are validated too: a larger hbar widens the
            # Bopp range, and letting an invalid one through would surface
            # as a fatal non-finite check when a pending regrid applies
            # (worker rollback cannot help there — lockstep geometry must
            # stay uniform).
            try:
                hbar = msg.params.hbar_eff or s.cfg.hbar_eff
                expr = msg.params.U if msg.params.U is not None \
                    else s.cfg.potential
                probe = await compile_for(s.validation_grid(), expr,
                                          hbar, s.cfg.variants)
                if msg.params.U is not None:
                    cp = probe
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
    t, geom, variants = rec
    flags = 0 if live else protocol.FLAG_REPLAY
    if s.cfg.mode == "runahead" and live:
        flags |= protocol.FLAG_LIVE_PREVIEW
    # geometry comes from the RECORD, never the session's current grid —
    # replay across a regrid boundary must decode with the old geometry
    return protocol.pack_frame(k, t, geom, variants, flags=flags)


async def _sender(ws, s, recv_task):
    last_sent = -1
    last_wall = monotonic()
    last_status = 0.0
    last_running = s.clock.running
    await ws.send_text(json.dumps(s.status()))
    while not recv_task.done():
        now = monotonic()
        lc = s.history.latest_complete()
        cursor = s.clock.advance_cursor(now - last_wall, lc, last_sent)
        last_wall = now

        # Control channel FIRST: play/pause echoes and periodic status must
        # never queue behind a burst of binary frame sends — the transport
        # button's state depends on them arriving promptly.
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

        k = None
        live = True
        seek = getattr(s, "pending_seek", None)
        if seek is not None:
            s.pending_seek = None
            k = seek                    # already clamped by the handler
            live = k >= lc
            last_sent = -1              # force resend even of the same index
        elif lc >= 0:
            target = int(cursor)
            # A playback-only run must deliver EVERY record: while
            # sequential records remain unsent, stay in the replay branch
            # even when a send blocked long enough for the wall clock to
            # lump the cursor past the frontier — coalescing over that gap
            # is what used to teleport playback straight to the end.
            gap = s.clock.stop_at_frontier and last_sent < lc
            if target >= lc and not gap:
                k = lc                   # live: coalesce to newest
                # (runahead keeps the cursor pinned here until the user
                # seeks, so the newest frame previews while computing;
                # after a seek both modes replay from history identically)
            else:
                # Replay: exact sequential records from history, paced by
                # the cursor. Batch the sends (the loop ticks at ~20 Hz;
                # one record per tick would cap replay at 20 records/s) —
                # but under a WALL-CLOCK budget with preemption: to a slow
                # client each send can block for seconds on backpressure,
                # and an unbounded batch would starve the control channel
                # and keep streaming frames long after a pause arrived.
                # If the client can't keep up, pull the cursor back so
                # playback slips in wall time rather than skipping records.
                first, _ = s.history.extent()
                nxt = max(last_sent + 1, first)
                t0 = monotonic()
                while nxt <= min(target, lc):
                    payload = _pack_record(s, nxt, live=False)
                    if payload is None:
                        break
                    await ws.send_bytes(payload)
                    last_sent = nxt
                    nxt += 1
                    if not s.clock.running or s.pending_seek is not None \
                       or monotonic() - t0 > 0.2:
                        break
                if s.pending_seek is None and last_sent < target:
                    s.clock.set_cursor(last_sent, lc)

        if k is not None and k != last_sent:
            payload = _pack_record(s, k, live)
            if payload is not None:
                await ws.send_bytes(payload)
                last_sent = k

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
    # claim the session BEFORE the first await — two near-simultaneous
    # connects must not both pass the check above
    s.ws_attached = True
    s.pending_seek = None
    recv_task = None
    try:
        await ws.accept()
        recv_task = asyncio.create_task(_receiver(ws, s))
        await _sender(ws, s, recv_task)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("streamer for session %s failed", s.id)
    finally:
        if recv_task is not None:
            recv_task.cancel()
            # retrieve its result: a client disconnect ends the receiver
            # with WebSocketDisconnect, which would otherwise be logged as
            # "Task exception was never retrieved" at GC time
            with suppress(Exception, asyncio.CancelledError):
                await recv_task
        s.ws_attached = False
        s.clock.set_running(False)     # pause on disconnect; TTL takes over
        s.last_seen = time.monotonic()
