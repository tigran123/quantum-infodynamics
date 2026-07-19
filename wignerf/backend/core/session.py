"""
SimSession: 1-4 SolverWorkers + SessionClock (the shared record-time grid
and pacing policy) + FrameHistory + the handoff to the asyncio streamer.

Pacing (see plan):
- interactive: workers compute only while their frontier leads the display
  cursor by at most LEAD records; the streamer advances the cursor at
  `rate` a.u. per wall second.
- runahead: workers compute flat out until t passes t2; the streamer shows
  the newest lockstep-complete record as a live preview.

Sessions are registered in SESSIONS; a WS detach pauses the session and
starts the idle TTL. ttl_sweeper() (spawned from main's lifespan) closes
idle sessions.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import deque

from .history import FrameHistory
from .worker import SolverWorker

log = logging.getLogger(__name__)

LEAD = 3
WS_IDLE_TTL = 120.0

SESSIONS = {}
_LOCK = threading.Lock()


class SessionClock:
    def __init__(self, t1, record_dt, mode, rate, t2):
        self._cond = threading.Condition()
        self.t1 = float(t1)
        self.record_dt = float(record_dt)
        self.mode = mode
        self.rate = float(rate)
        self.t2 = t2
        self.sign = 1
        self.running = False
        self.cursor = 0.0            # display position in record units
        self.browsed = False         # runahead: user touched the timeline
        self._t = [self.t1]          # t of record k (append-only)
        self._anchor = (0, self.t1)  # (k, t) of the last sign change

    def _t_of(self, k):
        # Multiply from the anchor rather than accumulating record_dt, so
        # t_k = t1 + k*record_dt holds exactly within a direction segment.
        ak, at = self._anchor
        while len(self._t) <= k:
            n = len(self._t)
            self._t.append(at + (n - ak)*self.sign*self.record_dt)
        return self._t[k]

    def t_of(self, k):
        with self._cond:
            return self._t_of(k)

    def next_target(self, frontier):
        """Next (k, t_k) for a worker whose newest record is `frontier`,
        or None when it should idle (paused / lead-capped / done)."""
        with self._cond:
            if not self.running:
                return None
            k = frontier + 1
            if self.mode == "interactive" and k - self.cursor > LEAD:
                return None
            if self.mode == "runahead" and self.t2 is not None:
                t_prev = self._t_of(frontier)
                if (self.sign > 0 and t_prev >= self.t2 - 1e-12) or \
                   (self.sign < 0 and t_prev <= self.t2 + 1e-12):
                    return None
            return k, self._t_of(k)

    def wait_work(self, timeout=0.1):
        with self._cond:
            self._cond.wait(timeout)

    def kick(self):
        with self._cond:
            self._cond.notify_all()

    def set_running(self, v):
        with self._cond:
            self.running = bool(v)
            self._cond.notify_all()

    def set_rate(self, v):
        with self._cond:
            self.rate = float(v)

    def set_sign(self, s):
        with self._cond:
            self.sign = 1 if s > 0 else -1
            self._anchor = (len(self._t) - 1, self._t[-1])
            self._cond.notify_all()

    def advance_cursor(self, elapsed, latest_complete):
        """Called by the streamer; returns the updated cursor. The cursor
        may lead the frontier by up to LEAD so workers keep computing.
        In runahead mode the cursor stays pinned to the frontier (newest-
        frame preview while the timeline fills) until the user seeks."""
        with self._cond:
            if self.running:
                self.cursor = min(self.cursor + self.rate*elapsed/self.record_dt,
                                  latest_complete + LEAD)
                if self.mode == "runahead" and not self.browsed:
                    self.cursor = float(max(self.cursor, latest_complete))
                self._cond.notify_all()
            return self.cursor

    def set_cursor(self, k):
        with self._cond:
            self.cursor = float(k)
            self.browsed = True
            self._cond.notify_all()


class SimSession:
    def __init__(self, cfg, compiled_potential, loop, device, fft_threads,
                 history_bytes):
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.compiled_potential = compiled_potential
        self.loop = loop
        self.device = device
        self.fft_threads = fft_threads
        self.history = FrameHistory(len(cfg.variants), history_bytes)
        self.clock = SessionClock(cfg.t1, cfg.record_dt, cfg.mode, cfg.rate, cfg.t2)
        self.frame_evt = asyncio.Event()
        self.msgs = deque(maxlen=64)     # server->client JSON side channel
        self.ws_attached = False
        self.last_seen = time.monotonic()
        self.closed = False
        # Sessions ALWAYS start paused (both modes): computation begins only
        # on the user's explicit play/solve command. Run-ahead differs from
        # interactive solely in pacing once running (flat-out to t2 vs
        # cursor-gated).
        self.workers = [SolverWorker(self, key, i)
                        for i, key in enumerate(cfg.variants)]

    def start(self):
        for w in self.workers:
            w.start()

    # thread-safe: called from worker threads
    def notify_frame(self):
        try:
            self.loop.call_soon_threadsafe(self.frame_evt.set)
        except RuntimeError:
            pass   # loop already closed during shutdown

    def post_msg(self, d):
        self.msgs.append(d)
        self.notify_frame()

    def post_error(self, message, detail=None):
        log.error("session %s: %s", self.id, message)
        self.post_msg({"type": "error", "message": message, "detail": detail})

    # called from the router/streamer coroutines
    def apply_params(self, change, cp=None):
        if change.dt_sign is not None:
            self.clock.set_sign(change.dt_sign)
        cmd = {"kind": "params", "cp": cp, "mass": change.mass, "c": change.c,
               "hbar_eff": change.hbar_eff, "tol": change.tol}
        if cp is not None or any(v is not None for v in
                                 (change.mass, change.c, change.hbar_eff, change.tol)):
            for w in self.workers:
                w.cmd_q.put(cmd)
        self.clock.kick()
        applied = {k: v for k, v in change.model_dump().items() if v is not None}
        self.post_msg({"type": "params_applied", "applied": applied,
                       "at_record": self.history.latest_complete() + 1})

    def status(self):
        first, last = self.history.extent()
        t0, t1_ = self.history.t_extent()
        return {
            "type": "status",
            "session_id": self.id,
            "running": self.clock.running,
            "mode": self.clock.mode,
            "t2": self.clock.t2,
            "rate": self.clock.rate,
            "sign": self.clock.sign,
            "record_dt": self.clock.record_dt,
            "record_extent": [first, last],
            "t_extent": [t0, t1_],
            "cursor": self.clock.cursor,
            "history_bytes": self.history.nbytes(),
            "per_variant": [{"variant": w.key, "dt": w.dt,
                             "steps_per_sec": round(w.steps_per_sec, 2),
                             "steps_total": w.steps_total}
                            for w in self.workers],
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        for w in self.workers:
            w.stop()
        for w in self.workers:
            w.join(timeout=3.0)
        with _LOCK:
            SESSIONS.pop(self.id, None)


def create_session(cfg, compiled_potential, device, fft_threads, history_bytes):
    loop = asyncio.get_running_loop()
    s = SimSession(cfg, compiled_potential, loop, device, fft_threads,
                   history_bytes)
    with _LOCK:
        SESSIONS[s.id] = s
    s.start()
    return s


def get_session(sid):
    with _LOCK:
        return SESSIONS.get(sid)


def close_all():
    for s in list(SESSIONS.values()):
        s.close()


async def ttl_sweeper():
    while True:
        await asyncio.sleep(15.0)
        now = time.monotonic()
        for s in list(SESSIONS.values()):
            if not s.ws_attached and now - s.last_seen > WS_IDLE_TTL:
                log.info("session %s idle > %.0fs, closing", s.id, WS_IDLE_TTL)
                s.close()
