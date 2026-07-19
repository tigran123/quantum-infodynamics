"""
SimSession: 1-4 SolverWorkers + SessionClock (the shared record-time grid
and pacing policy) + FrameHistory + the handoff to the asyncio streamer.
The device spec (WIGNERF_DEVICE) is resolved to a fastest-first pool and
variant workers are spread over it (assign_devices) — each worker owns its
own ArrayBackend, so multi-GPU needs no propagator changes.

Pacing (see plan):
- Computation ALWAYS runs at full speed — `delay` (seconds injected
  between played-back frames; 0 = default = as fast as the client
  renders) paces only the display, never the workers. Delay 0 is safe:
  replay slips on WS backpressure instead of skipping frames.
- While solving with the display attached to the frontier, the cursor just
  follows the newest lockstep-complete record (coalescing). Detached from
  the frontier (user seeked back, or a playback-only run), the display
  replays history one frame per `delay` seconds.
- interactive computes until paused; runahead stops when t passes t2.

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
from .protocol import VARIANTS
from .worker import SolverWorker
from .xp import resolve_devices

log = logging.getLogger(__name__)

WS_IDLE_TTL = 120.0

SESSIONS = {}
_LOCK = threading.Lock()


def assign_devices(variant_keys, devices):
    """Map each variant to a device from the pool (both in given order;
    devices are fastest first). Lockstep gates on the slowest worker, so
    the costliest variants — relativistic need the most substeps per
    record, quantum breaks the tie — go to the fastest device, which also
    takes the larger share when the split is uneven. Returns {key: device}."""
    cost = {k: (VARIANTS[k]["relativistic"], VARIANTS[k]["quantum"])
            for k in variant_keys}
    ranked = sorted(variant_keys, key=lambda k: cost[k], reverse=True)
    ndev = min(len(devices), len(ranked))
    base, extra = divmod(len(ranked), ndev)
    out, i = {}, 0
    for j in range(ndev):
        n = base + (1 if j < extra else 0)
        for k in ranked[i:i + n]:
            out[k] = devices[j]
        i += n
    return out


class SessionClock:
    def __init__(self, t1, record_dt, mode, delay, t2):
        self._cond = threading.Condition()
        self.t1 = float(t1)
        self.record_dt = float(record_dt)
        self.mode = mode
        self.delay = float(delay)    # seconds between played-back frames; 0 = max
        self.t2 = t2
        self.sign = 1
        self.running = False
        self.cursor = 0.0            # display position in record units
        self.stop_at_frontier = False  # playback-only run: pause at frontier
        self.browsed = False         # display detached from the live frontier
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

    def _runahead_done(self, frontier):
        if self.t2 is None:
            return False
        t_prev = self._t_of(frontier)
        return (self.sign > 0 and t_prev >= self.t2 - 1e-12) or \
               (self.sign < 0 and t_prev <= self.t2 + 1e-12)

    def next_target(self, frontier):
        """Next (k, t_k) for a worker whose newest record is `frontier`,
        or None when it should idle (paused / playback-only / done).
        Computation always proceeds at full speed — the display cursor
        never gates the workers (`delay` paces only what the client sees)."""
        with self._cond:
            if not self.running or self.stop_at_frontier:
                return None
            if self.mode == "runahead" and self._runahead_done(frontier):
                return None
            k = frontier + 1
            return k, self._t_of(k)

    def wait_work(self, timeout=0.1):
        with self._cond:
            self._cond.wait(timeout)

    def kick(self):
        with self._cond:
            self._cond.notify_all()

    def set_running(self, v, latest_complete=None):
        """`latest_complete` (the frontier at play time) decides whether this
        run is playback-only: pressing play behind the frontier (BOTH modes)
        or after a finished run-ahead must replay history and PAUSE at the
        end — computing new records is always an explicit request made AT
        the frontier (the transport button shows "Solve" exactly then;
        interactive then computes until paused, runahead until t2)."""
        with self._cond:
            self.running = bool(v)
            if not self.running:
                self.stop_at_frontier = False
            elif latest_complete is not None:
                behind = self.cursor < latest_complete
                self.stop_at_frontier = behind or \
                    (self.mode == "runahead" and
                     self._runahead_done(latest_complete))
                # Solve pressed AT the frontier re-attaches the display to
                # it (live coalescing); play behind starts detached (replay)
                self.browsed = behind
            self._cond.notify_all()

    def set_delay(self, v):
        with self._cond:
            self.delay = float(v)

    def set_sign(self, s):
        with self._cond:
            self.sign = 1 if s > 0 else -1
            self._anchor = (len(self._t) - 1, self._t[-1])
            self._cond.notify_all()

    def advance_cursor(self, elapsed, latest_complete, delivered):
        """Called by the streamer; returns the updated cursor. `delay`
        (seconds between played-back frames; 0 = as fast as the client
        renders) paces the DISPLAY only. Attached to the frontier while
        solving, the cursor just follows the newest complete record
        (coalescing preview — both modes). Detached (`browsed`: the user
        seeked back, or a playback-only run), it replays history one frame
        per `delay` seconds. A playback-only run (stop_at_frontier)
        auto-pauses at the frontier instead of rolling into computation —
        but only once the streamer has actually DELIVERED the frontier
        record (`delivered` = newest record index sent): `elapsed` includes
        time spent blocked in a send to a slow client, so the wall clock
        alone can lump the cursor past records nobody has seen yet."""
        with self._cond:
            if latest_complete >= 0:
                if self.running and (self.stop_at_frontier or self.browsed):
                    step = elapsed/self.delay if self.delay > 0 \
                        else float("inf")
                    self.cursor = min(self.cursor + step,
                                      float(latest_complete))
                    if self.stop_at_frontier and delivered >= latest_complete \
                       and self.cursor >= latest_complete:
                        self.running = False
                        self.stop_at_frontier = False
                        self.browsed = False   # arrived at the frontier
                elif not self.browsed:
                    # Attached to the frontier: follow it while solving AND
                    # while PAUSED — a pause leaves in-flight records landing
                    # after it, and the display must settle on the final
                    # frontier or the transport would offer "Play" over a
                    # few phantom records the user never rewound to.
                    self.cursor = float(max(self.cursor, latest_complete))
                self._cond.notify_all()
            return self.cursor

    def set_cursor(self, k, latest_complete):
        """Seek/slip the display. Seeking to the frontier re-attaches the
        cursor to it (live coalescing); anywhere behind detaches it."""
        with self._cond:
            self.cursor = float(k)
            self.browsed = k < latest_complete
            self._cond.notify_all()


class SimSession:
    def __init__(self, cfg, compiled_potential, loop, device, fft_threads,
                 history_bytes):
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.compiled_potential = compiled_potential
        self.loop = loop
        self.devices = resolve_devices(device)
        self.fft_threads = fft_threads
        self.history = FrameHistory(len(cfg.variants), history_bytes)
        self.clock = SessionClock(cfg.t1, cfg.record_dt, cfg.mode, cfg.delay, cfg.t2)
        self.frame_evt = asyncio.Event()
        self.msgs = deque(maxlen=64)     # server->client JSON side channel
        self.ws_attached = False
        self.last_seen = time.monotonic()
        self.closed = False
        # Sessions ALWAYS start paused (both modes): computation begins only
        # on the user's explicit play/solve command. Run-ahead differs from
        # interactive solely in pacing once running (flat-out to t2 vs
        # cursor-gated).
        assignment = assign_devices(cfg.variants, self.devices)
        self.workers = [SolverWorker(self, key, i, assignment[key])
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
            "delay": self.clock.delay,
            "sign": self.clock.sign,
            "record_dt": self.clock.record_dt,
            "record_extent": [first, last],
            "t_extent": [t0, t1_],
            "cursor": self.clock.cursor,
            "history_bytes": self.history.nbytes(),
            "devices": self.devices,
            "per_variant": [{"variant": w.key, "dt": w.dt,
                             "device": w.device,
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
