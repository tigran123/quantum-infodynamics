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
from dataclasses import dataclass, replace

from . import boundary, videoexport
from .grid import GridState
from .history import FrameHistory
from .potential import PotentialError, compile_potential
from .protocol import VARIANTS
from .worker import SolverWorker
from .xp import resolve_devices

log = logging.getLogger(__name__)

WS_IDLE_TTL = 120.0

# How many records a fast variant worker may run ahead of the lockstep
# frontier (the newest record ALL variants have landed on). Without a bound
# the fast variants (classical, non-relativistic) outrun the slow ones
# without limit: the incomplete records above the frontier cannot be
# evicted, so the history grows past its byte cap — and the fast workers
# burn device time the slowest worker (the actual rate limiter of the
# stream) needs. The slowest worker is never gated, so no deadlock.
SKEW_MARGIN = 2

SESSIONS = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class RegridPlan:
    """A committed grid change: every record >= k_star is computed on
    `state`. k_star is chosen past ALL in-flight records, so the switch is
    lockstep-uniform; a committed plan is never cancelled (toggling
    auto_expand off only gates future scheduling — a worker may already
    have applied it)."""
    epoch: int
    k_star: int
    state: GridState


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

    def next_target(self, frontier, latest_complete):
        """Next (k, t_k) for a worker whose newest record is `frontier`,
        or None when it should idle (paused / playback-only / done / too far
        ahead of the lockstep frontier `latest_complete` — see SKEW_MARGIN).
        Computation always proceeds at full speed — the display cursor
        never gates the workers (`delay` paces only what the client sees)."""
        with self._cond:
            if not self.running or self.stop_at_frontier:
                return None
            if self.mode == "runahead" and self._runahead_done(frontier):
                return None
            k = frontier + 1
            if k > latest_complete + 1 + SKEW_MARGIN:
                return None
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
                    # A run-ahead that reached t2 is FINISHED: its workers
                    # already idle (next_target returns None), so leaving
                    # `running` set would freeze the transport on "Pause"
                    # forever and lock out every paused-only action (mp4
                    # export). Delivery-aware like the playback stop above:
                    # never end the run over records nobody has seen.
                    if self.running and self.mode == "runahead" \
                       and self._runahead_done(latest_complete) \
                       and delivered >= latest_complete:
                        self.running = False
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
                 history_bytes, max_grid=4096):
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.compiled_potential = compiled_potential
        self.loop = loop
        self.devices = resolve_devices(device)
        self.fft_threads = fft_threads
        self.history = FrameHistory(len(cfg.variants), history_bytes)
        self.clock = SessionClock(cfg.t1, cfg.record_dt, cfg.mode, cfg.delay, cfg.t2)
        # boundary watch: detection is always on (worker.report_edge every
        # record); auto_expand governs only the response and is live-
        # toggleable like tol (session-level policy, no worker involvement)
        self.auto_expand = cfg.auto_expand
        self.max_grid = int(max_grid)
        self._edge_lock = threading.Lock()
        self._edge = {}              # slot -> latest EdgeState
        self._edge_posted = []       # axes signature of the last boundary msg
        self.boundary_state = {"axes": [], "x_mass": 0.0, "p_mass": 0.0}
        # live grid window on the frozen lattice; regrids replace it at
        # commit time (workers switch their propagators at plan.k_star)
        self.grid_state = GridState.from_spec(cfg.grid)
        self._prev_grid_state = None   # pre-regrid window while a plan runs
        self._regrid_plan = None
        self._regrid_epoch = 0
        self._capped_posted = False    # latched warnings (reset on all-clear)
        self._invalid_posted = False
        self.frame_evt = asyncio.Event()
        self.msgs = deque(maxlen=64)     # server->client JSON side channel
        # live parameter changes, in record order: an mp4 export prints the
        # ones inside its range, or its "how to reproduce this" block would
        # be a lie about the frames after the change
        self.param_log = []
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

    # thread-safe: called from worker threads after every record
    def report_edge(self, slot, k, edge):
        """Aggregate per-variant edge-band state; post a 'boundary' event
        only when the set of tripped axes changes (never per-record spam —
        an empty axes list announces the all-clear). With auto_expand on, a
        tripped state also attempts to schedule a regrid (retried every
        record until a plan commits or the state clears)."""
        with self._edge_lock:
            self._edge[slot] = edge
            axes = sorted({a for e in self._edge.values() for a in e.axes})
            x_mass = max(e.x_mass for e in self._edge.values())
            p_mass = max(e.p_mass for e in self._edge.values())
            self.boundary_state = {"axes": axes, "x_mass": x_mass,
                                   "p_mass": p_mass}
            if axes != self._edge_posted:
                self._edge_posted = axes
                self.post_msg({"type": "boundary", "record": k, "axes": axes,
                               "x_mass": x_mass, "p_mass": p_mass,
                               "action": "warn"})
            if not axes:
                self._capped_posted = False
                self._invalid_posted = False
            want = (bool(axes) and self.auto_expand
                    and not self._plan_pending() and not self._invalid_posted)
        if want:
            # off the lock: support scan is O(N), U revalidation ~ms sympy
            self._schedule_regrid(k, axes)

    def _plan_pending(self):
        """Committed but not yet landed by all variants (edge lock held)."""
        p = self._regrid_plan
        return p is not None and self.history.latest_complete() < p.k_star

    # called by workers between records (see worker._run)
    def current_regrid(self):
        with self._edge_lock:
            return self._regrid_plan

    def validation_grid(self):
        """Duck-typed grid (.x1/.x2/.x_extended) for live-U validity checks:
        the live window, unioned with the pre-regrid one while a plan is
        pending (the old window keeps evolving until k_star)."""
        with self._edge_lock:
            gs = self.grid_state
            if self._plan_pending() and self._prev_grid_state is not None:
                return self._prev_grid_state.union(gs)
            return gs

    def _potential_invalid(self, expr, win, hbar_eff):
        """Probe expr's validity for the active variant families on the
        window `win` (its extended Bopp range under `hbar_eff`). Returns a
        reason string, or None when valid."""
        try:
            cp = compile_potential(expr, x_range=(win.x1, win.x2),
                                   x_extended=win.x_extended(hbar_eff))
        except PotentialError as e:
            return str(e)
        needs_q = any(VARIANTS[v]["quantum"] for v in self.cfg.variants)
        needs_c = any(not VARIANTS[v]["quantum"] for v in self.cfg.variants)
        if (needs_q and not cp.quantum_valid) or \
           (needs_c and not cp.classical_valid):
            return "; ".join(cp.reasons)
        return None

    def _schedule_regrid(self, k, axes):
        """Plan and commit an exact fixed-lattice regrid (move and/or double
        of the tripped axes, support centered, capped at max_grid). The
        WHOLE plan/validate/commit runs under the edge lock: plan commits
        and physics commits (apply_params) are mutually exclusive, so a
        plan is always validated against the exact physics the workers will
        hold when applying it. The ~ms sympy probe under the lock happens
        at most once per regrid attempt. Callers must NOT hold the lock."""
        with self._edge_lock:
            if self._plan_pending():
                return
            gs = self.grid_state
            rec = self.history.get(self.history.latest_complete())
            if rec is None:
                return                 # evicted/incomplete: retry next record
            _t, geom, frames = rec
            if geom != gs.geom():
                return                 # regrid landed under us: retry later
            sx = [boundary.support_cells(vf.rho, gs.dx) for vf in frames]
            sp = [boundary.support_cells(vf.phi, gs.dp) for vf in frames]
            lo_x, hi_x = min(a for a, _ in sx), max(b for _, b in sx)
            lo_p, hi_p = min(a for a, _ in sp), max(b for _, b in sp)
            x_plan = boundary.plan_axis(gs.ox, gs.Nx, lo_x, hi_x,
                                        self.max_grid) if "x" in axes else None
            p_plan = boundary.plan_axis(gs.op, gs.Np, lo_p, hi_p,
                                        self.max_grid) if "p" in axes else None
            new, kinds, capped = gs, {}, []
            for ax, pl in (("x", x_plan), ("p", p_plan)):
                if pl is None:
                    continue
                if pl.kind == "capped":
                    capped.append(ax)
                elif ax == "x":
                    new = replace(new, ox=pl.offset, Nx=pl.n)
                    kinds["x"] = pl.kind
                else:
                    new = replace(new, op=pl.offset, Np=pl.n)
                    kinds["p"] = pl.kind
            if capped and not self._capped_posted:
                self._capped_posted = True
                self.post_msg({"type": "boundary", "record": k,
                               "axes": capped, "action": "capped",
                               "max_grid": self.max_grid,
                               "x_mass": self.boundary_state["x_mass"],
                               "p_mass": self.boundary_state["p_mass"]})
            if new == gs:
                return
            if (new.ox, new.Nx) != (gs.ox, gs.Nx):
                # the extended Bopp range moves with the x-window (dp is
                # frozen, so its half-width never changes): revalidate U on
                # the union of old and new windows BEFORE committing
                reason = self._potential_invalid(self.cfg.potential,
                                                 gs.union(new),
                                                 self.cfg.hbar_eff)
                if reason is not None:
                    if not self._invalid_posted:
                        self._invalid_posted = True
                        self.post_msg({"type": "boundary", "record": k,
                                       "axes": axes,
                                       "action": "invalid_potential",
                                       "message": "cannot expand: %s" % reason,
                                       "x_mass": self.boundary_state["x_mass"],
                                       "p_mass": self.boundary_state["p_mass"]})
                    return
            k_star = max(self.history.variant_frontier(s)
                         for s in range(len(self.workers))) + 2
            self._regrid_epoch += 1
            self._prev_grid_state = gs
            self.grid_state = new
            plan = self._regrid_plan = RegridPlan(self._regrid_epoch,
                                                  k_star, new)
            self.post_msg({"type": "regrid", "at_record": plan.k_star,
                           "epoch": plan.epoch, "kind": kinds,
                           "grid": {"x1": new.x1, "x2": new.x2, "Nx": new.Nx,
                                    "p1": new.p1, "p2": new.p2, "Np": new.Np}})
        log.info("session %s: regrid epoch %d at record %d: %s -> "
                 "[%g, %g]x[%g, %g] %dx%d", self.id, plan.epoch, plan.k_star,
                 kinds, new.x1, new.x2, new.p1, new.p2, new.Nx, new.Np)
        self.clock.kick()

    # called from the router/streamer coroutines
    def apply_params(self, change, cp=None):
        """Apply the fields that actually DIFFER from what is live.

        A change that changes nothing is dropped whole — no worker command,
        no log entry, no event. The UI sends complete fields (PotentialEditor's
        "Apply live" always carries the U(x) draft, edited or not), and a
        param_log full of U changes that never happened makes an exported
        video's "how to reproduce this" block a lie about its own frames.
        """
        new, old = {}, {}

        def diff(field, value, current):
            if value is not None and value != current:
                new[field] = value
                old[field] = current

        diff("U", change.U, self.cfg.potential)
        for f in ("mass", "c", "hbar_eff", "tol"):
            diff(f, getattr(change, f), getattr(self.cfg, f))
        diff("dt_sign", change.dt_sign, self.clock.sign)
        diff("auto_expand", change.auto_expand, self.auto_expand)
        if not new:
            return
        if "U" not in new:
            # unchanged expression: the workers keep their callables. Even a
            # simultaneous hbar change needs no new cp — Propagator.rebuild()
            # re-derives the Bopp-shifted arrays from the current hbar_eff.
            cp = None
        cmd = {"kind": "params", "cp": cp, "mass": new.get("mass"),
               "c": new.get("c"), "hbar_eff": new.get("hbar_eff"),
               "tol": new.get("tol")}
        physics = [f for f in ("U", "mass", "c", "hbar_eff", "tol") if f in new]
        if physics:
            with self._edge_lock:
                # U/hbar move the extended Bopp range. A plan in flight was
                # validated under the OLD physics, and the worker-side
                # non-finite check at regrid application is fatal by design
                # (lockstep geometry must stay uniform) — so revalidate the
                # plan's union window under the incoming values BEFORE any
                # worker can see them. The streamer's validation_grid()
                # check covers plans it could see; this closes the race of
                # a plan committing during that ~ms compile (plan commits
                # hold this same lock, so no plan can slip in mid-check).
                if ("U" in new or "hbar_eff" in new) \
                   and self._plan_pending() and self._prev_grid_state is not None:
                    reason = self._potential_invalid(
                        new.get("U", self.cfg.potential),
                        self._prev_grid_state.union(self.grid_state),
                        new.get("hbar_eff", self.cfg.hbar_eff))
                else:
                    reason = None
                if reason is not None:
                    self.post_msg({"type": "error", "code": "bad_potential",
                                   "message": "rejected while a domain "
                                   "change is in flight: %s" % reason})
                    # only the physics is rejected; dt_sign/auto_expand in the
                    # same message still stand (and still get logged)
                    for f in physics:
                        new.pop(f)
                        old.pop(f)
                else:
                    for w in self.workers:
                        w.cmd_q.put(cmd)
                    # Track the applied values on the session: later U compiles
                    # validate on the extended Bopp range, which depends on the
                    # CURRENT hbar_eff, and status() must report current physics.
                    # Optimistic — a worker that rejects the change rolls itself
                    # back and posts an error.
                    for f in ("mass", "c", "hbar_eff", "tol"):
                        if f in new:
                            setattr(self.cfg, f, new[f])
                    if "U" in new:
                        self.cfg.potential = new["U"]
                        self.compiled_potential = cp
                        self._invalid_posted = False  # new U: expansion may work
        if not new:
            return                      # nothing survived the rejection above
        if "dt_sign" in new:
            self.clock.set_sign(new["dt_sign"])
        if "auto_expand" in new:
            # AFTER the physics commit: an immediate schedule (the key
            # "warning fired -> user enables the toggle" flow — waiting for
            # the next report_edge would compute one more old-grid record,
            # and a PAUSED session would not schedule at all until after
            # Solve) must validate against the values that just landed,
            # even when both arrive in one combined message.
            # cfg too, not just the session flag: the exported metadata block
            # and its JSON twin read cfg, and they must not report
            # "auto-expand: off" for a run that visibly expanded
            self.auto_expand = self.cfg.auto_expand = bool(new["auto_expand"])
            if self.auto_expand:
                with self._edge_lock:
                    axes = list(self.boundary_state["axes"])
                    want = (bool(axes) and not self._plan_pending()
                            and not self._invalid_posted)
                if want:
                    self._schedule_regrid(self.history.latest_complete(), axes)
        self.clock.kick()
        at_record = self.history.latest_complete() + 1
        # `before` as well as `applied`: an export renders "ℏ 1 → 2" and
        # rewinds the header physics to the first exported record (describe.py)
        self.param_log.append({"at_record": at_record, "applied": new,
                               "before": old})
        self.post_msg({"type": "params_applied", "applied": new, "before": old,
                       "at_record": at_record})

    def status(self):
        first, last = self.history.extent()
        t0, t1_ = self.history.t_extent()
        gs = self.grid_state
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
            "history_cap_bytes": self.history.byte_cap,
            "devices": self.devices,
            # the LIVE expression: the setup form greys out "Apply live" when
            # its draft already equals it (a no-op is dropped anyway)
            "potential": self.cfg.potential,
            "mass": self.cfg.mass,
            "c": self.cfg.c,
            "hbar_eff": self.cfg.hbar_eff,
            "tol": self.cfg.tol,
            "grid": {"x1": gs.x1, "x2": gs.x2, "Nx": gs.Nx,
                     "p1": gs.p1, "p2": gs.p2, "Np": gs.Np},
            "auto_expand": self.auto_expand,
            "max_grid": self.max_grid,
            "boundary": self.boundary_state,
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
        videoexport.close_session(self.id)   # cancel exports, unlink files
        for w in self.workers:
            w.stop()
        for w in self.workers:
            w.join(timeout=3.0)
        with _LOCK:
            SESSIONS.pop(self.id, None)


def create_session(cfg, compiled_potential, device, fft_threads, history_bytes,
                   max_grid=4096):
    loop = asyncio.get_running_loop()
    s = SimSession(cfg, compiled_potential, loop, device, fft_threads,
                   history_bytes, max_grid=max_grid)
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
        videoexport.sweep(now)     # unlink exported files past their TTL
        for s in list(SESSIONS.values()):
            if not s.ws_attached and now - s.last_seen > WS_IDLE_TTL:
                log.info("session %s idle > %.0fs, closing", s.id, WS_IDLE_TTL)
                # close() joins worker threads (up to seconds for a wedged
                # one) — never block the event loop the streamers run on
                await asyncio.to_thread(s.close)
