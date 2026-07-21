"""
SolverWorker: one thread per ticked variant. Owns its own ArrayBackend
(FFT plans and CUDA context are per-thread), integrates with its own
adaptive dt, and lands exactly on every record time tau_k of the session
clock by clamping its final substep. Emits quantized VariantFrames into the
session's FrameHistory and signals the asyncio streamer.

Commands arrive on cmd_q as dicts:
  {"kind": "params", "cp": CompiledPotential|None, "mass": ..., "c": ...,
   "hbar_eff": ..., "tol": ...}   (absent keys = unchanged)
"""

import logging
import queue
import threading
import traceback
from time import monotonic

from . import boundary, initial, observables
from .grid import GridState, embed_window
from .propagator import Propagator
from .protocol import VARIANTS, VariantFrame, variant_id
from .quantize import quantize
from .xp import ArrayBackend

log = logging.getLogger(__name__)

_EXP_CACHE_MAX = 8


class SolverWorker(threading.Thread):
    def __init__(self, session, key, slot, device):
        super().__init__(daemon=True, name="wignerf-%s-%s" % (session.id, key))
        self.session = session
        self.key = key
        self.slot = slot
        self.device = device
        self.flavor = VARIANTS[key]
        self.cmd_q = queue.Queue()
        self.stop_evt = threading.Event()
        self._grid_state = None      # live window; regrids replace it
        self._applied_epoch = 0      # newest RegridPlan epoch applied here
        self.force_adjust = True
        self.dt = 0.0
        self.steps_total = 0
        self.steps_per_sec = 0.0
        self._exp_cache = {}
        self._rate_mark = (0, monotonic())

    def stop(self):
        self.stop_evt.set()
        self.session.clock.kick()

    # -- thread body --------------------------------------------------------

    def run(self):
        try:
            self._run()
        except Exception as e:
            log.exception("worker %s failed", self.name)
            self.session.post_error("variant '%s' solver died: %s" % (self.key, e),
                                    detail=traceback.format_exc())
            # Pause the session: with this variant dead the lockstep
            # frontier can never advance, and the siblings would fill the
            # history with records that never complete (and never evict).
            self.session.clock.set_running(False)
        finally:
            self._release_gpu_pool()

    def _release_gpu_pool(self):
        """Return unused CuPy pool blocks to the driver when the session
        closes. CuPy pools freed memory per process, so nvidia-smi keeps
        showing it as 'used' even when idle — releasing here means closed
        sessions visibly give their VRAM back. The pool is per-device, so
        re-enter THIS worker's device; blocks still referenced by other
        live sessions are untouched (free_all_blocks frees only free
        blocks)."""
        backend = getattr(self, "_backend", None)
        if backend is None or not backend.is_gpu:
            return
        try:
            with backend.device():
                xp = backend.xp
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            log.debug("GPU pool release failed", exc_info=True)

    def _run(self):
        cfg = self.session.cfg
        backend = ArrayBackend(device=self.device,
                               fft_threads=self.session.fft_threads)
        self._backend = backend
        with backend.device():
            # the worker's window mirrors the session's GridState (same
            # dataclass arithmetic), so record geometry and lattice points
            # agree with the scheduler bitwise
            self._grid_state = GridState.from_spec(cfg.grid)
            g, Wnat, _ = initial.from_spec(
                cfg.grid, cfg.ic, cfg.hbar_eff, backend,
                grid=self._grid_state.make_grid(backend))
            U, dUdx = self.session.compiled_potential.for_backend(backend)
            prop = Propagator(g, mass=cfg.mass, c=cfg.c, hbar_eff=cfg.hbar_eff,
                              tol=cfg.tol, U=U, dUdx=dUdx, **self.flavor)
            if not self._finite(prop, backend):
                raise ValueError("non-finite propagator exponents "
                                 "(check U(x), mass, c)")
            W = g.shift2d(Wnat)
            t = cfg.t1
            self.dt = cfg.record_dt/8.
            self._emit(0, t, W, prop, backend)      # record 0 = the Cauchy data
            frontier = 0
            while not self.stop_evt.is_set():
                self._drain_commands(prop, backend)
                tgt = self.session.clock.next_target(
                    frontier, self.session.history.latest_complete())
                if tgt is None:
                    self.session.clock.wait_work(0.1)
                    continue
                k, t_tgt = tgt
                # a scheduled regrid applies to every record >= its k_star
                # (k_star is past all in-flight records, so the switch is
                # lockstep-uniform); the epoch guard makes stale plans inert
                plan = self.session.current_regrid()
                if plan is not None and k >= plan.k_star \
                   and self._applied_epoch < plan.epoch:
                    W = self._apply_regrid(plan, prop, backend, W)
                    self._applied_epoch = plan.epoch
                W, t = self._advance(prop, W, t, t_tgt)
                self._emit(k, t_tgt, W, prop, backend)
                frontier = k

    # -- stepping -----------------------------------------------------------

    @staticmethod
    def _finite(prop, backend):
        xp = backend.xp
        return bool(xp.isfinite(prop.dU).all()) and bool(xp.isfinite(prop.dT).all())

    def _exponents(self, prop, dts):
        pair = self._exp_cache.get(dts)
        if pair is None:
            if len(self._exp_cache) >= _EXP_CACHE_MAX:
                self._exp_cache.clear()
            pair = prop.exponents(dts)
            self._exp_cache[dts] = pair
        return pair

    def _advance(self, prop, W, t, t_tgt):
        eps = 1e-12*max(1.0, abs(t_tgt))
        while abs(t_tgt - t) > eps and not self.stop_evt.is_set():
            direction = 1.0 if t_tgt > t else -1.0
            if self.dt == 0.0 or (self.dt > 0) != (direction > 0):
                self.dt = direction*(abs(self.dt) or self.session.cfg.record_dt/8.)
                self._exp_cache.clear()
                self.force_adjust = True
            rem = t_tgt - t
            adjust_due = self.force_adjust or self.steps_total % 20 == 0
            if adjust_due and abs(self.dt) <= abs(rem):
                # adjust_step only ever shrinks (as in solve.py): give dt a
                # chance to climb back after a transient (a stiff U applied
                # live, then reverted) by trying a 1/0.7 larger step — the
                # controller shrinks it right back if the accuracy is not
                # there. |rem| <= record_dt caps growth at one record.
                dt_try = self.dt/0.7
                if self.force_adjust or abs(dt_try) > abs(rem):
                    dt_try = self.dt
                W, self.dt, eU, eT = prop.adjust_step(dt_try, W)
                self._exp_cache = {self.dt: (eU, eT)}
                self.force_adjust = False
                t += self.dt
            else:
                dts = direction*min(abs(self.dt), abs(rem))
                W = prop.solve_spectral(W, *self._exponents(prop, dts))
                t += dts
            self.steps_total += 1
        return W, t_tgt    # land exactly on the record time (no drift)

    def _emit(self, k, t, W, prop, backend):
        wq, wmin, wmax = quantize(W, backend)
        obs = observables.compute(W, prop)
        vf = VariantFrame(vid=variant_id(**self.flavor), wq=wq,
                          wmin=wmin, wmax=wmax, E=obs.E,
                          x_mean=obs.x_mean, x_std=obs.x_std,
                          p_mean=obs.p_mean, p_std=obs.p_std,
                          purity=obs.purity,
                          dt=self.dt, rho=obs.rho, phi=obs.phi)
        self.session.history.put(k, t, self.slot, vf, self._grid_state.geom())
        # boundary watch every record: O(Nx+Np) host sums on the marginals
        # observables already brought over — no extra device sync
        self.session.report_edge(
            self.slot, k,
            boundary.edge_report(obs.rho, obs.phi, prop.grid.dx, prop.grid.dp))
        self.session.notify_frame()
        # a landing record may open the skew gate for waiting siblings
        self.session.clock.kick()
        n, mark = self._rate_mark
        now = monotonic()
        if now - mark > 1.0:
            self.steps_per_sec = (self.steps_total - n)/(now - mark)
            self._rate_mark = (self.steps_total, now)

    # -- regrid -------------------------------------------------------------

    def _apply_regrid(self, plan, prop, backend, W):
        """Exact fixed-lattice regrid of the live state: whole-cell window
        move/double on the frozen (dx, dp) lattice — W values are COPIED to
        their identical lattice points, entering cells are zero, nothing is
        ever interpolated. The transform runs in natural order (the window
        overlap is contiguous there; fftshifted order would split it)."""
        old, new = self._grid_state, plan.state
        Wnew = embed_window(prop.grid.unshift2d(W), old, new, backend.xp)
        g = new.make_grid(backend)
        prop.set_grid(g)
        if not self._finite(prop, backend):
            # the session pre-validated U on the new window, so this is a
            # genuine invariant break -> the worker-death path pauses the run
            raise ValueError("non-finite propagator exponents after regrid "
                             "to [%g, %g]x[%g, %g]"
                             % (new.x1, new.x2, new.p1, new.p2))
        self._grid_state = new
        self._exp_cache.clear()
        self.force_adjust = True
        log.info("%s: regrid epoch %d applied at k>=%d: [%g, %g]x[%g, %g] %dx%d",
                 self.name, plan.epoch, plan.k_star,
                 new.x1, new.x2, new.p1, new.p2, new.Nx, new.Np)
        return g.shift2d(Wnew)

    # -- commands -----------------------------------------------------------

    def _drain_commands(self, prop, backend):
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            if cmd.get("kind") != "params":
                continue
            prev = dict(U=prop.U, dUdx=prop.dUdx, mass=prop.mass, c=prop.c,
                        hbar_eff=prop.hbar_eff, tol=prop.tol)
            try:
                kwargs = {}
                if cmd.get("cp") is not None:
                    U, dUdx = cmd["cp"].for_backend(backend)
                    kwargs.update(U=U, dUdx=dUdx)
                for f in ("mass", "c", "hbar_eff", "tol"):
                    if cmd.get(f) is not None:
                        kwargs[f] = cmd[f]
                prop.set_physics(**kwargs)
                if not self._finite(prop, backend):
                    raise ValueError("non-finite propagator exponents")
            except Exception as e:
                prop.set_physics(**prev)   # roll back, keep evolving
                self.session.post_error(
                    "variant '%s': parameter change rejected (%s)" % (self.key, e))
            else:
                self._exp_cache.clear()
                self.force_adjust = True
