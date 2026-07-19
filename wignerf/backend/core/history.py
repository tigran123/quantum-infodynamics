"""
FrameHistory: the in-RAM producer/consumer store between solver workers and
the WebSocket streamer.

Records are indexed by an append-only integer n (the record grid); each
record holds one VariantFrame slot per session variant plus the physical
time t_n. A record is "complete" when every variant has landed on it —
`latest_complete()` is the lockstep frontier the streamer reads.

Byte-capped: oldest records are evicted first; the timeline learns the
retained window from `extent()` / eviction notices.
"""

import threading


class FrameHistory:
    def __init__(self, n_variants, byte_cap):
        self.n_variants = int(n_variants)
        self.byte_cap = int(byte_cap)
        self._lock = threading.Lock()
        self._recs = {}     # n -> [VariantFrame | None] * n_variants
        self._t = {}        # n -> float
        self._bytes = 0
        self._first = 0            # oldest retained index
        self._complete = -1        # highest n with records 0..n... complete contiguously
        self._frontier = [-1]*self.n_variants
        self.evicted = False       # set when eviction has occurred since last check

    @staticmethod
    def _frame_bytes(vf):
        return vf.wq.nbytes + vf.rho.nbytes + vf.phi.nbytes + 96

    def put(self, n, t, slot, vframe):
        with self._lock:
            rec = self._recs.get(n)
            if rec is None:
                rec = [None]*self.n_variants
                self._recs[n] = rec
                self._t[n] = t
            if rec[slot] is not None:
                self._bytes -= self._frame_bytes(rec[slot])
            rec[slot] = vframe
            self._bytes += self._frame_bytes(vframe)
            self._frontier[slot] = max(self._frontier[slot], n)
            # advance the lockstep frontier
            while True:
                nxt = self._recs.get(self._complete + 1)
                if nxt is None or any(v is None for v in nxt):
                    break
                self._complete += 1
            # evict oldest complete records over the cap (keep at least 2)
            while (self._bytes > self.byte_cap
                   and self._first < self._complete - 1):
                old = self._recs.pop(self._first, None)
                self._t.pop(self._first, None)
                if old:
                    self._bytes -= sum(self._frame_bytes(v) for v in old if v)
                self._first += 1
                self.evicted = True

    def get(self, n):
        """(t, [VariantFrame...]) if record n is retained and complete."""
        with self._lock:
            if n < self._first or n > self._complete:
                return None
            rec = self._recs.get(n)
            if rec is None or any(v is None for v in rec):
                return None
            return self._t[n], list(rec)

    def latest_complete(self):
        with self._lock:
            return self._complete

    def variant_frontier(self, slot):
        with self._lock:
            return self._frontier[slot]

    def extent(self):
        """(first retained, last complete) — (-1, -1) when empty."""
        with self._lock:
            if self._complete < 0:
                return (-1, -1)
            return (self._first, self._complete)

    def t_extent(self):
        with self._lock:
            if self._complete < 0:
                return (None, None)
            return (self._t.get(self._first), self._t.get(self._complete))

    def nbytes(self):
        with self._lock:
            return self._bytes

    def take_evicted_flag(self):
        with self._lock:
            e, self.evicted = self.evicted, False
            return e
