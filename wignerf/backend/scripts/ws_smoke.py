"""
Headless smoke test against a LIVE server (no browser needed):

    .venv/bin/uvicorn main:app --port 8010 &
    .venv/bin/python scripts/ws_smoke.py [http://127.0.0.1:8010]

Creates a harmonic four-variant session, plays it, and asserts streaming
invariants: monotone record indices, exact record-time spacing, unit norm
after dequantization, flat energy, lockstep bundles, exact seek.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import websockets

from core import protocol
from core.quantize import dequantize

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"

CFG = {
    "grid": dict(x1=-6.0, x2=6.0, Nx=256, p1=-7.0, p2=7.0, Np=256),
    "potential": "x^2/2",
    "ic": {"type": "mixture",
           "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707,
                           "sigma_p": 0.707}]},
    "variants": ["qn", "qr", "cn", "cr"],
    "c": 10.0,          # low c so the relativistic variants visibly differ
    "record_dt": 0.05,
    "delay": 0.0,      # seconds between played-back frames (0 = max speed)
}


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        r = await http.post("/api/sessions", json=CFG)
        r.raise_for_status()
        info = r.json()
        sid = info["session_id"]
        print("session", sid, "variants", info["variants"])

        ws_url = BASE.replace("http", "ws") + info["ws_url"]
        # compression=None: never negotiate permessage-deflate (12x slower
        # for the multi-MiB frame bundles; the server disables it too)
        async with websockets.connect(ws_url, max_size=64*1024*1024,
                                      compression=None) as ws:
            await ws.send(json.dumps({"type": "play"}))
            frames, by_rec = [], {}
            while len(frames) < 20:
                m = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(m, (bytes, bytearray)):
                    f = protocol.unpack_frame(m)
                    frames.append(f)
                    by_rec[f.record] = f
                else:
                    d = json.loads(m)
                    if d["type"] == "error":
                        raise SystemExit("server error: %s" % d)

            recs = [f.record for f in frames]
            assert recs == sorted(set(recs)), "records not strictly increasing"
            for f in frames:
                assert abs(f.t - f.record*CFG["record_dt"]) < 1e-9, "t spacing broken"
                assert len(f.variants) == 4, "lockstep bundle incomplete"
                g = f.geom
                assert (g.x1, g.x2, g.p1, g.p2) == (-6.0, 6.0, -7.0, 7.0), \
                    "header geometry mismatch"
                for v in f.variants:
                    W = dequantize(v.wq, v.wmin, v.wmax)
                    norm = W.sum()*((g.x2 - g.x1)/g.Nx)*((g.p2 - g.p1)/g.Np)
                    assert abs(norm - 1.0) < 1e-2, "norm drifted: %g" % norm
            E0 = {v.vid: v.E for v in frames[0].variants}
            for v in frames[-1].variants:
                assert abs(v.E - E0[v.vid]) < 5e-3*max(1.0, abs(E0[v.vid])), \
                    "energy drift on vid %d" % v.vid
            print("streamed %d lockstep bundles up to record %d, invariants OK"
                  % (len(frames), recs[-1]))

            # exact seek while paused
            await ws.send(json.dumps({"type": "pause"}))
            target = recs[len(recs)//2]
            await ws.send(json.dumps({"type": "seek", "record": target}))
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=10)
                if isinstance(m, (bytes, bytearray)):
                    f = protocol.unpack_frame(m)
                    if f.record == target:
                        ref = by_rec[target]
                        assert f.t == ref.t
                        assert all((a.wq == b.wq).all()
                                   for a, b in zip(f.variants, ref.variants)), \
                            "seek returned different bytes"
                        break
            print("seek(%d) returned the identical record" % target)

        r = await http.delete("/api/sessions/%s" % sid)
        r.raise_for_status()
        print("OK")


if __name__ == "__main__":
    asyncio.run(main())
