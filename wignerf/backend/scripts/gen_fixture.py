"""
Write a golden binary frame bundle + JSON metadata for the frontend
decoder's vitest (frontend/src/lib/__fixtures__/). Run from backend/:

    .venv/bin/python scripts/gen_fixture.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import protocol

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "__fixtures__"


def main():
    rng = np.random.default_rng(42)
    Nx, Np = 8, 4
    variants = []
    meta = []
    for vid in (protocol.variant_id(True, False), protocol.variant_id(False, True)):
        wq = rng.integers(0, 65536, size=(Nx, Np), dtype=np.uint16)
        rho = rng.random(Nx).astype(np.float32)
        phi = rng.random(Np).astype(np.float32)
        vf = protocol.VariantFrame(vid=vid, wq=wq, wmin=-0.25, wmax=1.5,
                                   E=2.5, x_mean=1.0, x_std=0.5,
                                   p_mean=-1.0, p_std=0.25, purity=0.875,
                                   dt=0.01, rho=rho, phi=phi)
        variants.append(vf)
        meta.append({"vid": vid, "wmin": -0.25, "wmax": 1.5, "E": 2.5,
                     "x_mean": 1.0, "x_std": 0.5, "p_mean": -1.0,
                     "p_std": 0.25, "purity": 0.875, "dt": 0.01,
                     "wq": wq.flatten().tolist(),
                     "rho": rho.tolist(), "phi": phi.tolist()})

    geom = protocol.RecordGeom(Nx=Nx, Np=Np, x1=-6.0, x2=6.0, p1=-7.0, p2=7.0)
    buf = protocol.pack_frame(7, 0.35, geom, variants,
                              flags=protocol.FLAG_REPLAY)
    # decode round-trip as a self-check
    f = protocol.unpack_frame(buf)
    assert (f.record, f.t) == (7, 0.35) and f.geom == geom

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frame.bin").write_bytes(buf)
    (OUT / "frame.json").write_text(json.dumps({
        "record": 7, "t": 0.35, "Nx": Nx, "Np": Np,
        "x1": geom.x1, "x2": geom.x2, "p1": geom.p1, "p2": geom.p2,
        "flags": protocol.FLAG_REPLAY, "variants": meta}, indent=1))
    print("wrote", OUT / "frame.bin", len(buf), "bytes")


if __name__ == "__main__":
    main()
