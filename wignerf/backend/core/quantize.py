"""
Frame quantization: float64 W -> uint16 + (Wmin, Wmax). Runs on the device
(the cast happens before the PCIe download on GPU backends).
"""


def quantize(W, backend):
    xp = backend.xp
    wmin = float(W.min())
    wmax = float(W.max())
    span = wmax - wmin
    if span < 1e-300:
        span = 1.0
    q = xp.clip(xp.rint((W - wmin)*(65535.0/span)), 0, 65535).astype(xp.uint16)
    return backend.asnumpy(q), wmin, wmax


def dequantize(q, wmin, wmax):
    """Host-side inverse (tests/ws_smoke)."""
    return wmin + q.astype("float64")*((wmax - wmin)/65535.0)
