"""
Array/FFT backend selection: CuPy on a CUDA device or NumPy on the CPU.

Each SolverWorker owns its own ArrayBackend instance, so FFT plans and
buffers are never shared between threads (pyFFTW plan execution is not
re-entrant; CuPy device context is per-thread).

Device selection (config.py reads WIGNERF_DEVICE). The spec names a POOL:
resolve_devices() expands it to an ordered list of concrete devices,
fastest first, and each session spreads its variant workers over that list
(core/session.py assign_devices). ArrayBackend itself always binds ONE
concrete device.
  "auto"          - all CUDA devices fastest-first if cupy imports, else CPU.
  "cpu"           - NumPy; FFT provider chain: pyfftw -> scipy.fft -> numpy.fft.
  "cuda:N"        - CuPy on CUDA device N.
  "cuda:1,cuda:0" - explicit pool; the written order IS the speed ranking.
"""

from contextlib import nullcontext
import logging
import os

import numpy

# CUDA's default enumeration is fastest-first, which disagrees with
# nvidia-smi's PCI order (and would make "cuda:1" mean different cards in
# different tools). Pin PCI order BEFORE cupy is first imported so device
# indices always match nvidia-smi.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = logging.getLogger(__name__)

# Speed of light in Hartree atomic units (1/alpha, CODATA 2018).
C_AU = 137.035999084


def _import_cupy():
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception as e:
        log.info("cupy unavailable: %s", e)
    return None


def _gpu_speed_key(cupy, i):
    """Crude compute ranking: SM count first (82 on the 3090 vs 68 on the
    2080 Ti), total memory as the tiebreak. clockRate is deprecated (reads
    0 under CUDA 13) so it cannot participate."""
    props = cupy.cuda.runtime.getDeviceProperties(i)
    return (props["multiProcessorCount"], props["totalGlobalMem"])


def resolve_devices(spec):
    """Expand a WIGNERF_DEVICE spec into an ordered list of concrete device
    strings ("cuda:N" / "cpu"), fastest first. An explicit comma list is
    trusted as written — its order is the speed ranking."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty device spec %r" % spec)
    if parts == ["auto"]:
        cupy = _import_cupy()
        if cupy is None:
            return ["cpu"]
        n = cupy.cuda.runtime.getDeviceCount()
        order = sorted(range(n), key=lambda i: _gpu_speed_key(cupy, i),
                       reverse=True)
        return ["cuda:%d" % i for i in order]
    if len(set(parts)) != len(parts):
        raise ValueError("duplicate devices in %r" % spec)
    for p in parts:
        if p == "cpu":
            continue
        if p == "auto":
            raise ValueError("'auto' cannot appear in a device list")
        if not p.startswith("cuda"):
            raise ValueError("unknown device spec %r" % p)
        cupy = _import_cupy()
        if cupy is None:
            raise RuntimeError("%r requested but cupy/CUDA is not available" % p)
        idx = int(p.split(":")[1]) if ":" in p else 0
        if idx >= cupy.cuda.runtime.getDeviceCount():
            raise RuntimeError("CUDA device %d does not exist" % idx)
    return parts


class ArrayBackend:
    def __init__(self, device="auto", fft_threads=1):
        self.fft_threads = max(1, int(fft_threads))
        self.xp = numpy
        self.is_gpu = False
        self.device_index = None
        self.fft_provider = None

        if device == "auto":
            device = resolve_devices("auto")[0]   # fastest device in the pool
        if device == "cpu":
            self._use_cpu()
        elif device.startswith("cuda"):
            cupy = _import_cupy()
            if cupy is None:
                raise RuntimeError("WIGNERF_DEVICE=%s but cupy/CUDA is not available" % device)
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx >= cupy.cuda.runtime.getDeviceCount():
                raise RuntimeError("CUDA device %d does not exist" % idx)
            self._use_gpu(cupy, idx)
        else:
            raise ValueError("unknown device spec %r" % device)

    def _use_gpu(self, cupy, index):
        self.xp = cupy
        self.is_gpu = True
        self.device_index = index
        self.fft_provider = "cupy"
        with cupy.cuda.Device(index):
            name = cupy.cuda.runtime.getDeviceProperties(index)["name"].decode()
        self.name = "cuda:%d (%s)" % (index, name)
        log.info("backend: %s", self.name)

    def _use_cpu(self):
        # Provider chain in the spirit of dynamics/solve.py: pyfftw is the
        # fastest, numpy the slowest.
        try:
            import pyfftw  # noqa: F401
            self.fft_provider = "pyfftw"
        except ImportError:
            try:
                import scipy.fft  # noqa: F401
                self.fft_provider = "scipy"
            except ImportError:
                self.fft_provider = "numpy"
        self.name = "cpu (%s)" % self.fft_provider
        log.info("backend: %s", self.name)

    # -- device/context helpers ------------------------------------------

    def device(self):
        """Context manager binding this backend's CUDA device to the current
        thread (no-op on CPU). Workers wrap their whole run loop in it."""
        if self.is_gpu:
            return self.xp.cuda.Device(self.device_index)
        return nullcontext()

    def synchronize(self):
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def asnumpy(self, a):
        return self.xp.asnumpy(a) if self.is_gpu else a

    def fftshift(self, a, axes=None):
        return self.xp.fft.fftshift(a, axes=axes)

    def ifftshift(self, a, axes=None):
        return self.xp.fft.ifftshift(a, axes=axes)

    # -- FFT plan factory --------------------------------------------------

    def fft_pair(self, shape, axis):
        """Return (fft, ifft) callables for complex128 arrays of the given
        shape along the given axis. pyFFTW plans/buffers are owned by this
        backend instance and must not be called from two threads at once."""
        if self.fft_provider == "cupy":
            xp = self.xp
            return (lambda a: xp.fft.fft(a, axis=axis),
                    lambda a: xp.fft.ifft(a, axis=axis))
        if self.fft_provider == "pyfftw":
            import pyfftw
            kw = dict(axis=axis, threads=self.fft_threads,
                      planner_effort="FFTW_ESTIMATE",
                      overwrite_input=False, auto_align_input=True,
                      auto_contiguous=True, avoid_copy=False)
            a = pyfftw.empty_aligned(shape, dtype="complex128")
            fwd = pyfftw.builders.fft(a, **kw)
            b = pyfftw.empty_aligned(shape, dtype="complex128")
            bwd = pyfftw.builders.ifft(b, **kw)
            return fwd, bwd
        if self.fft_provider == "scipy":
            import scipy.fft as sfft
            workers = self.fft_threads
            return (lambda a: sfft.fft(a, axis=axis, workers=workers),
                    lambda a: sfft.ifft(a, axis=axis, workers=workers))
        return (lambda a: numpy.fft.fft(a, axis=axis),
                lambda a: numpy.fft.ifft(a, axis=axis))
