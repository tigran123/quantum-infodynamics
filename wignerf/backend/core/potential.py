"""
Safe compilation of user-entered analytic potentials U(x).

Pipeline: token screen (the security boundary) -> sympy parse in a
whitelisted namespace -> per-variant-family validity -> lambdify for
numpy (and cupy when a GPU backend is active).

Validity model (see plan): the Bopp arguments of the quantum differential
are REAL (x -+ hbar*theta/2, complex dtype only), so:
- quantum-valid  = U real and finite on the extended real range
  [x1 - hbar*theta_amp/2, x2 + hbar*theta_amp/2] (numeric probe; Abs(x)
  is quantum-valid, no analyticity required). Complex values there
  (branch cuts) make the evolution non-unitary: reported as a warning,
  since an absorbing potential may be intended.
- classical-valid = dU/dx well-defined on [x1, x2] with no DiracDelta
  (so a Heaviside step potential is quantum-only).
"""

import io
import math
import tokenize
from dataclasses import dataclass, field

import numpy
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor)

MAX_EXPR_LEN = 500
PROBE_POINTS = 4096
MAX_POW_DIGITS = 300   # numeric powers must stay below ~1e300 (float range)

_X = sp.Symbol("x", real=True)

_SMOOTH = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log,
    "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    "pi": sp.pi, "E": sp.E,
}
_NONSMOOTH = {
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "Max": sp.Max, "Min": sp.Min, "Mod": sp.Mod,
    "Piecewise": sp.Piecewise, "Heaviside": sp.Heaviside,
}
_LOCALS = {**_SMOOTH, **_NONSMOOTH, "x": _X,
           "True": sp.true, "False": sp.false}   # Piecewise conditions
_ALLOWED_NAMES = set(_LOCALS)

_TRANSFORMS = standard_transformations + (convert_xor,)


class PotentialError(ValueError):
    pass


@dataclass
class CompiledPotential:
    expr_str: str
    expr: object
    dUdx_expr: object            # None when classical-invalid
    quantum_valid: bool
    classical_valid: bool
    reasons: list = field(default_factory=list)    # hard per-family failures
    warnings: list = field(default_factory=list)
    latex: str = ""
    dUdx_latex: str = ""
    U: object = None             # numpy callable, array in -> array out
    dUdx: object = None
    U_gpu: object = None         # cupy callables, built on demand
    dUdx_gpu: object = None

    def for_backend(self, backend):
        """(U, dUdx) callables for the given ArrayBackend."""
        if not backend.is_gpu:
            return self.U, self.dUdx
        if self.U_gpu is None:
            self.U_gpu = _lambdify(self.expr, "cupy")
            if self.dUdx_expr is not None:
                self.dUdx_gpu = _lambdify(self.dUdx_expr, "cupy")
        return self.U_gpu, self.dUdx_gpu


def _screen_tokens(src):
    """Reject anything but whitelisted names, numbers and operators BEFORE
    sympy's parse_expr (which evals) ever sees the string."""
    if len(src) > MAX_EXPR_LEN:
        raise PotentialError("expression too long (max %d characters)" % MAX_EXPR_LEN)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError as e:
        raise PotentialError("cannot tokenize expression: %s" % e) from None
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string not in _ALLOWED_NAMES:
            raise PotentialError("name '%s' is not allowed" % tok.string)
        if tok.type == tokenize.OP and tok.string in (".", "=", ":=", ";", "@"):
            raise PotentialError("operator '%s' is not allowed" % tok.string)
        if tok.type == tokenize.STRING:
            raise PotentialError("string literals are not allowed")


def _screen_powers(expr):
    """Reject numeric powers of astronomical magnitude BEFORE evaluation:
    parse_expr(evaluate=True) of e.g. 9**9**9 would try to materialize a
    ~4e8-digit integer, pinning a CPU and gigabytes of RAM. Called on the
    unevaluated parse; the magnitude estimate uses float logs only."""
    for node in sp.preorder_traversal(expr):
        if not (isinstance(node, sp.Pow) and node.base.is_number
                and node.exp.is_number):
            continue
        try:
            b = abs(complex(node.base.evalf(8)))
            p = abs(complex(node.exp.evalf(8)))
        except (OverflowError, TypeError, ValueError):
            raise PotentialError("numeric power is too large") from None
        if b > 0.0 and b != 1.0 and p*abs(math.log10(b)) > MAX_POW_DIGITS:
            raise PotentialError(
                "numeric power is too large (|result| would exceed 1e%d)"
                % MAX_POW_DIGITS)


def _lambdify(expr, modules):
    f = sp.lambdify(_X, expr, modules=modules)
    def wrapped(v, _f=f):
        r = _f(v)
        # constant expressions return scalars; broadcast to the input shape
        if getattr(r, "shape", ()) != getattr(v, "shape", ()):
            r = r*numpy.ones_like(v) if modules == "numpy" else r + 0*v
        return r
    return wrapped


def compile_potential(expr_str, x_range=None, x_extended=None):
    """Compile a U(x) expression string.

    x_range     -- (x1, x2): the visible grid range, used for the classical
                   dUdx probe.
    x_extended  -- (xa, xb): the extended range the quantum propagator
                   evaluates U on (xa = x1 - hbar*theta_amp/2, ...); when
                   None, the quantum probe is skipped (validity assumed).

    Raises PotentialError for expressions that are rejected outright
    (bad syntax, forbidden names, wrong free symbols). Per-family validity
    failures are reported in the returned object, not raised.
    """
    src = expr_str.strip()
    if not src:
        raise PotentialError("empty expression")
    _screen_tokens(src)
    # two-phase parse: an unevaluated pass feeds the power screen (nothing
    # is materialized), then the real evaluated parse
    try:
        unev = parse_expr(src, local_dict=_LOCALS, transformations=_TRANSFORMS,
                          evaluate=False)
    except Exception as e:
        raise PotentialError("parse error: %s" % e) from None
    if isinstance(unev, sp.Basic):
        _screen_powers(unev)
    try:
        expr = parse_expr(src, local_dict=_LOCALS, transformations=_TRANSFORMS,
                          evaluate=True)
    except Exception as e:
        raise PotentialError("parse error: %s" % e) from None
    if not isinstance(expr, sp.Expr):
        raise PotentialError("not a valid expression in x")
    if not expr.free_symbols <= {_X}:
        extra = ", ".join(sorted(str(s) for s in expr.free_symbols - {_X}))
        raise PotentialError("only 'x' may appear as a variable (got: %s)" % extra)
    if expr.has(sp.I):
        raise PotentialError("U(x) must be a real expression")

    cp = CompiledPotential(expr_str=src, expr=expr, dUdx_expr=None,
                           quantum_valid=True, classical_valid=True,
                           latex=sp.latex(expr))
    cp.U = _lambdify(expr, "numpy")

    # -- classical family: dU/dx -----------------------------------------
    try:
        d = sp.diff(expr, _X).doit()
    except Exception as e:
        d = None
        cp.classical_valid = False
        cp.reasons.append("classical: cannot differentiate (%s)" % e)
    if d is not None:
        if d.has(sp.DiracDelta):
            cp.classical_valid = False
            cp.reasons.append("classical: dU/dx contains a Dirac delta "
                              "(hard wall) - not representable as a force")
        else:
            if d.has(sp.Heaviside):
                cp.warnings.append("classical: dU/dx has a finite jump "
                                   "(Heaviside) - force is discontinuous")
            cp.dUdx_expr = d
            cp.dUdx_latex = sp.latex(d)
            cp.dUdx = _lambdify(d, "numpy")

    # -- symbolic pole detection (numeric probes can straddle a pole without
    # ever sampling it). Best effort: sympy cannot decide singularities for
    # every whitelisted expression (e.g. Piecewise) — then the numeric
    # probes below are the only guard.
    def _poles_in(e, lo, hi):
        try:
            s = sp.singularities(e, _X, sp.Interval(lo, hi))
            if isinstance(s, sp.sets.sets.FiniteSet):
                return [float(v) for v in s if v.is_real]
        except Exception:
            pass
        return []

    if x_extended is not None:
        pts = _poles_in(expr, *x_extended)
        if pts:
            cp.quantum_valid = False
            cp.reasons.append("quantum: U is singular at x = %s inside the "
                              "extended range [%.4g, %.4g] the propagator "
                              "evaluates it on"
                              % (", ".join("%.4g" % p for p in pts), *x_extended))
    if cp.classical_valid and cp.dUdx_expr is not None and x_range is not None:
        pts = _poles_in(cp.dUdx_expr, *x_range)
        if pts:
            cp.classical_valid = False
            cp.reasons.append("classical: dU/dx is singular at x = %s"
                              % ", ".join("%.4g" % p for p in pts))

    # -- numeric probes -----------------------------------------------------
    with numpy.errstate(all="ignore"):
        if cp.classical_valid and cp.dUdx is not None and x_range is not None:
            xs = numpy.linspace(x_range[0], x_range[1], PROBE_POINTS)
            dv = numpy.asarray(cp.dUdx(xs))
            bad = ~numpy.isfinite(dv)
            if bad.any():
                cp.classical_valid = False
                cp.reasons.append("classical: dU/dx is not finite for x in "
                                  "[%.4g, %.4g]" % (xs[bad].min(), xs[bad].max()))
        if x_extended is not None:
            xs = numpy.linspace(x_extended[0], x_extended[1],
                                PROBE_POINTS).astype(numpy.complex128)
            uv = numpy.asarray(cp.U(xs))
            bad = ~numpy.isfinite(uv)
            if bad.any():
                cp.quantum_valid = False
                xb = xs.real[bad.reshape(xs.shape) if bad.shape == xs.shape else bad]
                cp.reasons.append(
                    "quantum: U is not finite on the extended range "
                    "[%.4g, %.4g] the propagator evaluates it on"
                    % (xb.min(), xb.max()))
            else:
                im = numpy.abs(uv.imag) > 1e-12*(1. + numpy.abs(uv.real))
                if im.any():
                    xb = xs.real[im]
                    cp.warnings.append(
                        "quantum: U takes complex values for x in "
                        "[%.4g, %.4g] (extended range) - evolution will be "
                        "non-unitary (absorbing) there" % (xb.min(), xb.max()))
    return cp


def sample_potential(cp, x1, x2, n=400):
    """Sample U over [x1, x2] for the preview plot. Non-finite values are
    mapped to None (JSON null) so the client can show gaps."""
    xs = numpy.linspace(float(x1), float(x2), int(n))
    with numpy.errstate(all="ignore"):
        us = numpy.asarray(cp.U(xs), dtype=numpy.float64)
    ok = numpy.isfinite(us)
    return xs.tolist(), [float(u) if o else None for u, o in zip(us, ok)]
