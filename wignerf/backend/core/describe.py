"""
Human-readable descriptions of a session: the analytic initial condition and
the parameter block that make an exported video self-contained (everything
needed to reproduce the run is ON the frame).

Plain Unicode text ONLY — never matplotlib mathtext: U(x) is user input
("x^2/2", "Abs(x)", ...) and would either fail to parse or render as
nonsense in math mode. The same facts also go into the mp4 metadata tag as
JSON (`config_json`), which is what a machine should read back.

No matplotlib import here on purpose: this module is pure string work and
is unit-tested as such.
"""

import json


def _num(v):
    """Compact number formatting: 2 stays "2", 0.70711 keeps its digits."""
    f = float(v)
    if f == int(f) and abs(f) < 1e16:
        return "%d" % int(f)
    return "%.6g" % f


def _shift(var, v):
    """"x − 2.5" / "x + 2.5" / "x" — never the "x−−2.5" a bare _num gives."""
    f = float(v)
    if f == 0.0:
        return var
    return "%s %s %s" % (var, "−" if f > 0 else "+", _num(abs(f)))


def _sigma_p_of(comp, ic_type, hbar_eff):
    """sigma_p as the solver uses it (derived for cat states — see
    initial.minimal_sigma_p; components carry their own for mixtures)."""
    if ic_type == "cat":
        return hbar_eff/(2.*comp.sigma_x)
    return comp.sigma_p


def ic_expression(ic, hbar_eff):
    """The initial condition as an analytic expression, with the concrete
    numbers substituted. Returns a list of text lines.

    - mixture: W(x,p,0) itself (initial.mixture_wigner) — a weighted sum of
      normalized Gaussian blobs.
    - cat: the WAVEFUNCTION psi(x,0) (initial.cat_wigner builds W from the
      pairwise cross-Wigner closed form, which is far longer to print and
      carries no extra information): W = Wigner[psi] is the complete and
      compact specification.
    """
    comps = list(ic.components)
    hbar = float(hbar_eff)
    if ic.type == "cat":
        terms = []
        for c in comps:
            amp = "√%s·" % _num(c.weight) if c.weight != 1.0 else ""
            phase = "e^(i%s)·" % _num(c.phase) if c.phase else ""
            u = "(%s)" % _shift("x", c.x0)
            terms.append(
                "%s%s(2π·%s²)^(−1/4)·exp(−%s²/(4·%s²) + i·%s·%s/ℏ)"
                % (amp, phase, _num(c.sigma_x), u, _num(c.sigma_x),
                   _num(c.p0), u))
        lines = ["IC (cat state, ℏ = %s):  W(x,p,0) = Wigner[ψ],  ψ(x,0) = "
                 "S^(−1/2)·[" % _num(hbar)]
        lines.append("    " + "  +  ".join(terms) + " ]")
        sig = ", ".join("σ%d = %s (σp = %s)"
                        % (j + 1, _num(c.sigma_x),
                           _num(_sigma_p_of(c, "cat", hbar)))
                        for j, c in enumerate(comps))
        lines.append("    with %s;  S = ⟨ψ|ψ⟩ (analytic normalization)" % sig)
        return lines

    wtot = sum(c.weight for c in comps)
    terms = []
    for c in comps:
        sp = _sigma_p_of(c, "mixture", hbar)
        amp = c.weight/(wtot*2.*3.141592653589793*c.sigma_x*sp)
        terms.append("%s·exp(−(%s)²/(2·%s²) − (%s)²/(2·%s²))"
                     % (_num(amp), _shift("x", c.x0), _num(c.sigma_x),
                        _shift("p", c.p0), _num(sp)))
    return ["IC (Gaussian mixture):  W(x,p,0) = ",
            "    " + "  +  ".join(terms)]


def param_lines(cfg, param_log=(), k0=None, k1=None):
    """The physics/setup block. `param_log` entries inside [k0, k1] are
    listed: a live U/mass/ℏ change mid-range would otherwise make the block
    a lie about the frames that follow it."""
    # label the fields exactly as the UI does: ℏ (not hbar_eff — the Physics
    # panel calls it ℏ) and "run-ahead" (the mode select's label, not the
    # wire value "runahead")
    lines = [
        "U(x) = %s" % cfg.potential,
        "m = %s   c = %s   ℏ = %s   tol = %s"
        % (_num(cfg.mass), _num(cfg.c), _num(cfg.hbar_eff), _num(cfg.tol)),
        "t₁ = %s   record_dt = %s   mode = %s%s   auto-expand: %s"
        % (_num(cfg.t1), _num(cfg.record_dt),
           "run-ahead" if cfg.mode == "runahead" else cfg.mode,
           ("  t₂ = %s" % _num(cfg.t2)) if cfg.t2 is not None else "",
           "on" if cfg.auto_expand else "off"),
    ]
    changes = [e for e in param_log
               if (k0 is None or e["at_record"] >= k0)
               and (k1 is None or e["at_record"] <= k1)]
    for e in changes:
        lines.append("live change at record %d: %s"
                     % (e["at_record"],
                        ", ".join("%s = %s" % (k, v)
                                  for k, v in e["applied"].items())))
    return lines


def config_json(cfg, param_log=(), **extra):
    """Machine-readable twin of the visible block (mp4 `comment` tag)."""
    d = {"generator": "wignerf",
         "config": json.loads(cfg.model_dump_json()),
         "param_log": list(param_log)}
    d.update(extra)
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
