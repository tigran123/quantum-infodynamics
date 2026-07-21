/** Session configuration shared by the setup panels and the view. */

import { C_AU } from './units'
import type { VariantKey } from './variants'

export interface GridCfg {
  x1: number
  x2: number
  Nx: number
  p1: number
  p2: number
  Np: number
}

export interface ICComponentCfg {
  x0: number
  p0: number
  sigma_x: number
  sigma_p: number | null
  weight: number
  phase: number
}

export interface ICCfg {
  type: 'mixture' | 'cat'
  components: ICComponentCfg[]
}

export interface SimConfig {
  grid: GridCfg
  potential: string
  ic: ICCfg
  variants: VariantKey[]
  mass: number
  c: number
  hbar_eff: number
  tol: number
  record_dt: number
  delay: number
  mode: 'interactive' | 'runahead'
  t2: number
  // boundary watch response: detection always runs server-side; this only
  // decides whether the domain auto-moves/doubles (live-toggleable)
  auto_expand: boolean
}

const STORAGE_KEY = 'wignerf.cfg'

/** Load the persisted setup (merged over defaults) — a hard reload must
 *  not silently reset mode/t2/grid/IC, or the user ends up running a
 *  different simulation than they configured. */
export function loadConfig(): SimConfig {
  const d = defaultConfig()
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (s && typeof s === 'object') {
      Object.assign(d.grid, s.grid ?? {})
      if (Array.isArray(s.ic?.components) && s.ic.components.length) {
        d.ic.type = s.ic.type === 'cat' ? 'cat' : 'mixture'
        d.ic.components = s.ic.components
      }
      for (const k of ['potential', 'mass', 'c', 'hbar_eff', 'tol',
                       'record_dt', 'delay', 'mode', 't2',
                       'auto_expand'] as const) {
        if (k in s) (d as unknown as Record<string, unknown>)[k] = s[k]
      }
      if (Array.isArray(s.variants)) {
        const v = s.variants.filter((x: string) =>
          ['qn', 'qr', 'cn', 'cr'].includes(x))
        if (v.length) d.variants = v
      }
    }
  } catch { /* corrupted storage -> defaults */ }
  return d
}

export function saveConfig(c: SimConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(c))
}

/** Restore the whole setup to defaults IN PLACE — the view holds the
 *  config in a long-lived reactive object, so nested objects/arrays must
 *  be mutated, not replaced, for existing bindings to keep working (the
 *  deep watcher then persists the defaults to localStorage). */
export function resetToDefaults(c: SimConfig) {
  const d = defaultConfig()
  Object.assign(c.grid, d.grid)
  c.ic.type = d.ic.type
  c.ic.components.splice(0, c.ic.components.length, ...d.ic.components)
  c.variants.splice(0, c.variants.length, ...d.variants)
  c.potential = d.potential
  c.mass = d.mass
  c.c = d.c
  c.hbar_eff = d.hbar_eff
  c.tol = d.tol
  c.record_dt = d.record_dt
  c.delay = d.delay
  c.mode = d.mode
  c.t2 = d.t2
  c.auto_expand = d.auto_expand
}

export function defaultConfig(): SimConfig {
  return {
    grid: { x1: -6.0, x2: 6.0, Nx: 256, p1: -7.0, p2: 7.0, Np: 256 },
    potential: 'x^2/2',
    ic: {
      type: 'mixture',
      // sigma = 0.70711 (not 0.707): sigma_x*sigma_p must be >= hbar/2 = 0.5
      // or the default state is (marginally) sub-Heisenberg and the purity
      // warning fires on first load. 0.70711^2 = 0.5000045.
      components: [
        { x0: 2.0, p0: 0.0, sigma_x: 0.70711, sigma_p: 0.70711, weight: 1, phase: 0 },
      ],
    },
    variants: ['qn', 'cn'],
    mass: 1.0,
    c: C_AU,
    hbar_eff: 1.0,
    tol: 0.01,
    record_dt: 0.05,
    delay: 0.0,   // seconds injected between played-back frames (0 = max speed)
    mode: 'interactive',
    t2: 20.0,
    auto_expand: false,
  }
}
