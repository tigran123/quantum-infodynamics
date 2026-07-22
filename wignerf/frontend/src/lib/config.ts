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

/** The subset of SimConfig that applies LIVE (status reports it back): the
 *  setup form marks a field that differs from it as edited-but-not-applied. */
export type LivePhysics = Pick<SimConfig,
  'potential' | 'mass' | 'c' | 'hbar_eff' | 'tol'>

const STORAGE_KEY = 'wignerf.cfg'
const ALL_KEYS = ['qn', 'qr', 'cn', 'cr'] as const

/**
 * Merge a loosely-typed config (localStorage, an imported setup file, an
 * mp4's metadata) into `target`, IN PLACE — the view holds the config in a
 * long-lived reactive object, so nested objects/arrays must be mutated, not
 * replaced, for existing bindings to keep working. Unknown keys and fields
 * of the wrong shape are ignored; `t2: null` (an interactive session's wire
 * form) leaves the form's value alone.
 */
export function mergeConfig(target: SimConfig, s: unknown) {
  if (!s || typeof s !== 'object') return
  const src = s as Record<string, any>
  if (src.grid && typeof src.grid === 'object') Object.assign(target.grid, src.grid)
  if (Array.isArray(src.ic?.components) && src.ic.components.length) {
    target.ic.type = src.ic.type === 'cat' ? 'cat' : 'mixture'
    target.ic.components.splice(0, target.ic.components.length,
                                ...src.ic.components.map(
                                  (c: Record<string, unknown>) => ({ ...c })))
  }
  for (const k of ['potential', 'mass', 'c', 'hbar_eff', 'tol',
                   'record_dt', 'delay', 'mode', 't2',
                   'auto_expand'] as const) {
    if (k in src && src[k] != null)
      (target as unknown as Record<string, unknown>)[k] = src[k]
  }
  if (Array.isArray(src.variants)) {
    const v = src.variants.filter((x: string) =>
      (ALL_KEYS as readonly string[]).includes(x))
    if (v.length) target.variants.splice(0, target.variants.length, ...v)
  }
}

/** Load the persisted setup (merged over defaults) — a hard reload must
 *  not silently reset mode/t2/grid/IC, or the user ends up running a
 *  different simulation than they configured. */
export function loadConfig(): SimConfig {
  const d = defaultConfig()
  try {
    mergeConfig(d, JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null'))
  } catch { /* corrupted storage -> defaults */ }
  return d
}

/**
 * Apply an imported setup document to the live form. Accepts what
 * `GET /api/sessions/{id}/setup` writes ({format, version, config}), the
 * same blob as carried in an exported mp4's `comment` tag ({generator,
 * config, param_log, export}), and a bare config object. Throws an Error
 * whose message is meant to be shown to the user.
 */
export function importConfig(target: SimConfig, doc: unknown) {
  if (!doc || typeof doc !== 'object')
    throw new Error('not a wignerf setup file')
  const d = doc as Record<string, any>
  const cfg = (d.config && typeof d.config === 'object') ? d.config : d
  if (!cfg.grid || typeof cfg.grid !== 'object' || typeof cfg.potential !== 'string'
      || !cfg.ic || !Array.isArray(cfg.ic.components))
    throw new Error('not a wignerf setup file (no grid/potential/IC)')
  for (const k of ['x1', 'x2', 'p1', 'p2', 'Nx', 'Np'] as const)
    if (typeof cfg.grid[k] !== 'number' || !Number.isFinite(cfg.grid[k]))
      throw new Error(`grid.${k} is missing or not a number`)
  // the API enforces this too, but a clear message here beats a 422 after
  // the user presses Restart
  for (const k of ['Nx', 'Np'] as const)
    if (cfg.grid[k] % 2 !== 0) throw new Error(`grid.${k} must be even`)
  if (!cfg.ic.components.length) throw new Error('the IC has no components')
  if (cfg.ic.type !== 'mixture' && cfg.ic.type !== 'cat')
    throw new Error(`unknown IC type "${cfg.ic.type}"`)
  if (Array.isArray(cfg.variants)
      && !cfg.variants.some((v: string) => (ALL_KEYS as readonly string[]).includes(v)))
    throw new Error('no known variants in the file')
  mergeConfig(target, cfg)
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
