/** Variant keys (the four toggles) and their display metadata. */

export type VariantKey = 'qn' | 'qr' | 'cn' | 'cr'

export const ALL_VARIANTS: VariantKey[] = ['qn', 'qr', 'cn', 'cr']

// Distinct dash patterns: variants frequently produce IDENTICAL curves
// (harmonic: quantum == classical exactly; c = 137: rel ~ nonrel), and the
// last-drawn series would otherwise hide the rest. Later series are drawn
// on top, so their gaps let the earlier (longer-dashed/solid) show through.
export const VARIANT_META: Record<VariantKey,
  { label: string; color: string; dash: number[] }> = {
  qn: { label: 'Quantum, non-relativistic', color: '#38bdf8', dash: [] }, // sky, solid
  qr: { label: 'Quantum, relativistic', color: '#a78bfa', dash: [12, 7] }, // violet
  cn: { label: 'Classical, non-relativistic', color: '#fbbf24', dash: [6, 6] }, // amber
  cr: { label: 'Classical, relativistic', color: '#34d399', dash: [2, 6] }, // emerald
}

/** vid bitfield (bit0 quantum, bit1 relativistic) -> key. */
export function keyOfVid(vid: number): VariantKey {
  return ((vid & 1 ? 'q' : 'c') + (vid & 2 ? 'r' : 'n')) as VariantKey
}
