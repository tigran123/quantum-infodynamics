/**
 * Hartree atomic units are canonical everywhere in wignerf; SI appears in
 * DISPLAY LABELS ONLY (never in computation). CODATA 2018 factors.
 */

export const AU_TIME_FS = 2.4188843265857e-2 // 1 a.u. of time in fs
export const AU_LENGTH_ANGSTROM = 0.529177210903 // 1 bohr in Angstrom
export const AU_ENERGY_EV = 27.211386245988 // 1 hartree in eV
export const C_AU = 137.035999084 // speed of light, 1/alpha

export function fmtTime(t: number): string {
  return `${t.toFixed(3)} a.u. (${(t * AU_TIME_FS).toPrecision(4)} fs)`
}

export function fmtEnergy(e: number): string {
  return `${e.toPrecision(6)} Ha (${(e * AU_ENERGY_EV).toPrecision(4)} eV)`
}

export function fmtLength(x: number): string {
  return `${x.toPrecision(4)} a₀ (${(x * AU_LENGTH_ANGSTROM).toPrecision(4)} Å)`
}
