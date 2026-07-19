/** What the transport button would do right now — shared by ControlBar
 *  (label + Solve gating) and the Space shortcut so they can never
 *  disagree. 'solve' = computes new records, 'play' = pure playback. */

import type { SessionStatus } from '../composables/useSession'

export type TransportAction = 'pause' | 'play' | 'solve'

export function transportAction(
  st: SessionStatus | null,
  lastRecord: number | null,
): TransportAction {
  if (st?.running) return 'pause'
  if (!st) return 'solve'
  const last = st.record_extent?.[1] ?? -1
  if (last < 0) return 'solve'
  if (st.mode === 'runahead') {
    const tEnd = st.t_extent?.[1]
    const done = st.t2 != null && tEnd != null &&
      (st.sign > 0 ? tEnd >= st.t2 - 1e-9 : tEnd <= st.t2 + 1e-9)
    return done ? 'play' : 'solve'
  }
  const cur = lastRecord ?? st.cursor
  return cur < last ? 'play' : 'solve'
}
