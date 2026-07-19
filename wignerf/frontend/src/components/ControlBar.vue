<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { SessionStatus } from '../composables/useSession'
import { displayInterval } from '../lib/perf'
import { transportAction } from '../lib/transport'
import { fmtEnergy, fmtTime } from '../lib/units'

const props = defineProps<{
  status: SessionStatus | null
  lastFrame: Frame | null
  setupValid: boolean
}>()

const emit = defineEmits<{
  (e: 'command', cmd: Record<string, unknown>): void
}>()

const running = computed(() => props.status?.running ?? false)

/**
 * Delay dial: how much time is injected between played-back frames.
 * Leftmost = "0" (the default) = one record per display refresh — the
 * fastest speed at which EVERY frame is still painted. The client sends
 * the measured refresh interval for it (the server keeps honest seconds),
 * and every position is clamped to at least that interval, so delivery
 * never outpaces painting and nothing is visually skipped. To the right,
 * DIAL_STEPS log-spaced values from 20 ms up to 1.5 s per frame.
 * Computation is NEVER paced by it (workers always run flat out), and it
 * is only settable while PAUSED: pause, change, resume. The thumb
 * position is local state (`dial`) so the 1 Hz status echo cannot yank it
 * around mid-drag; the echo re-syncs it when idle.
 */
const DELAY_MIN = 0.02
const DELAY_MAX = 1.5
const DIAL_STEPS = 45         // dial positions 0 (refresh-paced) .. DIAL_STEPS
function dialToDelay(i: number): number {
  return i <= 0 ? 0
    : DELAY_MIN*Math.pow(DELAY_MAX/DELAY_MIN, (i - 1)/(DIAL_STEPS - 1))
}
function delayToDial(d: number): number {
  if (d <= displayInterval()*1.05) return 0
  const i = 1 + (DIAL_STEPS - 1)
    * Math.log(d/DELAY_MIN) / Math.log(DELAY_MAX/DELAY_MIN)
  return Math.min(Math.max(Math.round(i), 1), DIAL_STEPS)
}
const dial = ref(delayToDial(props.status?.delay ?? 0))
let dialTouched = 0
watch(() => props.status?.delay, (d) => {
  if (d != null && performance.now() - dialTouched > 750)
    dial.value = delayToDial(d)
})
const delayLabel = computed(() => {
  const d = dialToDelay(dial.value)
  return d === 0 ? '0' : d < 1 ? `${Math.round(d*1000)} ms` : `${d.toFixed(1)} s`
})
const delayTitle = computed(() => running.value
  ? 'pause first to change the frame delay — it paces playback only '
    + '(computation always runs at full speed)'
  : 'time injected between played-back frames — 0 (default) means one '
    + 'frame per display refresh, the fastest speed at which every frame '
    + 'is still painted')

/**
 * Solve / Play / Pause — the label tells you IN ADVANCE what the button
 * will do: "Solve" = pressing it computes new records (GPU/CPU work);
 * "Play" = pure playback of already-computed history; "Pause" while
 * running. Playback-only runs auto-pause at the frontier (the backend
 * flips running off and the button becomes "Solve") — computation only
 * ever starts from an explicit Solve. Run-ahead: solving until t2 is
 * reached, pure playback afterwards. Solve is DISABLED while the setup
 * form holds invalid data (potential draft / IC preview) — computing
 * while the visible setup is broken misleads; playback stays allowed.
 */
const action = computed(() =>
  transportAction(props.status, props.lastFrame?.record ?? null))
const playLabel = computed(() =>
  action.value === 'pause' ? 'Pause' : action.value === 'play' ? 'Play' : 'Solve')
const solveBlocked = computed(() => action.value === 'solve' && !props.setupValid)

function togglePlay(ev?: Event) {
  // drop focus so a later Space is the global shortcut, never a second
  // native click on this button (double-fire made Space look erratic)
  ;(ev?.currentTarget as HTMLElement | null)?.blur()
  if (solveBlocked.value) return
  emit('command', { type: action.value === 'pause' ? 'pause' : 'play' })
}

function setDelay(ev: Event) {
  dialTouched = performance.now()
  const v = Number((ev.target as HTMLInputElement).value)
  dial.value = v
  emit('command', { type: 'delay',
                    seconds: Math.max(dialToDelay(v), displayInterval()) })
}

const t = computed(() => props.lastFrame ? fmtTime(props.lastFrame.t) : '—')

/**
 * Direct t entry: when paused in pure-playback state (button says "Play"),
 * the t readout becomes an editbox on click — clicking a timeline pixel
 * cannot reach a specific record among hundreds, but science can type.
 * The entered t seeks to the nearest record.
 */
const editingT = ref(false)
const tDraft = ref('')
const tInput = ref<HTMLInputElement | null>(null)
const canEditT = computed(() => !running.value && playLabel.value === 'Play')

function startEditT() {
  if (!canEditT.value || !props.lastFrame) return
  tDraft.value = props.lastFrame.t.toFixed(3)
  editingT.value = true
  void nextTick(() => tInput.value?.select())
}

function commitT() {
  if (!editingT.value) return
  editingT.value = false
  const st = props.status
  const tv = Number(tDraft.value)
  if (!st || !Number.isFinite(tv)) return
  const [k0, k1] = st.record_extent
  const t0 = st.t_extent?.[0]
  if (t0 == null || k1 < 0) return
  const step = st.record_dt * (st.sign || 1)
  const k = Math.min(Math.max(k0 + Math.round((tv - t0) / step), k0), k1)
  emit('command', { type: 'seek', record: k })
}
const E = computed(() => {
  const v = props.lastFrame?.variants[0]
  return v ? fmtEnergy(v.E) : '—'
})
const uncert = computed(() => {
  const v = props.lastFrame?.variants[0]
  return v ? (v.xStd * v.pStd).toFixed(4) : '—'
})
const purity = computed(() => {
  const v = props.lastFrame?.variants[0]
  return v ? v.purity.toFixed(6) : '—'
})
const stepInfo = computed(() => {
  // Tag each variant with its device only when the pool actually has
  // more than one device — single-device setups keep the compact form.
  const multi = (props.status?.devices?.length ?? 1) > 1
  return (props.status?.per_variant ?? [])
    .map((v) => `${v.variant}${multi ? `[${v.device}]` : ''}: dt=${v.dt.toExponential(2)} @${v.steps_per_sec}/s`)
    .join('   ')
})
</script>

<template>
  <!-- Every readout sits in a FIXED-width box: live-updating text must
       never change the layout (page scrollbars used to flicker). -->
  <div class="flex items-center gap-4 px-3 py-2 bg-neutral-900 border-t border-neutral-800 text-sm text-neutral-200 whitespace-nowrap overflow-hidden">
    <button
      class="w-20 py-1 shrink-0 rounded font-medium disabled:opacity-40 disabled:cursor-not-allowed"
      :class="playLabel === 'Solve' ? 'bg-pink-800 hover:bg-pink-700'
                                    : 'bg-sky-700 hover:bg-sky-600'"
      :disabled="solveBlocked"
      :title="solveBlocked
        ? 'setup is invalid — fix the potential / initial condition first'
        : playLabel === 'Solve'
          ? 'will compute new records (GPU/CPU work) — shortcut: Space'
          : playLabel === 'Play'
            ? 'pure playback of computed history — shortcut: Space'
            : 'shortcut: Space'"
      @click="togglePlay($event)"
    >{{ playLabel }}</button>

    <label class="flex items-center gap-2 shrink-0"
           :class="running ? 'opacity-40' : ''" :title="delayTitle">
      <span class="text-neutral-400">delay</span>
      <input type="range" :min="0" :max="DIAL_STEPS" step="1"
             :disabled="running" :value="dial" @input="setDelay"
             @change="(ev: Event) => (ev.target as HTMLInputElement).blur()"
             class="w-36" />
      <span class="tabular-nums w-24 truncate">{{ delayLabel }}</span>
    </label>

    <div class="tabular-nums w-60 truncate shrink-0">
      <span class="text-neutral-400">t =</span>
      <input v-if="editingT" ref="tInput" v-model="tDraft"
             class="w-28 bg-neutral-800 border border-sky-600 rounded px-1 tabular-nums"
             @keydown.enter="commitT" @keydown.esc="editingT = false"
             @blur="commitT" />
      <span v-else
            :class="canEditT ? 'cursor-pointer underline decoration-dotted decoration-neutral-600' : ''"
            :title="canEditT ? 'click to type t directly (seeks to the nearest record)' : ''"
            @click="startEditT">{{ t }}</span>
    </div>
    <div class="tabular-nums w-64 truncate shrink-0"><span class="text-neutral-400">E =</span> {{ E }}</div>
    <div class="tabular-nums w-36 truncate shrink-0"><span class="text-neutral-400">ΔX·ΔP =</span> {{ uncert }}</div>
    <div class="tabular-nums w-36 truncate shrink-0"
         :title="'purity of the first active variant'"><span class="text-neutral-400">γ =</span> {{ purity }}</div>
    <div class="ml-auto min-w-0 truncate text-right text-xs text-neutral-500 tabular-nums"
         :title="stepInfo">{{ stepInfo }}</div>
  </div>
</template>
