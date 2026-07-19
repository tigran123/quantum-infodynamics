<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import type { Frame } from '../lib/protocol'
import type { SessionStatus } from '../composables/useSession'
import { fmtEnergy, fmtTime } from '../lib/units'

const props = defineProps<{
  status: SessionStatus | null
  lastFrame: Frame | null
}>()

const emit = defineEmits<{
  (e: 'command', cmd: Record<string, unknown>): void
}>()

const running = computed(() => props.status?.running ?? false)
const rate = computed(() => props.status?.rate ?? 1.0)

/**
 * Solve / Play / Pause — the label tells you IN ADVANCE what the button
 * will do: "Solve" = pressing it computes new records (GPU/CPU work);
 * "Play" = pure playback of already-computed history; "Pause" while
 * running. Interactive: playback while the cursor is behind the frontier
 * (it rolls into solving when it catches up). Run-ahead: solving until t2
 * is reached, pure playback afterwards.
 */
const playLabel = computed(() => {
  if (running.value) return 'Pause'
  const st = props.status
  if (!st) return 'Solve'
  const last = st.record_extent?.[1] ?? -1
  if (last < 0) return 'Solve'
  if (st.mode === 'runahead') {
    const tEnd = st.t_extent?.[1]
    const done = st.t2 != null && tEnd != null &&
      (st.sign > 0 ? tEnd >= st.t2 - 1e-9 : tEnd <= st.t2 + 1e-9)
    return done ? 'Play' : 'Solve'
  }
  const cur = props.lastFrame?.record ?? st.cursor
  return cur < last ? 'Play' : 'Solve'
})

function togglePlay() {
  emit('command', { type: running.value ? 'pause' : 'play' })
}

function setRate(ev: Event) {
  const v = Number((ev.target as HTMLInputElement).value)
  emit('command', { type: 'rate', au_per_second: Math.pow(10, v) })
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
const stepInfo = computed(() =>
  (props.status?.per_variant ?? [])
    .map((v) => `${v.variant}: dt=${v.dt.toExponential(2)} @${v.steps_per_sec}/s`)
    .join('   '))
</script>

<template>
  <!-- Every readout sits in a FIXED-width box: live-updating text must
       never change the layout (page scrollbars used to flicker). -->
  <div class="flex items-center gap-4 px-3 py-2 bg-neutral-900 border-t border-neutral-800 text-sm text-neutral-200 whitespace-nowrap overflow-hidden">
    <button
      class="w-20 py-1 shrink-0 rounded font-medium"
      :class="playLabel === 'Solve' ? 'bg-pink-800 hover:bg-pink-700'
                                    : 'bg-sky-700 hover:bg-sky-600'"
      :title="playLabel === 'Solve'
        ? 'will compute new records (GPU/CPU work)'
        : playLabel === 'Play' ? 'pure playback of computed history' : ''"
      @click="togglePlay"
    >{{ playLabel }}</button>

    <label class="flex items-center gap-2 shrink-0">
      <span class="text-neutral-400">speed</span>
      <input type="range" min="-1" max="1.3" step="0.05"
             :value="Math.log10(rate)" @input="setRate" class="w-36" />
      <span class="tabular-nums w-24 truncate">{{ rate.toFixed(2) }} a.u./s</span>
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
