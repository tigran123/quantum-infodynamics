<script setup lang="ts">
/**
 * mp4 export of an already-computed record range.
 *
 * The backend renders the frames off its own history (routers/export.py) —
 * the browser only configures the job, watches its progress and downloads
 * the file. Export is PAUSED-only: a running session evicts old records
 * once the history cap is reached, and the whole point of the feature is
 * to film a range you have already played back and judged interesting.
 *
 * Progress arrives as 'export' events on the session WebSocket; a 1 s REST
 * poll runs alongside so a missed event can never leave the bar stuck.
 */
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { api } from '../api'
import type { ExportEvent, SessionStatus } from '../composables/useSession'
import { fmtTime } from '../lib/units'
import { VARIANT_META, type VariantKey } from '../lib/variants'

const props = defineProps<{
  status: SessionStatus | null
  sessionId: string | null
  event: ExportEvent | null
  variants: VariantKey[]
  currentRecord: number
  /** the SPA's "grid lines on plots" setting — the video follows it, on
   *  the charts AND on the W heatmaps */
  showGrid: boolean
}>()

const emit = defineEmits<{
  (e: 'command', cmd: Record<string, unknown>): void
}>()

const RESOLUTIONS = [
  { label: '1920 × 1080 (FHD)', width: 1920, height: 1080 },
  { label: '2560 × 1440 (QHD)', width: 2560, height: 1440 },
  { label: '3840 × 2160 (4K UHD)', width: 3840, height: 2160 },
]

interface ExportForm {
  k0: number
  k1: number
  stride: number
  fps: number
  // stored as the pixel width, NOT as an index into RESOLUTIONS: the list
  // changes over time and a stale index would silently pick another size
  width: number
  variants: VariantKey[]
}

const STORAGE_KEY = 'wignerf.export'

function loadForm(): ExportForm {
  const d: ExportForm = { k0: 0, k1: 0, stride: 1, fps: 30, width: 1920,
                          variants: [...props.variants] }
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (s && typeof s === 'object') {
      for (const k of ['stride', 'fps'] as const)
        if (typeof s[k] === 'number') d[k] = s[k]
      if (RESOLUTIONS.some((r) => r.width === s.width)) d.width = s.width
    }
  } catch { /* corrupted storage -> defaults */ }
  return d
}

const form = reactive(loadForm())
const resolution = computed(() =>
  RESOLUTIONS.find((r) => r.width === form.width) ?? RESOLUTIONS[0])
// the RANGE is never persisted (it belongs to this session's history), only
// the encoding preferences
watch(() => [form.stride, form.fps, form.width], () => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(
    { stride: form.stride, fps: form.fps, width: form.width }))
})

const open = ref(false)
const busy = ref(false)
const error = ref('')
const job = ref<ExportEvent | null>(null)
let poll = 0

/**
 * The retained extent, from the SERVER on demand. The streamed status is
 * pushed at ~1 Hz and lags a burst of frames, so right after a pause it can
 * still report a frontier hundreds of records old — seeding the range from
 * it would silently export less than the user computed. `GET /sessions/{id}`
 * reads history.extent() directly; the status echo only ever widens it.
 */
const liveExtent = ref<[number, number] | null>(null)
const extent = computed(() =>
  liveExtent.value ?? props.status?.record_extent ?? [-1, -1])

async function refreshExtent() {
  if (!props.sessionId) return
  try {
    const { data } = await api.get<SessionStatus>(`/sessions/${props.sessionId}`)
    liveExtent.value = data.record_extent
  } catch { /* session gone — the status fallback still applies */ }
}

watch(() => props.status?.record_extent, (e) => {
  if (e && liveExtent.value && e[1] > liveExtent.value[1]) liveExtent.value = e
})
watch(() => props.sessionId, () => {
  liveExtent.value = null
  // a restart closes the old session, and closing it deletes its exports:
  // a "ready" badge pointing at a deleted file would be a lie
  job.value = null
  clearInterval(poll)
})
const hasHistory = computed(() => extent.value[1] >= 0)
const running = computed(() => props.status?.running ?? false)
// The button stays live while the session computes: rendering itself needs
// a paused session (the backend 409s, and a running session evicts old
// records), but a DISABLED button whose only explanation is a tooltip is
// how this feature first looked broken. Opening the panel explains the
// gate, and Render pauses the run first.
const disabled = computed(() => !hasHistory.value || !props.sessionId)
const disabledWhy = computed(() =>
  !hasHistory.value ? 'nothing computed yet'
    : 'render the computed range to an mp4 video')

/** t of a record index, from the timeline's own linear mapping. */
function tOf(k: number): string {
  const st = props.status
  const t0 = st?.t_extent?.[0]
  if (!st || t0 == null) return ''
  return fmtTime(t0 + (k - extent.value[0])*st.record_dt*(st.sign || 1))
}

const frames = computed(() => {
  const n = Math.floor((form.k1 - form.k0)/Math.max(1, form.stride)) + 1
  return Math.max(0, n)
})
const duration = computed(() => frames.value/Math.max(1, form.fps))

// A range the user has not touched follows the history: pausing on their
// behalf (Render while computing) lands more records, and the export must
// cover them instead of stopping where the panel happened to open.
const rangeTouched = ref(false)

function useAll() {
  form.k0 = extent.value[0]
  form.k1 = extent.value[1]
  rangeTouched.value = false
}

function useFromHere() {
  form.k0 = Math.min(Math.max(props.currentRecord, extent.value[0]),
                     extent.value[1])
  form.k1 = extent.value[1]
  rangeTouched.value = true
}

// opening the panel seeds the range from the live extent (the previous
// session's indices mean nothing here) and drops a finished job's state
watch(open, async (v) => {
  if (!v) return
  await refreshExtent()
  useAll()
  form.variants = form.variants.filter((k) => props.variants.includes(k))
  if (!form.variants.length) form.variants = [...props.variants]
  // a FINISHED job is kept: rendering continues while the popover is
  // closed, and reopening it is how you collect the file (clearing here
  // threw away the only link to a file that still exists on the server)
  error.value = ''
})

// live progress from the session socket (same job only)
watch(() => props.event, (e) => {
  if (e && (!job.value || e.job_id === job.value.job_id)) job.value = e
})

function toggleVariant(v: VariantKey) {
  const i = form.variants.indexOf(v)
  if (i >= 0) {
    if (form.variants.length === 1) return    // the API needs >= 1
    form.variants.splice(i, 1)
  } else {
    form.variants.push(v)
    form.variants.sort((a, b) => props.variants.indexOf(a) - props.variants.indexOf(b))
  }
}

/** Pause the run and wait for the server to confirm it (in-flight records
 *  land after the command, so the frontier keeps moving for a moment). */
async function pauseAndSettle(): Promise<boolean> {
  emit('command', { type: 'pause' })
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 250))
    try {
      const { data } = await api.get<SessionStatus>(`/sessions/${props.sessionId}`)
      if (!data.running) {
        liveExtent.value = data.record_extent
        return true
      }
    } catch { return false }
  }
  return false
}

async function start() {
  if (!props.sessionId || busy.value) return
  busy.value = true
  error.value = ''
  job.value = null            // a new render supersedes the previous file
  const r = resolution.value
  try {
    if (running.value) {
      if (!(await pauseAndSettle())) {
        error.value = 'could not pause the session — try the transport button'
        return
      }
      if (!rangeTouched.value) useAll()   // cover what the pause just added
    }
    const { data } = await api.post<ExportEvent>(
      `/sessions/${props.sessionId}/export`,
      { k0: form.k0, k1: form.k1, stride: form.stride, fps: form.fps,
        width: r.width, height: r.height, variants: form.variants,
        show_grid: props.showGrid })
    job.value = data
    startPoll(data.job_id)
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: string } } }
    error.value = err.response?.data?.detail ?? String(e)
  } finally {
    busy.value = false
  }
}

function startPoll(jid: string) {
  clearInterval(poll)
  poll = window.setInterval(async () => {
    try {
      const { data } = await api.get<ExportEvent>(`/exports/${jid}`)
      job.value = data
      if (data.state !== 'running' && data.state !== 'queued') clearInterval(poll)
    } catch {
      clearInterval(poll)       // job dropped (session closed / TTL)
    }
  }, 1000)
}

async function cancel() {
  const j = job.value
  if (!j) return
  clearInterval(poll)
  try { await api.delete(`/exports/${j.job_id}`) } catch { /* already gone */ }
  job.value = null
}

/** The file lives on the server until its TTL; hand the browser a plain
 *  download link built on the API base (so an nginx prefix is honoured). */
const downloadUrl = computed(() =>
  job.value && job.value.state === 'done'
    ? `${api.defaults.baseURL}/exports/${job.value.job_id}/file`
    : '')

const pct = computed(() => {
  const j = job.value
  return j && j.total ? Math.round((100*j.done)/j.total) : 0
})

/**
 * The header button IS the notification: rendering continues while the
 * popover is closed (the poll and the WS 'export' events keep running), so
 * the button carries the progress and then the finished state — reopening
 * the panel is how you collect the file.
 */
const buttonLabel = computed(() => {
  const j = job.value
  if (!j || open.value) return '⤓ export mp4'
  if (j.state === 'running' || j.state === 'queued')
    return `⤓ export ${pct.value}%`
  if (j.state === 'done') return '⤓ export ready'
  if (j.state === 'error') return '⤓ export failed'
  return '⤓ export mp4'
})
const buttonClass = computed(() => {
  const s = open.value ? null : job.value?.state
  if (s === 'done') return 'bg-emerald-700 hover:bg-emerald-600 text-white'
  if (s === 'error') return 'bg-red-800 hover:bg-red-700 text-white'
  if (s === 'running' || s === 'queued')
    return 'bg-sky-800 hover:bg-sky-700 text-white'
  return 'bg-neutral-800 hover:bg-neutral-700'
})
const buttonTitle = computed(() => {
  const j = job.value
  if (!open.value && j?.state === 'done')
    return `${j.filename} is ready (${(j.bytes/2**20).toFixed(1)} MiB) — `
      + 'click to download; the server keeps it for 30 minutes'
  if (!open.value && j?.state === 'error')
    return `export failed: ${j.error ?? 'unknown error'}`
  if (!open.value && (j?.state === 'running' || j?.state === 'queued'))
    return `rendering ${j.done}/${j.total} frames — click to watch or cancel`
  return disabledWhy.value
})

onBeforeUnmount(() => clearInterval(poll))
</script>

<template>
  <div class="relative">
    <button class="px-2 py-0.5 rounded text-xs tabular-nums
                   disabled:opacity-40 disabled:cursor-not-allowed"
            :class="buttonClass" :disabled="disabled" :title="buttonTitle"
            @click="open = !open;
                    (($event as MouseEvent).currentTarget as HTMLElement)?.blur()">
      {{ buttonLabel }}
    </button>
    <div v-if="open" class="fixed inset-0 z-40" @click="open = false"></div>
    <div v-if="open"
         class="absolute left-0 top-full mt-2 z-50 w-[26rem] rounded border border-neutral-700
                bg-neutral-900 shadow-xl p-3 text-xs space-y-2">
      <h4 class="font-semibold text-neutral-300">Export the computed range to mp4</h4>

      <div class="grid grid-cols-[3.5rem_1fr] gap-x-2 gap-y-1.5 items-center">
        <span class="text-neutral-500">records</span>
        <div class="flex items-center gap-1">
          <input v-model.number="form.k0" type="number" class="wf-num w-20"
                 :min="extent[0]" :max="extent[1]"
                 @input="rangeTouched = true" />
          <span class="text-neutral-600">…</span>
          <input v-model.number="form.k1" type="number" class="wf-num w-20"
                 :min="extent[0]" :max="extent[1]"
                 @input="rangeTouched = true" />
          <button class="px-1.5 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 whitespace-nowrap"
                  title="the whole retained history"
                  @click="refreshExtent().then(useAll)">all</button>
          <button class="px-1.5 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 whitespace-nowrap"
                  title="from the current time position to the frontier"
                  @click="refreshExtent().then(useFromHere)">here →</button>
        </div>

        <span class="text-neutral-500">t</span>
        <div class="text-neutral-400 tabular-nums truncate"
             :title="'physical time of the first and last exported record'">
          {{ tOf(form.k0) }} → {{ tOf(form.k1) }}
        </div>

        <span class="text-neutral-500">every</span>
        <div class="flex items-center gap-1">
          <input v-model.number="form.stride" type="number" min="1" max="1000"
                 class="wf-num w-16" />
          <span class="text-neutral-400">record(s)</span>
          <span class="ml-2 text-neutral-500">fps</span>
          <input v-model.number="form.fps" type="number" min="1" max="120"
                 class="wf-num w-16" />
        </div>

        <span class="text-neutral-500">size</span>
        <select v-model.number="form.width" class="wf-num w-40"
                title="4K renders ~4× more pixels per frame — expect the
                       export to take about four times as long as FHD">
          <option v-for="r in RESOLUTIONS" :key="r.label" :value="r.width">
            {{ r.label }}
          </option>
        </select>

        <span class="text-neutral-500">panels</span>
        <div class="flex items-center gap-2">
          <label v-for="v in props.variants" :key="v"
                 class="flex items-center gap-1 cursor-pointer select-none"
                 :title="VARIANT_META[v].label">
            <input type="checkbox" :checked="form.variants.includes(v)"
                   @change="toggleVariant(v)" />
            <span :style="{ color: VARIANT_META[v].color }">{{ v.toUpperCase() }}</span>
          </label>
        </div>
      </div>

      <p class="text-neutral-400">
        {{ frames }} frames → {{ duration.toFixed(1) }} s of video, grid lines
        {{ showGrid ? 'on' : 'off' }} (follows the setup panel).
        Each frame carries the panels, the marginals, the E/ΔX·ΔP/γ series
        and the parameters + IC needed to reproduce the run.
      </p>

      <p v-if="running" class="text-amber-400">
        The session is still computing. Rendering needs a paused session —
        Render will pause it first (computation resumes with Solve).
      </p>

      <p v-if="error" class="text-red-400">{{ error }}</p>

      <div v-if="job" class="space-y-1">
        <div class="h-2 rounded bg-neutral-800 overflow-hidden">
          <div class="h-full bg-sky-600 transition-[width] duration-200"
               :style="{ width: pct + '%' }"></div>
        </div>
        <p class="text-neutral-400 tabular-nums">
          {{ job.state }} — {{ job.done }}/{{ job.total }} frames
          <span v-if="job.state === 'done'">
            ({{ (job.bytes/2**20).toFixed(1) }} MiB)
          </span>
          <span v-if="job.error" class="text-red-400">{{ job.error }}</span>
        </p>
      </div>

      <div class="flex items-center gap-2 pt-1">
        <button class="px-2 py-1 rounded bg-sky-700 hover:bg-sky-600 font-medium
                       disabled:opacity-40 disabled:cursor-not-allowed"
                :disabled="busy || frames < 1
                           || job?.state === 'running' || job?.state === 'queued'"
                @click="start">{{ running ? 'Pause &amp; render' : 'Render' }}</button>
        <a v-if="downloadUrl" :href="downloadUrl" :download="job!.filename"
           class="px-2 py-1 rounded bg-emerald-700 hover:bg-emerald-600 font-medium text-white">
          ⤓ download
        </a>
        <button v-if="job?.state === 'running' || job?.state === 'queued'"
                class="px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
                @click="cancel">Cancel</button>
        <button class="ml-auto px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
                @click="open = false">Close</button>
      </div>
    </div>
  </div>
</template>
