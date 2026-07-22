<script setup lang="ts">
/**
 * Simulator view. Two layouts (persisted):
 * - landscape: [setup+IC column] [W panel grid] [plots column]
 * - portrait (tall screens): three columns on top — setup | IC | plots —
 *   with the W panel grid full-width underneath.
 *
 * Live-applicable changes (U, c, mass, ℏ, tol, dt sign) go through
 * set_params into the running session; grid/IC/variant changes require a
 * restart (fresh session from the same-config Cauchy data).
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import ControlBar from '../components/ControlBar.vue'
import ExportPanel from '../components/ExportPanel.vue'
import ICEditor from '../components/ICEditor.vue'
import PanelGrid from '../components/PanelGrid.vue'
import PlotsColumn from '../components/PlotsColumn.vue'
import SetupPanel from '../components/SetupPanel.vue'
import Timeline from '../components/Timeline.vue'
import { useSession } from '../composables/useSession'
import { loadConfig, saveConfig } from '../lib/config'
import { displayInterval } from '../lib/perf'
import { transportAction } from '../lib/transport'
import { ALL_VARIANTS, VARIANT_META, type VariantKey } from '../lib/variants'

const session = useSession()
const currentRecord = ref(0)
const createError = ref('')
const restartCount = ref(0)
const showSetup = ref(true)
const restartNeeded = ref(false)
const reconnecting = ref(false)

// Solve gate: the transport must not start a computation while the setup
// form holds invalid data (family-invalid potential draft, IC the preview
// endpoint rejects). The editors emit their verdicts after every compile/
// preview round-trip; hiding the setup discards the (unapplied) drafts, so
// the gate reopens.
const potentialValid = ref(false)
const icValid = ref(false)
const setupValid = computed(() =>
  !showSetup.value || (potentialValid.value && icValid.value))

const layout = ref<'landscape' | 'portrait'>(
  (localStorage.getItem('wignerf.layout') as 'landscape' | 'portrait') ?? 'landscape')
const showHelp = ref(false)
watch(layout, (v) => localStorage.setItem('wignerf.layout', v))

const showGrid = ref(localStorage.getItem('wignerf.grid') !== '0')
watch(showGrid, (v) => localStorage.setItem('wignerf.grid', v ? '1' : '0'))

// setup persists across reloads: a hard refresh must not silently reset
// mode/t2/grid/IC to defaults
const cfg = reactive(loadConfig())
watch(cfg, () => saveConfig(cfg), { deep: true })
const activeVariants = ref<VariantKey[]>([...cfg.variants])
const activeGrid = ref({ ...cfg.grid })
const sessionId = computed(() => session.info.value?.session_id ?? null)
// NOTE: showGrid is deliberately NOT part of this key — toggling grid
// lines must never remount/blank the panels (charts rebuild internally)
const plotsKey = computed(() => String(restartCount.value))

let unsub: (() => void) | null = null

function payload() {
  const p: Record<string, unknown> = JSON.parse(JSON.stringify(cfg))
  if (cfg.mode === 'interactive') delete p.t2
  // delay 0 is the dial's "one frame per display refresh" position: the
  // server needs honest seconds, and a fresh session must not flood
  p.delay = Math.max(cfg.delay, displayInterval())
  return p
}

async function restart() {
  createError.value = ''
  try {
    unsub?.()
    await session.create(payload())
    activeVariants.value = [...cfg.variants]
    activeGrid.value = { ...cfg.grid }
    currentRecord.value = 0
    restartNeeded.value = false
    restartCount.value++
    unsub = session.onFrame((f) => {
      currentRecord.value = f.record
      // geometry is a per-record fact (auto-expand regrids): follow the
      // PAINTED frame so panels, axis overlays and marginal axes stay in
      // sync while scrubbing across a regrid boundary in either direction
      const g = activeGrid.value
      if (f.Nx !== g.Nx || f.Np !== g.Np || f.x1 !== g.x1 || f.x2 !== g.x2
          || f.p1 !== g.p1 || f.p2 !== g.p2)
        activeGrid.value = { x1: f.x1, x2: f.x2, Nx: f.Nx,
                             p1: f.p1, p2: f.p2, Np: f.Np }
    })
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: string } } }
    createError.value = err.response?.data?.detail ?? String(e)
  }
}

function toggleVariant(v: VariantKey) {
  const i = cfg.variants.indexOf(v)
  if (i >= 0) {
    if (cfg.variants.length === 1) return   // >=1 constraint
    cfg.variants.splice(i, 1)
  } else {
    cfg.variants.push(v)
    cfg.variants.sort((a, b) => ALL_VARIANTS.indexOf(a) - ALL_VARIANTS.indexOf(b))
  }
  // variant changes discard the computed history (a new variant has no
  // state at the current t), so they only take effect on YOUR restart —
  // never behind your back
  restartNeeded.value = true
}

function applyLive(params: Record<string, unknown>) {
  session.send({ type: 'set_params', params })
}

// Boundary watch surfacing: a dismissible amber warning while W sits in
// the edge band (the server posts an all-clear that removes it), and a
// transient notice for each auto-expand regrid.
const boundaryText = computed(() => {
  const b = session.boundary.value
  if (!b) return ''
  const axes = b.axes.join(', ')
  if (b.action === 'capped')
    return `W(x,p,t) reached the ${axes} edge but the domain is at the ` +
           `${b.max_grid ?? ''}-cell cap (WIGNERF_MAX_GRID) — mass is wrapping`
  if (b.action === 'invalid_potential')
    return b.message ?? 'cannot expand: U(x) is invalid on the larger domain'
  return `W(x,p,t) is approaching the ${axes} edge — mass will wrap around` +
         (cfg.auto_expand ? '' : '; enable auto-expand or restart with a larger domain')
})
// the server's status is the authority on the live toggle (delay-dial
// pattern): a reattach to a surviving session must not show a stale box
watch(() => session.status.value?.auto_expand, (v) => {
  if (typeof v === 'boolean' && v !== cfg.auto_expand) cfg.auto_expand = v
})
const regridFlash = ref('')
let regridTimer = 0
watch(session.regrid, (r) => {
  if (!r) return
  const g = r.grid
  const verb = Object.values(r.kind).includes('double') ? 'expanded' : 'moved'
  regridFlash.value = `domain ${verb} to [${+g.x1.toFixed(4)}, ${+g.x2.toFixed(4)}] × ` +
    `[${+g.p1.toFixed(4)}, ${+g.p2.toFixed(4)}], ${g.Nx}×${g.Np}, at record ${r.at_record}`
  clearTimeout(regridTimer)
  regridTimer = window.setTimeout(() => { regridFlash.value = '' }, 6000)
})

/**
 * Self-heal after an unexpected WebSocket close (e.g. a backend restart,
 * common with `uvicorn --reload`): keep probing until the backend answers;
 * if our session survived, reattach the socket, otherwise recreate the
 * session from the current config.
 */
let recovering = false
session.onClose(() => { void recover() })

async function recover() {
  if (recovering) return
  recovering = true
  reconnecting.value = true
  const sid = session.info.value?.session_id
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500))
      try {
        if (!sid) break
        await api.get(`/sessions/${sid}`)
        session.reconnect()          // session survived: reattach
        return
      } catch (e: unknown) {
        const st = (e as { response?: { status?: number } }).response?.status
        if (st === 404) break        // backend is up, session is gone
        // no response at all: backend still down — keep waiting
      }
    }
    await restart()                  // recreate from the current config
  } finally {
    recovering = false
    reconnecting.value = false
  }
}

// Keyboard shortcuts: Space = play/pause (documented in the transport
// button's tooltip), R = reverse time direction, and — paused only —
// ←/→ = seek ±10% of the computed history, Home/End = first record /
// frontier. BUTTON is excluded so a focused button keeps its native
// Space=click and never double-fires with this handler (transport
// controls also blur themselves after use).
function onKey(ev: KeyboardEvent) {
  // before the focus guard: Esc dismisses the help popover from anywhere
  if (ev.code === 'Escape' && showHelp.value) {
    showHelp.value = false
    return
  }
  const tag = (ev.target as HTMLElement).tagName
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA'
      || tag === 'BUTTON') return
  if (ev.code === 'Space') {
    ev.preventDefault()
    // same gate as the transport button: never start a computation
    // (action 'solve') while the setup form is invalid
    const act = transportAction(session.status.value,
                                session.lastFrame.value?.record ?? null)
    if (act === 'solve' && !setupValid.value) return
    session.send({ type: act === 'pause' ? 'pause' : 'play' })
  } else if (ev.code === 'KeyR') {
    const sign = session.status.value?.sign ?? 1
    session.send({ type: 'set_params', params: { dt_sign: sign > 0 ? -1 : 1 } })
  } else if (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight'
             || ev.code === 'Home' || ev.code === 'End') {
    const st = session.status.value
    const [k0, k1] = st?.record_extent ?? [0, -1]
    if (!st || st.running || k1 < 0) return   // paused-only, needs history
    ev.preventDefault()
    // The backend clamps every seek into the TRUE extent, and our cached
    // record_extent can be stale (pausing races the final record's
    // completion, and no status is pushed while paused) — so send the
    // unclamped target and let the backend be the authority. End uses a
    // sentinel index for "the frontier, wherever it really is".
    // No optimistic cursor state on purpose: a held key recomputes from
    // the last painted frame, so stepping is paced by frame delivery
    // instead of flying through the whole timeline between paints.
    const step = Math.max(1, Math.round((k1 - k0) / 10))
    const cur = Math.round(session.lastFrame.value?.record ?? st.cursor)
    const k = ev.code === 'Home' ? 0
            : ev.code === 'End'  ? Number.MAX_SAFE_INTEGER
            : ev.code === 'ArrowLeft' ? Math.max(0, cur - step)
            :                           cur + step
    if (k !== cur) session.send({ type: 'seek', record: k })
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKey)
  void restart()
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  unsub?.()
  void session.destroy()
})
</script>

<template>
  <div class="h-screen flex flex-col overflow-hidden bg-neutral-950 text-neutral-100">
    <header class="px-3 py-2 text-sm border-b border-neutral-800 flex items-center gap-4 flex-wrap">
      <button class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-xs"
              @click="showSetup = !showSetup">
        {{ showSetup ? '◀ hide setup' : '▶ setup' }}
      </button>
      <span class="font-semibold tracking-wide">wignerf</span>

      <div class="flex items-center gap-3">
        <label v-for="v in ALL_VARIANTS" :key="v"
               class="flex items-center gap-1 cursor-pointer select-none"
               :title="VARIANT_META[v].label">
          <input type="checkbox" :checked="cfg.variants.includes(v)"
                 @change="toggleVariant(v)" />
          <span :style="{ color: VARIANT_META[v].color }">{{ v.toUpperCase() }}</span>
        </label>
      </div>

      <button class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-xs"
              :title="'toggle landscape/portrait layout'"
              @click="layout = layout === 'landscape' ? 'portrait' : 'landscape'">
        {{ layout === 'landscape' ? '⬒ portrait' : '⬓ landscape' }}
      </button>

      <ExportPanel :status="session.status.value" :session-id="sessionId"
                   :event="session.exportEvent.value" :variants="activeVariants"
                   :current-record="currentRecord" :show-grid="showGrid"
                   @command="session.send" />

      <div class="relative">
        <button class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-xs"
                title="keyboard shortcuts and mouse controls"
                @click="showHelp = !showHelp;
                        (($event as MouseEvent).currentTarget as HTMLElement)?.blur()">? help</button>
        <div v-if="showHelp" class="fixed inset-0 z-40" @click="showHelp = false"></div>
        <div v-if="showHelp"
             class="absolute left-0 top-full mt-2 z-50 w-96 rounded border border-neutral-700 bg-neutral-900 shadow-xl p-3 text-xs space-y-2">
          <h4 class="font-semibold text-neutral-300">Keyboard</h4>
          <div class="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-neutral-300">
            <span><kbd class="wf-kbd">Space</kbd></span>
            <span>Solve / Play / Pause (the transport button)</span>
            <span><kbd class="wf-kbd">R</kbd></span>
            <span>reverse time direction (applies at the frontier)</span>
            <span><kbd class="wf-kbd">←</kbd> <kbd class="wf-kbd">→</kbd></span>
            <span>while paused: step the time position by 10% of the computed history</span>
            <span><kbd class="wf-kbd">Home</kbd> <kbd class="wf-kbd">End</kbd></span>
            <span>while paused: jump to the first record / the frontier</span>
            <span>click <span class="text-neutral-400">t =</span></span>
            <span>while paused: type an exact time — <kbd class="wf-kbd">Enter</kbd> seeks to the
              nearest record, <kbd class="wf-kbd">Esc</kbd> cancels</span>
          </div>
          <h4 class="font-semibold text-neutral-300 pt-1">Mouse (plots &amp; W panels)</h4>
          <div class="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-neutral-300">
            <span>drag</span>
            <span>charts: zoom to selection (x, y or box) · W panels &amp; IC preview: pan</span>
            <span>wheel</span>
            <span>zoom at the cursor (charts: x-axis, <kbd class="wf-kbd">Shift</kbd> = y-axis)</span>
            <span>double-click</span>
            <span>reset the view</span>
          </div>
        </div>
      </div>

      <span v-if="restartNeeded" class="text-amber-400 text-xs">
        setup changed —
        <button class="underline" @click="restart">restart</button> to apply
      </span>
      <span v-if="boundaryText" class="text-amber-400 text-xs">
        ⚠ {{ boundaryText }}
        <button class="underline" title="dismiss (reappears if the boundary state changes)"
                @click="session.boundary.value = null">×</button>
      </span>
      <span v-if="regridFlash" class="text-sky-400 text-xs">⤢ {{ regridFlash }}</span>
      <span v-if="reconnecting" class="ml-auto text-amber-400">
        backend disconnected — reconnecting…
      </span>
      <span v-else-if="!session.connected.value" class="ml-auto text-amber-400">connecting…</span>
    </header>

    <div v-if="createError" class="px-3 py-2 text-red-400 text-sm">{{ createError }}</div>

    <!-- ================= landscape ================= -->
    <main v-if="layout === 'landscape'" class="flex-1 min-h-0 flex gap-2 p-2">
      <aside v-if="showSetup" class="w-80 shrink-0 overflow-y-auto space-y-4 pr-1 text-sm">
        <SetupPanel :cfg="cfg" :live="session.connected.value" :sign="session.status.value?.sign ?? 1"
                    :live-grid="session.status.value?.grid ?? null"
                    :max-grid="session.status.value?.max_grid ?? 4096" v-model:show-grid="showGrid"
                    @dirty="restartNeeded = true" @restart="restart"
                    @apply-live="applyLive"
                    @potential-validity="(v: boolean) => potentialValid = v" />
        <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                  :show-grid="showGrid" @changed="restartNeeded = true"
                  @validity="(v: boolean) => icValid = v" />
      </aside>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :domain="activeGrid"
                   :show-grid="showGrid" />
      </div>

      <aside class="w-[380px] shrink-0 overflow-y-auto">
        <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                     :variants="activeVariants" :grid="activeGrid"
                     :last-frame="session.lastFrame.value"
                     :show-grid="showGrid" :plots-key="plotsKey" />
      </aside>
    </main>

    <!-- ================= portrait ================== -->
    <main v-else class="flex-1 min-h-0 flex flex-col gap-2 p-2">
      <div v-if="showSetup" class="grid grid-cols-3 gap-3 max-h-[46%] min-h-0 text-sm">
        <div class="overflow-y-auto min-h-0 pr-1">
          <SetupPanel :cfg="cfg" :live="session.connected.value" :sign="session.status.value?.sign ?? 1"
                    :live-grid="session.status.value?.grid ?? null"
                    :max-grid="session.status.value?.max_grid ?? 4096" v-model:show-grid="showGrid"
                      @dirty="restartNeeded = true" @restart="restart"
                      @apply-live="applyLive"
                      @potential-validity="(v: boolean) => potentialValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                    :show-grid="showGrid" @changed="restartNeeded = true"
                    @validity="(v: boolean) => icValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                       :variants="activeVariants" :grid="activeGrid"
                       :last-frame="session.lastFrame.value"
                       :show-grid="showGrid" :plots-key="plotsKey" />
        </div>
      </div>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :domain="activeGrid"
                   :show-grid="showGrid" />
      </div>
    </main>

    <div v-for="(e, i) in session.errors.value" :key="i"
         class="px-3 py-1 text-xs text-red-400">
      {{ e }}
      <button class="underline ml-2 text-red-300" title="dismiss this error"
              @click="session.errors.value.splice(i, 1)">×</button>
    </div>

    <Timeline
      :status="session.status.value"
      :current-record="currentRecord"
      @seek="(r) => session.send({ type: 'seek', record: r })"
    />
    <ControlBar
      :status="session.status.value"
      :last-frame="session.lastFrame.value"
      :setup-valid="setupValid"
      @command="session.send"
    />
  </div>
</template>
