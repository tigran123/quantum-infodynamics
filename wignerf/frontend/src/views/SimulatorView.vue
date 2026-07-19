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
import ICEditor from '../components/ICEditor.vue'
import PanelGrid from '../components/PanelGrid.vue'
import PlotsColumn from '../components/PlotsColumn.vue'
import SetupPanel from '../components/SetupPanel.vue'
import Timeline from '../components/Timeline.vue'
import { useSession } from '../composables/useSession'
import { loadConfig, saveConfig } from '../lib/config'
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
    unsub = session.onFrame((f) => { currentRecord.value = f.record })
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

// Keyboard shortcuts: Space = play/pause, R = reverse time direction.
function onKey(ev: KeyboardEvent) {
  const tag = (ev.target as HTMLElement).tagName
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
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

      <span v-if="restartNeeded" class="text-amber-400 text-xs">
        setup changed —
        <button class="underline" @click="restart">restart</button> to apply
      </span>
      <span v-if="reconnecting" class="ml-auto text-amber-400">
        backend disconnected — reconnecting…
      </span>
      <span v-else-if="!session.connected.value" class="ml-auto text-amber-400">connecting…</span>
    </header>

    <div v-if="createError" class="px-3 py-2 text-red-400 text-sm">{{ createError }}</div>

    <!-- ================= landscape ================= -->
    <main v-if="layout === 'landscape'" class="flex-1 min-h-0 flex gap-2 p-2">
      <aside v-if="showSetup" class="w-80 shrink-0 overflow-y-auto space-y-4 pr-1 text-sm">
        <SetupPanel :cfg="cfg" :live="session.connected.value" :sign="session.status.value?.sign ?? 1" v-model:show-grid="showGrid"
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
          <SetupPanel :cfg="cfg" :live="session.connected.value" :sign="session.status.value?.sign ?? 1" v-model:show-grid="showGrid"
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
         class="px-3 py-1 text-xs text-red-400">{{ e }}</div>

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
