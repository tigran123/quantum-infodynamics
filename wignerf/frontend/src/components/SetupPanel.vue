<script setup lang="ts">
/**
 * Setup controls: potential editor, physics parameters (live-applied),
 * grid geometry + grid-lines display toggle, run mode. The IC editor is a
 * separate component so the portrait layout can place it in its own column.
 */
import PotentialEditor from './PotentialEditor.vue'
import { resetToDefaults, type SimConfig } from '../lib/config'

const props = defineProps<{ cfg: SimConfig; live: boolean; sign?: number }>()

const showGrid = defineModel<boolean>('showGrid', { required: true })

const emit = defineEmits<{
  (e: 'restart'): void
  (e: 'dirty'): void
  (e: 'apply-live', params: Record<string, unknown>): void
  (e: 'potential-validity', valid: boolean): void
}>()

/** Restore the persisted setup (grid, potential, physics, run mode, IC,
 *  variants) to defaults; display prefs (layout, grid lines) are separate
 *  localStorage keys and stay untouched. */
function resetSetup() {
  if (!confirm('Reset the ENTIRE setup (grid, potential, physics, run mode, IC, variants) to defaults?')) return
  resetToDefaults(props.cfg)
  emit('dirty')
}
</script>

<template>
  <div class="space-y-4">
    <PotentialEditor
      v-model="props.cfg.potential"
      :grid="props.cfg.grid" :hbar-eff="props.cfg.hbar_eff"
      :live="live" :variants="props.cfg.variants"
      @update:model-value="emit('dirty')"
      @apply-live="(expr) => emit('apply-live', { U: expr })"
      @validity="(v) => emit('potential-validity', v)"
      @grid-dirty="emit('dirty')"
    />

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Physics</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500" title="rest mass, mₑ = 1">m</span>
          <input v-model.number="props.cfg.mass" type="number" step="any" min="0"
                 class="wf-num" @change="emit('apply-live', { mass: props.cfg.mass })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="speed of light: 137.036 physical, 1 = old toy runs">c</span>
          <input v-model.number="props.cfg.c" type="number" step="any" min="0.1"
                 class="wf-num" @change="emit('apply-live', { c: props.cfg.c })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="value of ℏ in the evolution equations (a.u.: physical value 1); dial it below 1 to watch the classical limit emerge">ℏ</span>
          <input v-model.number="props.cfg.hbar_eff" type="number" step="any" min="0.001"
                 class="wf-num" @change="emit('apply-live', { hbar_eff: props.cfg.hbar_eff })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500" title="adaptive-step relative tolerance">tol</span>
          <input v-model.number="props.cfg.tol" type="number" step="any" min="1e-6" max="0.5"
                 class="wf-num" @change="emit('apply-live', { tol: props.cfg.tol })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="time direction of NEWLY computed records: flips the sign of dt in the propagator at the frontier. Already-computed history is unaffected — use the timeline to move within it. Shortcut: R">t dir</span>
          <select class="wf-num" :value="(props.sign ?? 1) > 0 ? 1 : -1"
                  @change="emit('apply-live', { dt_sign: Number(($event.target as HTMLSelectElement).value) })">
            <option :value="1">forward</option>
            <option :value="-1">backward</option>
          </select>
        </label>
      </div>
      <p class="text-xs text-neutral-400">
        m, c, ℏ, tol apply live at the frontier; grid &amp; IC need a restart.
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Grid</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">x₁,x₂</span>
          <input v-model.number="props.cfg.grid.x1" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <input v-model.number="props.cfg.grid.x2" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">p₁,p₂</span>
          <input v-model.number="props.cfg.grid.p1" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <input v-model.number="props.cfg.grid.p2" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">Nx</span>
          <select v-model.number="props.cfg.grid.Nx" class="wf-num" @change="emit('dirty')">
            <option v-for="n in [256, 512, 1024, 2048]" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">Np</span>
          <select v-model.number="props.cfg.grid.Np" class="wf-num" @change="emit('dirty')">
            <option v-for="n in [256, 512, 1024, 2048]" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <label class="flex items-center gap-1 col-span-2 cursor-pointer select-none"
               title="axis grid lines on all plots, the W panels and the IC preview">
          <input type="checkbox" v-model="showGrid" />
          <span class="text-neutral-400">grid lines on plots</span>
        </label>
      </div>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Run</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1"
               title="interactive: no end time — Solve keeps computing new records until you pause. run-ahead: Solve computes at full speed until t = t₂, then pauses; the button becomes Play — pure playback of the finished history.">
          <span class="w-14 text-neutral-500">mode</span>
          <select v-model="props.cfg.mode" class="wf-num" @change="emit('dirty')">
            <option value="interactive">interactive</option>
            <option value="runahead">run-ahead</option>
          </select>
        </label>
        <label class="flex items-center gap-1" v-if="props.cfg.mode === 'runahead'">
          <span class="w-14 text-neutral-500">t₂</span>
          <input v-model.number="props.cfg.t2" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-14 text-neutral-500" title="physical time per record">Δτ rec</span>
          <input v-model.number="props.cfg.record_dt" type="number" step="any" min="0.001"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <!-- playback speed lives ONLY in the transport bar; a session
             always starts at 1.00 a.u./s -->
      </div>
      <button class="w-full py-1.5 rounded bg-sky-800 hover:bg-sky-700 font-medium"
              @click="emit('restart')">Restart session</button>
      <button class="w-full py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs text-neutral-300"
              title="restore grid, potential, physics, run mode, IC and variants to their defaults"
              @click="resetSetup">Reset setup to defaults</button>
    </section>
  </div>
</template>
