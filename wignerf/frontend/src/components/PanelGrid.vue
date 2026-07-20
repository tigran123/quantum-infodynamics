<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import { VARIANT_META, type VariantKey } from '../lib/variants'
import { createViewWindow } from '../lib/viewWindow'
import WignerPanel from './WignerPanel.vue'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]   // in bundle order (= session config order)
  domain: GridCfg
  showGrid: boolean
}>()

// Zoom/pan coupling: decoupled by default (each panel its own window);
// the "link zoom" toggle drives all panels from one shared window.
const LINK_KEY = 'wignerf.linkZoom'
const linked = ref(localStorage.getItem(LINK_KEY) === '1')
watch(linked, (v) => localStorage.setItem(LINK_KEY, v ? '1' : '0'))

const views = [createViewWindow(), createViewWindow(),
               createViewWindow(), createViewWindow()]
const shared = createViewWindow()
// coupling adopts the window of the panel the user last zoomed/panned
let lastTouched = 0
views.forEach((v, i) => watch(v, () => { lastTouched = i }))

function toggleLink() {
  if (!linked.value) {
    Object.assign(shared, views[lastTouched])
  } else {
    // decouple in place: every panel keeps the current shared window
    for (const v of views) Object.assign(v, shared)
  }
  linked.value = !linked.value
}

const gridClass = computed(() => {
  const n = props.variants.length
  if (n <= 1) return 'grid-cols-1 grid-rows-1'
  if (n === 2) return 'grid-cols-2 grid-rows-1'
  return 'grid-cols-2 grid-rows-2'
})
</script>

<template>
  <div class="relative w-full h-full min-h-0">
    <div class="grid gap-1 w-full h-full min-h-0" :class="gridClass">
      <div v-for="(v, i) in variants" :key="v"
           class="min-h-0 border rounded overflow-hidden"
           :style="{ borderColor: VARIANT_META[v].color + '66' }">
        <WignerPanel :frame-source="frameSource" :variant-index="i"
                     :label="VARIANT_META[v].label"
                     :domain="domain" :show-grid="showGrid"
                     :view="linked ? shared : views[i]" />
      </div>
    </div>
    <label v-if="variants.length > 1"
           class="absolute top-1 right-2 z-10 flex items-center gap-1 text-xs
                  text-white bg-black/75 px-1.5 py-0.5 rounded cursor-pointer select-none"
           title="couple zoom/pan across all panels (coupling adopts the last-zoomed panel's view)">
      <input type="checkbox" :checked="linked" @change="toggleLink" />
      <span>link zoom</span>
    </label>
  </div>
</template>
