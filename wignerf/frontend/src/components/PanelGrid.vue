<script setup lang="ts">
import { computed } from 'vue'
import type { Frame } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import { VARIANT_META, type VariantKey } from '../lib/variants'
import WignerPanel from './WignerPanel.vue'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]   // in bundle order (= session config order)
  domain: GridCfg
  showGrid: boolean
}>()

const gridClass = computed(() => {
  const n = props.variants.length
  if (n <= 1) return 'grid-cols-1 grid-rows-1'
  if (n === 2) return 'grid-cols-2 grid-rows-1'
  return 'grid-cols-2 grid-rows-2'
})
</script>

<template>
  <div class="grid gap-1 w-full h-full min-h-0" :class="gridClass">
    <div v-for="(v, i) in variants" :key="v"
         class="min-h-0 border rounded overflow-hidden"
         :style="{ borderColor: VARIANT_META[v].color + '66' }">
      <WignerPanel :frame-source="frameSource" :variant-index="i"
                   :label="VARIANT_META[v].label"
                   :domain="domain" :show-grid="showGrid" />
    </div>
  </div>
</template>
