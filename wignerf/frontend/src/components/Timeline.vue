<script setup lang="ts">
import { computed } from 'vue'
import type { SessionStatus } from '../composables/useSession'

const props = defineProps<{
  status: SessionStatus | null
  currentRecord: number
}>()

const emit = defineEmits<{
  (e: 'seek', record: number): void
}>()

const extent = computed(() => props.status?.record_extent ?? [-1, -1])
const span = computed(() => Math.max(1, extent.value[1] - extent.value[0]))
const cursorPct = computed(() => {
  if (extent.value[1] < 0) return 0
  return (100 * (props.currentRecord - extent.value[0])) / span.value
})

function click(ev: MouseEvent) {
  if (extent.value[1] < 0) return
  const el = ev.currentTarget as HTMLElement
  const frac = (ev.clientX - el.getBoundingClientRect().left) / el.clientWidth
  const rec = Math.round(extent.value[0] + frac * span.value)
  emit('seek', rec)
}
</script>

<template>
  <div
    class="relative h-4 mx-3 my-1 bg-neutral-800 rounded cursor-pointer select-none"
    title="click to seek"
    @click="click"
  >
    <!-- computed (retained) extent fill -->
    <div class="absolute inset-y-0 left-0 bg-neutral-600/60 rounded"
         :style="{ width: '100%' }" v-if="extent[1] >= 0"></div>
    <!-- cursor -->
    <div class="absolute inset-y-0 w-0.5 bg-sky-400"
         :style="{ left: cursorPct + '%' }" v-if="extent[1] >= 0"></div>
    <div class="absolute right-1 top-0 text-[10px] leading-4 text-neutral-400 tabular-nums"
         v-if="extent[1] >= 0">
      {{ currentRecord }} / [{{ extent[0] }}, {{ extent[1] }}]
    </div>
  </div>
</template>
