import { createRouter, createWebHashHistory } from 'vue-router'
import SimulatorView from './views/SimulatorView.vue'

// Hash routing so the statically-served SPA needs no fallback handler
// (same convention as urantia-library).
export default createRouter({
  history: createWebHashHistory(),
  routes: [{ path: '/', name: 'simulator', component: SimulatorView }],
})
