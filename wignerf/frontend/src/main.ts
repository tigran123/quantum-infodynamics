import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { measureDisplayInterval } from './lib/perf'
import './style.css'

measureDisplayInterval()
createApp(App).use(router).mount('#app')
