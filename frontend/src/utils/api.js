import axios from 'axios'

// Works locally (proxied by Vite) AND on Vercel (uses deployed Render URL)
const BASE = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL + '/api'
  : '/api'

const api = axios.create({
  baseURL: BASE,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

export const runAudit        = (payload) => api.post('/audit/', payload)
export const getProviderInfo = ()         => api.get('/audit/provider')
export const getDemoFairness = ()         => api.get('/fairness/demo')
export const getDemoShap     = ()         => api.get('/fairness/shap/demo')
export const getSampleReport = ()         => api.get('/reports/sample')
export const getAuditHistory = ()         => api.get('/audit/history')

export default api
