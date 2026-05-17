/**
 * useAudit.js
 * Custom React hook managing the full audit lifecycle:
 * loading state, rotating messages, result caching, and error handling.
 */
import { useState, useCallback, useRef } from 'react'
import { runAudit } from '../utils/api'

const LOADING_MESSAGES = [
  'Scanning for demographic bias patterns...',
  'Evaluating fairness metrics across groups...',
  'Cross-referencing with EU AI Act 2024...',
  'Computing SHAP proxy variable signatures...',
  'Calibrating equalized odds thresholds...',
  'Running NIST AI RMF compliance checks...',
  'Analysing disparate impact ratios...',
  'Checking IEEE 7000 value alignment...',
]

/**
 * @returns {{
 *   result: object|null,
 *   loading: boolean,
 *   error: string|null,
 *   loadingMessage: string,
 *   history: object[],
 *   submit: (payload: object) => Promise<void>,
 *   reset: () => void,
 *   clearError: () => void,
 * }}
 */
export function useAudit() {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [loadingMessage, setLoadingMessage] = useState(LOADING_MESSAGES[0])
  const [history, setHistory] = useState([])
  const intervalRef = useRef(null)

  const startMessageRotation = useCallback(() => {
    let idx = 0
    intervalRef.current = setInterval(() => {
      idx = (idx + 1) % LOADING_MESSAGES.length
      setLoadingMessage(LOADING_MESSAGES[idx])
    }, 1800)
  }, [])

  const stopMessageRotation = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  const submit = useCallback(async (payload) => {
    setError(null)
    setResult(null)
    setLoading(true)
    setLoadingMessage(LOADING_MESSAGES[0])
    startMessageRotation()

    try {
      const { data } = await runAudit(payload)
      setResult(data)
      setHistory(prev => [
        { ...data, submittedAt: new Date().toISOString(), description: payload.description },
        ...prev.slice(0, 9),   // keep last 10
      ])
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'Audit failed. Check your ANTHROPIC_API_KEY in .env'
      setError(msg)
    } finally {
      stopMessageRotation()
      setLoading(false)
    }
  }, [startMessageRotation, stopMessageRotation])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  const clearError = useCallback(() => setError(null), [])

  return { result, loading, error, loadingMessage, history, submit, reset, clearError }
}
