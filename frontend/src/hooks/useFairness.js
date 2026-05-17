/**
 * useFairness.js
 * Custom hook for fetching and managing fairness metrics + SHAP data.
 * Handles loading, error, and attribute-level selection state.
 */
import { useState, useEffect, useCallback } from 'react'
import { getDemoFairness, getDemoShap } from '../utils/api'

/**
 * @returns {{
 *   fairnessData: object|null,
 *   shapData: object|null,
 *   loading: boolean,
 *   error: string|null,
 *   selectedAttr: string,
 *   setSelectedAttr: (attr: string) => void,
 *   showAfter: boolean,
 *   toggleBeforeAfter: () => void,
 *   refresh: () => void,
 * }}
 */
export function useFairness() {
  const [fairnessData, setFairnessData] = useState(null)
  const [shapData, setShapData]         = useState(null)
  const [loading, setLoading]           = useState(false)
  const [error, setError]               = useState(null)
  const [selectedAttr, setSelectedAttr] = useState('gender')
  const [showAfter, setShowAfter]       = useState(true)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [fairRes, shapRes] = await Promise.all([
        getDemoFairness(),
        getDemoShap(),
      ])
      setFairnessData(fairRes.data)
      setShapData(shapRes.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load fairness data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const toggleBeforeAfter = useCallback(() => setShowAfter(v => !v), [])

  /**
   * Returns the parity group values for the currently selected attribute
   * and before/after toggle, falling back to empty object gracefully.
   */
  const currentGroupValues = useCallback(() => {
    if (!fairnessData?.attributes?.[selectedAttr]) return {}
    const key = showAfter ? 'after' : 'before'
    return fairnessData.attributes[selectedAttr][key]?.demographic_parity?.group_values ?? {}
  }, [fairnessData, selectedAttr, showAfter])

  return {
    fairnessData,
    shapData,
    loading,
    error,
    selectedAttr,
    setSelectedAttr,
    showAfter,
    toggleBeforeAfter,
    currentGroupValues,
    refresh: fetchAll,
  }
}
