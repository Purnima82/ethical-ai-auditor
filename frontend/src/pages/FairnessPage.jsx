import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { PageHeader, SectionLabel, Card, BarRow } from '../components/UI'
import { getDemoFairness } from '../utils/api'

const ATTRS = ['gender', 'race', 'age_group']
const ATTR_LABELS = { gender: 'Gender', race: 'Race / Ethnicity', age_group: 'Age Group' }

const DEMO_PARITY = {
  gender:    { groups: ['Male','Female','Non-binary'], before: [0.71,0.49,0.44], after: [0.74,0.69,0.68] },
  race:      { groups: ['White','Black','Hispanic','Asian','Other'], before: [0.73,0.44,0.47,0.69,0.51], after: [0.74,0.63,0.65,0.74,0.67] },
  age_group: { groups: ['18-24','25-34','35-44','45-54','55+'], before: [0.55,0.72,0.75,0.63,0.48], after: [0.68,0.74,0.76,0.72,0.66] },
}

const METRICS_TABLE = [
  { metric: 'Demographic Parity',    gender: '0.74 ✓', race: '0.69 ✓', age: '0.78 ✓', threshold: '< 0.10 diff' },
  { metric: 'Equalized Odds (TPR)',  gender: '0.81 ✓', race: '0.76 ✓', age: '0.79 ✓', threshold: '< 0.10 diff' },
  { metric: 'Equalized Odds (FPR)',  gender: '0.82 ✓', race: '0.78 ✓', age: '0.81 ✓', threshold: '< 0.10 diff' },
  { metric: 'Disparate Impact',      gender: '0.93 ✓', race: '0.86 ✓', age: '0.88 ✓', threshold: '≥ 0.80' },
  { metric: 'Calibration Error',     gender: '0.03 ✓', race: '0.04 ✓', age: '0.03 ✓', threshold: '< 0.05' },
]

export default function FairnessPage() {
  const [attr, setAttr] = useState('gender')
  const [showAfter, setShowAfter] = useState(true)
  const [threshold, setThreshold] = useState(50)

  const d = DEMO_PARITY[attr]
  const vals = showAfter ? d.after : d.before
  const chartData = d.groups.map((g, i) => ({ group: g, value: vals[i], before: d.before[i], after: d.after[i] }))

  const mean = vals.reduce((a, b) => a + b, 0) / vals.length

  const tpBase = Math.round(850 * (1 - threshold * 0.006))
  const tnBase = Math.round(920 * threshold * 0.009)
  const fpBase = Math.round(300 * (1 - threshold * 0.007))
  const fnBase = Math.round(850 - tpBase)

  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        icon="◈"
        title="Fairness Metrics"
        subtitle="Demographic parity, equalized odds, disparate impact — NIST AI RMF aligned"
        badge="3 ATTRIBUTES"
      />

      <div style={{ padding: '24px 32px' }}>

        {/* Attribute tabs */}
        <SectionLabel>Protected attribute</SectionLabel>
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {ATTRS.map(a => (
            <button key={a} onClick={() => setAttr(a)} style={{
              padding: '8px 18px', borderRadius: 20, fontSize: 12, fontWeight: 600,
              border: '1px solid var(--border)', cursor: 'pointer',
              background: attr === a ? 'rgba(124,109,240,0.15)' : 'none',
              color: attr === a ? 'var(--accent)' : 'var(--muted)',
              fontFamily: 'Syne, sans-serif',
              transition: 'all 0.15s',
            }}>
              {ATTR_LABELS[a]}
            </button>
          ))}
          <button onClick={() => setShowAfter(!showAfter)} style={{
            marginLeft: 'auto', padding: '8px 18px', borderRadius: 20,
            fontSize: 11, fontFamily: 'monospace',
            border: '1px solid var(--border)', cursor: 'pointer',
            background: 'none', color: 'var(--muted)',
          }}>
            Showing: {showAfter ? 'After mitigation' : 'Before mitigation'} ⇄
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 16, marginBottom: 24 }}>
          {/* Parity bar chart */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
              Demographic parity — {ATTR_LABELS[attr]}
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 16 }}>
              Approval rate per group · dashed = overall mean ({mean.toFixed(2)}) · ideal gap &lt; 0.10
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} barSize={28}>
                <XAxis dataKey="group" tick={{ fill: '#888898', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fill: '#888898', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', fontSize: 12 }}
                  cursor={{ fill: 'rgba(120,120,200,0.05)' }}
                  formatter={v => v.toFixed(3)}
                />
                <ReferenceLine y={mean} stroke="rgba(136,136,152,0.5)" strokeDasharray="4 4" />
                <Bar dataKey="value" name="Selection Rate" radius={[4,4,0,0]}>
                  {chartData.map((entry, i) => (
                    <rect key={i} fill={entry.value >= 0.7 ? '#3ecfb0' : entry.value >= 0.55 ? '#f5a623' : '#e05b8b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Threshold simulator */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Decision threshold simulator</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
              <span style={{ fontSize: 12, color: 'var(--muted)', minWidth: 70 }}>Threshold</span>
              <input type="range" min="30" max="80" step="1" value={threshold} onChange={e => setThreshold(+e.target.value)}
                style={{ flex: 1 }} />
              <span style={{ fontSize: 13, fontWeight: 600, minWidth: 36 }}>{(threshold / 100).toFixed(2)}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { label: 'True Positive', val: tpBase, color: 'var(--accent3)' },
                { label: 'False Positive', val: fpBase, color: 'var(--amber)' },
                { label: 'False Negative', val: fnBase, color: 'var(--accent2)' },
                { label: 'True Negative', val: tnBase, color: 'var(--accent)' },
              ].map(({ label, val, color }) => (
                <div key={label} style={{ background: 'var(--surface2)', borderRadius: 8, padding: '12px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color }}>{val}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 10, lineHeight: 1.6 }}>
              Adjust threshold to explore the fairness-accuracy tradeoff per group.
            </div>
          </Card>
        </div>

        {/* Before vs after bars */}
        <SectionLabel>Before vs after mitigation — {ATTR_LABELS[attr]}</SectionLabel>
        <Card style={{ marginBottom: 24 }}>
          {d.groups.map((g, i) => (
            <div key={g} style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>{g}</div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)', minWidth: 40 }}>Before</span>
                <div style={{ flex: 1, height: 7, background: 'var(--surface2)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ height: '100%', background: '#f09595', borderRadius: 4, width: `${d.before[i] * 100}%` }} />
                </div>
                <span style={{ fontSize: 11, minWidth: 34, fontWeight: 600, color: '#f09595' }}>{d.before[i].toFixed(2)}</span>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 4 }}>
                <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)', minWidth: 40 }}>After</span>
                <div style={{ flex: 1, height: 7, background: 'var(--surface2)', borderRadius: 4, overflow: 'hidden' }}>
                  <div className="bar-animate" style={{ height: '100%', background: 'var(--accent3)', borderRadius: 4, width: `${d.after[i] * 100}%` }} />
                </div>
                <span style={{ fontSize: 11, minWidth: 34, fontWeight: 600, color: 'var(--accent3)' }}>{d.after[i].toFixed(2)}</span>
              </div>
            </div>
          ))}
        </Card>

        {/* Metrics table */}
        <SectionLabel>Full metrics summary — all attributes</SectionLabel>
        <Card>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Metric','Gender','Race/Ethnicity','Age Group','NIST Threshold'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--muted)', fontWeight: 500, fontSize: 11, fontFamily: 'monospace', letterSpacing: '0.5px' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS_TABLE.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 600 }}>{row.metric}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--accent3)', fontFamily: 'monospace' }}>{row.gender}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--accent3)', fontFamily: 'monospace' }}>{row.race}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--accent3)', fontFamily: 'monospace' }}>{row.age}</td>
                  <td style={{ padding: '10px 12px', color: 'var(--muted)', fontFamily: 'monospace' }}>{row.threshold}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  )
}
