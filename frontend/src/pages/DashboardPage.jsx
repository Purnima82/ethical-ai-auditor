import React from 'react'
import { Link } from 'react-router-dom'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { PageHeader, SectionLabel, Card } from '../components/UI'

const RADAR_DATA = [
  { metric: 'Bias Risk',       before: 56, after: 78 },
  { metric: 'Transparency',    before: 49, after: 74 },
  { metric: 'Fairness',        before: 51, after: 81 },
  { metric: 'Accountability',  before: 60, after: 77 },
  { metric: 'Robustness',      before: 65, after: 80 },
  { metric: 'Data Ethics',     before: 53, after: 76 },
]

const IMPROVEMENT_DATA = [
  { attr: 'Gender',       before: 56, after: 74, delta: '+32%' },
  { attr: 'Race/Ethnicity', before: 51, after: 69, delta: '+35%' },
  { attr: 'Age Group',    before: 62, after: 78, delta: '+26%' },
]

const STATS = [
  { label: 'Overall Fairness Score', value: '74', unit: '/100', color: 'var(--accent3)', delta: '+20% after mitigation' },
  { label: 'Proxy Variables Found',  value: '3',  unit: ' features', color: 'var(--accent2)', delta: 'zip_code, school, prior_salary' },
  { label: 'Attributes Audited',     value: '3',  unit: ' protected', color: 'var(--accent)', delta: 'gender · race · age' },
  { label: 'Accuracy Retained',      value: '87.3', unit: '%', color: 'var(--amber)', delta: '+2.1% vs baseline' },
]

const QUICK_ACTIONS = [
  { to: '/audit',    icon: '⚖', label: 'Run New Audit',         desc: 'Audit any AI system with Claude AI' },
  { to: '/fairness', icon: '◈', label: 'Fairness Metrics',      desc: 'Demographic parity & equalized odds' },
  { to: '/shap',     icon: '◉', label: 'SHAP Explainability',   desc: 'Feature importance & proxy detection' },
  { to: '/reports',  icon: '▤', label: 'Generate Report',       desc: 'EU AI Act & NIST RMF compliance' },
]

export default function DashboardPage() {
  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        icon="⊞"
        title="Bias Detection Dashboard"
        subtitle="Hiring model · 12,400 samples · 3 protected attributes"
        badge="LIVE DEMO"
      />

      <div style={{ padding: '24px 32px' }}>

        {/* KPI Stats */}
        <SectionLabel>Key metrics — after mitigation</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: 12, marginBottom: 28 }}>
          {STATS.map(s => (
            <div key={s.label} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '16px' }}>
              <div style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.6px' }}>{s.label}</div>
              <div style={{ fontSize: 30, fontWeight: 800, color: s.color, lineHeight: 1 }}>
                {s.value}<span style={{ fontSize: 13, fontWeight: 400, color: 'var(--muted)' }}>{s.unit}</span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 5 }}>{s.delta}</div>
            </div>
          ))}
        </div>

        {/* Charts row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 28 }}>
          {/* Radar */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Ethics radar — before vs after</div>
            <div style={{ display: 'flex', gap: 16, marginBottom: 10, fontSize: 11, color: 'var(--muted)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--accent2)', display: 'inline-block' }} />Before
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--accent3)', display: 'inline-block' }} />After
              </span>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="rgba(120,120,200,0.15)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#888898', fontSize: 11 }} />
                <Radar name="Before" dataKey="before" stroke="var(--accent2)" fill="var(--accent2)" fillOpacity={0.1} />
                <Radar name="After"  dataKey="after"  stroke="var(--accent3)" fill="var(--accent3)" fillOpacity={0.2} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar improvement */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Fairness improvement by attribute</div>
            <div style={{ display: 'flex', gap: 16, marginBottom: 10, fontSize: 11, color: 'var(--muted)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f09595', display: 'inline-block' }} />Before
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--accent3)', display: 'inline-block' }} />After
              </span>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={IMPROVEMENT_DATA} barGap={4}>
                <XAxis dataKey="attr" tick={{ fill: '#888898', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#888898', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', fontSize: 12 }}
                  cursor={{ fill: 'rgba(120,120,200,0.05)' }}
                />
                <Bar dataKey="before" name="Before" fill="#f09595" radius={[3,3,0,0]} />
                <Bar dataKey="after"  name="After"  fill="#3ecfb0" radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Quick actions */}
        <SectionLabel>Quick actions</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: 12 }}>
          {QUICK_ACTIONS.map(q => (
            <Link key={q.to} to={q.to} style={{ textDecoration: 'none' }}>
              <div style={{
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: 12, padding: '16px', cursor: 'pointer',
                transition: 'border-color 0.2s',
              }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(124,109,240,0.4)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
              >
                <div style={{ fontSize: 22, marginBottom: 8 }}>{q.icon}</div>
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>{q.label}</div>
                <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>{q.desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
