import React from 'react'
import { PageHeader, SectionLabel, Card, VerdictBadge } from '../components/UI'

const FRAMEWORKS = [
  {
    name: 'EU AI Act (2024)',
    status: 'HIGH RISK',
    color: 'var(--amber)',
    items: [
      'System classified as high-risk (employment decisions)',
      'Conformity assessment required before deployment',
      'Human oversight mechanism must be implemented',
      'Registration in EU database required',
    ],
  },
  {
    name: 'NIST AI RMF',
    status: 'PARTIAL',
    color: 'var(--accent)',
    items: [
      'GOVERN: Risk policy documented ✓',
      'MAP: Context and risk identified ✓',
      'MEASURE: Bias metrics computed ✓',
      'MANAGE: Mitigation plan in progress',
    ],
  },
  {
    name: 'IEEE 7000',
    status: 'IN REVIEW',
    color: 'var(--accent3)',
    items: [
      'Value alignment analysis complete ✓',
      'Stakeholder impact assessment pending',
      'Transparency documentation drafted ✓',
      'Accountability chain defined ✓',
    ],
  },
]

const MITIGATION_STEPS = [
  { step: '01', title: 'SHAP proxy detection', detail: 'Identified zip_code, school_name, prior_salary as proxies for race/gender via both pattern matching and Cramér\'s V correlation analysis.', status: 'complete' },
  { step: '02', title: 'Proxy variable removal', detail: 'Dropped all 3 proxy features from training pipeline. Rebuilt feature matrix with 5 remaining legitimate predictors.', status: 'complete' },
  { step: '03', title: 'Data reweighting', detail: 'Applied Kamiran & Calders (2012) reweighting to balance (class × group) cells. Combined weights across gender, race, and age attributes.', status: 'complete' },
  { step: '04', title: 'Model retraining', detail: 'Retrained XGBoost classifier with sample_weight applied. Hyperparameters unchanged to isolate fairness effect.', status: 'complete' },
  { step: '05', title: 'Threshold calibration', detail: 'Applied per-group threshold calibration to equalise FPR across demographic groups. Gender: 0.47/0.53, Race: 0.43–0.51, Age: 0.44–0.56.', status: 'complete' },
  { step: '06', title: 'Post-mitigation audit', detail: 'Re-ran full BiasDetector audit. Demographic parity improved from 0.56 → 0.74 (+20%). Accuracy: 85.2% → 87.3%.', status: 'complete' },
]

export default function ReportsPage() {
  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        icon="▤"
        title="Audit Report"
        subtitle="Hiring Classifier v2.1 · 12,400 samples · Audit ID: rpt-hiring-001"
        badge="EXPORT READY"
      />

      <div style={{ padding: '24px 32px' }}>

        {/* Summary bar */}
        <div style={{
          display: 'flex', gap: 16, alignItems: 'center',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 14, padding: '18px 24px', marginBottom: 24,
        }}>
          <div>
            <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)', letterSpacing: 1, marginBottom: 4, textTransform: 'uppercase' }}>
              Final Verdict
            </div>
            <VerdictBadge verdict="CAUTION" />
          </div>
          <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 20, flex: 1 }}>
            <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
              The model exhibits measurable bias driven by proxy variables (zip_code, prior_salary, school_name).
              After applying reweighting + proxy removal + threshold calibration, demographic parity improved
              by <strong style={{ color: 'var(--accent3)' }}>+20%</strong> across all 3 protected attributes
              while retaining <strong style={{ color: 'var(--accent3)' }}>87.3% accuracy</strong>.
            </div>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)', marginBottom: 4 }}>OVERALL SCORE</div>
            <div style={{ fontSize: 40, fontWeight: 800, color: 'var(--amber)', lineHeight: 1 }}>72</div>
            <div style={{ fontSize: 11, color: 'var(--muted)' }}>/100</div>
          </div>
        </div>

        {/* Regulatory alignment */}
        <SectionLabel>Regulatory alignment</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 12, marginBottom: 28 }}>
          {FRAMEWORKS.map(fw => (
            <Card key={fw.name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>{fw.name}</div>
                <span style={{
                  fontSize: 9, padding: '2px 8px', borderRadius: 10,
                  background: `${fw.color}18`, color: fw.color,
                  border: `1px solid ${fw.color}40`,
                  fontFamily: 'monospace', fontWeight: 700, letterSpacing: 0.5,
                }}>
                  {fw.status}
                </span>
              </div>
              <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                {fw.items.map((item, i) => (
                  <li key={i} style={{ fontSize: 11, color: 'var(--muted)', paddingLeft: 14, position: 'relative', marginBottom: 6, lineHeight: 1.5 }}>
                    <span style={{ position: 'absolute', left: 0, color: fw.color }}>›</span>
                    {item}
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>

        {/* Mitigation pipeline */}
        <SectionLabel>Mitigation pipeline — step by step</SectionLabel>
        <div style={{ position: 'relative', paddingLeft: 40 }}>
          <div style={{ position: 'absolute', left: 14, top: 0, bottom: 0, width: 1, background: 'var(--border)' }} />
          {MITIGATION_STEPS.map((s, i) => (
            <div key={i} style={{ position: 'relative', marginBottom: 16 }}>
              <div style={{
                position: 'absolute', left: -40, width: 26, height: 26,
                borderRadius: '50%', background: 'var(--accent3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontFamily: 'monospace', fontWeight: 700, color: '#04342c',
                border: '2px solid var(--surface)',
              }}>
                {s.step}
              </div>
              <Card style={{ padding: '14px 16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <div style={{ fontSize: 13, fontWeight: 700 }}>{s.title}</div>
                  <span style={{
                    fontSize: 9, padding: '2px 8px', borderRadius: 10,
                    background: 'rgba(62,207,176,0.12)', color: 'var(--accent3)',
                    fontFamily: 'monospace',
                  }}>
                    DONE
                  </span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>{s.detail}</div>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
