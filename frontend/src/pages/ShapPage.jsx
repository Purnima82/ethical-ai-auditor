import React from 'react'
import { PageHeader, SectionLabel, Card } from '../components/UI'

const SHAP_DATA = [
  { feature: 'years_experience', val: 0.38,  proxy: false, proxFor: [] },
  { feature: 'education_level',  val: 0.29,  proxy: false, proxFor: [] },
  { feature: 'zip_code',         val: -0.22, proxy: true,  proxFor: ['race','socioeconomic'] },
  { feature: 'skills_match',     val: 0.18,  proxy: false, proxFor: [] },
  { feature: 'school_name',      val: -0.14, proxy: true,  proxFor: ['race','socioeconomic'] },
  { feature: 'prior_salary',     val: -0.11, proxy: true,  proxFor: ['gender','race'] },
  { feature: 'interview_score',  val: 0.09,  proxy: false, proxFor: [] },
  { feature: 'certifications',   val: 0.07,  proxy: false, proxFor: [] },
]

const GROUP_SHAP = {
  Male:   [{ f: 'years_experience', v: 0.42 }, { f: 'skills_match', v: 0.21 }, { f: 'education_level', v: 0.19 }],
  Female: [{ f: 'education_level', v: 0.38 }, { f: 'years_experience', v: 0.31 }, { f: 'prior_salary', v: -0.25 }],
  'Non-binary': [{ f: 'zip_code', v: -0.29 }, { f: 'school_name', v: -0.21 }, { f: 'skills_match', v: 0.18 }],
}

const maxAbs = Math.max(...SHAP_DATA.map(d => Math.abs(d.val)))

export default function ShapPage() {
  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        icon="◉"
        title="SHAP Explainability"
        subtitle="Feature importance + proxy variable detection · TreeSHAP · Fidelity 94%"
        badge="3 PROXY VARS FOUND"
      />

      <div style={{ padding: '24px 32px' }}>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 20, marginBottom: 20, fontSize: 12, color: 'var(--muted)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: '#378add', display: 'inline-block' }} />
            Positive impact (increases approval)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: '#d85a30', display: 'inline-block' }} />
            Negative impact (decreases approval)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: 'rgba(224,91,139,0.3)', border: '1px solid var(--accent2)', display: 'inline-block' }} />
            Proxy variable for protected attribute
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16, marginBottom: 24 }}>
          {/* SHAP waterfall */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Global SHAP feature importance</div>
            {SHAP_DATA.map((d, i) => {
              const pct = Math.round((Math.abs(d.val) / maxAbs) * 100)
              const isPos = d.val >= 0
              const barColor = d.proxy ? '#d85a30' : (isPos ? '#378add' : '#d85a30')
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <div style={{ minWidth: 130, fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    {d.feature}
                    {d.proxy && (
                      <span style={{
                        fontSize: 9, padding: '1px 6px', borderRadius: 4,
                        background: 'rgba(224,91,139,0.15)', color: 'var(--accent2)',
                        border: '1px solid rgba(224,91,139,0.25)', fontFamily: 'monospace',
                        whiteSpace: 'nowrap',
                      }}>
                        proxy
                      </span>
                    )}
                  </div>
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', height: 16, position: 'relative' }}>
                    {isPos ? (
                      <div className="bar-animate" style={{ height: 10, borderRadius: '0 3px 3px 0', background: barColor, width: `${pct}%` }} />
                    ) : (
                      <>
                        <div style={{ flex: 1 }} />
                        <div className="bar-animate" style={{ height: 10, borderRadius: '3px 0 0 3px', background: barColor, width: `${pct}%` }} />
                      </>
                    )}
                  </div>
                  <div style={{
                    fontSize: 11, fontWeight: 600, minWidth: 44, textAlign: 'right', fontFamily: 'monospace',
                    color: isPos ? '#185fa5' : '#993c1d',
                  }}>
                    {isPos ? '+' : ''}{d.val.toFixed(2)}
                  </div>
                </div>
              )
            })}
          </Card>

          {/* Proxy breakdown */}
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Proxy variable analysis</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 16, lineHeight: 1.6 }}>
              Features that correlate with protected attributes despite not being explicit demographic data.
            </div>
            {SHAP_DATA.filter(d => d.proxy).map((d, i) => (
              <div key={i} style={{
                background: 'rgba(224,91,139,0.05)', border: '1px solid rgba(224,91,139,0.15)',
                borderRadius: 10, padding: '12px 14px', marginBottom: 10,
              }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent2)', marginBottom: 4 }}>
                  {d.feature}
                </div>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>
                  SHAP value: <span style={{ color: '#993c1d', fontFamily: 'monospace' }}>{d.val.toFixed(2)}</span>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {d.proxFor.map(p => (
                    <span key={p} style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 10,
                      background: 'var(--surface2)', color: 'var(--muted)',
                      fontFamily: 'monospace',
                    }}>
                      proxies: {p}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            <div style={{
              background: 'rgba(62,207,176,0.07)', border: '1px solid rgba(62,207,176,0.15)',
              borderRadius: 10, padding: '12px 14px', marginTop: 8,
            }}>
              <div style={{ fontSize: 11, color: 'var(--accent3)', fontWeight: 600, marginBottom: 4 }}>
                Mitigation applied
              </div>
              <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}>
                All 3 proxy variables removed from the retraining pipeline. Combined with data reweighting, demographic parity improved from 0.56 → 0.74.
              </div>
            </div>
          </Card>
        </div>

        {/* Group-level SHAP comparison */}
        <SectionLabel>Group-level SHAP comparison — gender</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 12 }}>
          {Object.entries(GROUP_SHAP).map(([group, features]) => (
            <Card key={group}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 12 }}>{group}</div>
              {features.map((f, i) => {
                const isPos = f.v >= 0
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--muted)', minWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {f.f}
                    </div>
                    <div style={{ flex: 1, height: 8, background: 'var(--surface2)', borderRadius: 4, overflow: 'hidden' }}>
                      <div className="bar-animate" style={{
                        height: '100%', borderRadius: 4,
                        background: isPos ? '#378add' : '#d85a30',
                        width: `${Math.abs(f.v) * 100}%`,
                      }} />
                    </div>
                    <span style={{ fontSize: 11, fontFamily: 'monospace', color: isPos ? '#185fa5' : '#993c1d', minWidth: 36, textAlign: 'right' }}>
                      {isPos ? '+' : ''}{f.v.toFixed(2)}
                    </span>
                  </div>
                )
              })}
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
