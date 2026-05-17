import React, { useState } from 'react'
import { runAudit } from '../utils/api'
import { PageHeader, SectionLabel, ScoreCard, VerdictBadge, Card, Spinner } from '../components/UI'

const LOADING_MSGS = [
  'Scanning for demographic bias patterns...',
  'Evaluating fairness metrics across groups...',
  'Cross-referencing with EU AI Act 2024...',
  'Computing SHAP proxy variable signatures...',
  'Calibrating equalized odds thresholds...',
  'Running NIST AI RMF compliance checks...',
]

const EXAMPLES = [
  'A resume screening model trained on 10 years of hiring data that predicts candidate success based on past employee records.',
  'A loan approval system using zip code, income, employment type, and credit score to approve or deny mortgage applications.',
  'An NLP content moderation system trained primarily on English social media data to detect harmful posts across 12 languages.',
]

const SCORE_LABELS = {
  bias_risk: 'Bias Risk',
  transparency: 'Transparency',
  fairness: 'Fairness',
  accountability: 'Accountability',
  robustness: 'Robustness',
  data_ethics: 'Data Ethics',
}

const SEVERITY_COLOR = { critical: 'var(--accent2)', warning: 'var(--amber)', info: 'var(--accent3)' }
const SEVERITY_ICON  = { critical: '⚠', warning: '◆', info: '○' }

export default function AuditPage() {
  const [description, setDescription] = useState('')
  const [domain, setDomain] = useState('general')
  const [depth, setDepth] = useState('quick')
  const [loading, setLoading] = useState(false)
  const [loadMsg, setLoadMsg] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleAudit() {
    if (!description.trim() || description.trim().length < 20) {
      setError('Please provide at least 20 characters describing your AI system.')
      return
    }
    setError(null)
    setResult(null)
    setLoading(true)
    let idx = 0
    const interval = setInterval(() => {
      setLoadMsg(LOADING_MSGS[idx % LOADING_MSGS.length])
      idx++
    }, 1800)
    try {
      const { data } = await runAudit({ description, domain, depth })
      setResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Audit failed. Check your API key in .env')
    } finally {
      clearInterval(interval)
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '0 0 40px' }}>
      <PageHeader
        icon="⚖"
        title="Run Audit"
        subtitle="Describe your AI system — Claude AI analyses it for bias, fairness, and regulatory compliance"
        badge="CLAUDE POWERED"
      />

      <div style={{ padding: '24px 32px' }}>
        {/* Input */}
        <SectionLabel>AI system description</SectionLabel>
        <div style={{
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 12, padding: 16, marginBottom: 16,
        }}>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder="Describe your AI model or system in detail...

Examples:
• A hiring algorithm that ranks resumes based on historical employee data
• A loan approval model trained on credit history and zip code demographics
• A recidivism prediction model used in criminal sentencing decisions"
            style={{
              width: '100%', background: 'none', border: 'none', outline: 'none',
              color: 'var(--text)', fontFamily: 'monospace', fontSize: 13,
              lineHeight: 1.7, resize: 'none', minHeight: 130,
            }}
          />
        </div>

        {/* Example pills */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
            Try an example
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {EXAMPLES.map((ex, i) => (
              <button key={i} onClick={() => setDescription(ex)} style={{
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: 20, padding: '5px 14px', fontSize: 11,
                color: 'var(--muted)', cursor: 'pointer', fontFamily: 'monospace',
                transition: 'color 0.2s',
              }}
                onMouseEnter={e => e.target.style.color = 'var(--text)'}
                onMouseLeave={e => e.target.style.color = 'var(--muted)'}
              >
                Example {i + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 20 }}>
          {[
            { id: 'domain', label: 'Domain', value: domain, set: setDomain, opts: [
              ['general','General'],['hiring','Hiring & HR'],['finance','Finance & Credit'],
              ['healthcare','Healthcare'],['criminal-justice','Criminal Justice'],
              ['education','Education'],['content-moderation','Content Moderation'],
            ]},
            { id: 'depth', label: 'Depth', value: depth, set: setDepth, opts: [
              ['quick','Quick Scan'],['deep','Deep Audit'],['regulatory','Regulatory Check'],
            ]},
          ].map(({ id, label, value, set, opts }) => (
            <select key={id} value={value} onChange={e => set(e.target.value)} style={{
              background: 'var(--surface2)', border: '1px solid var(--border)',
              color: 'var(--text)', fontFamily: 'monospace', fontSize: 11,
              borderRadius: 20, padding: '7px 14px', outline: 'none', cursor: 'pointer',
            }}>
              {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          ))}
          <button
            onClick={handleAudit}
            disabled={loading}
            style={{
              marginLeft: 'auto',
              background: loading ? 'rgba(124,109,240,0.3)' : 'linear-gradient(135deg, #7c6df0, #e05b8b)',
              color: '#fff', border: 'none', borderRadius: 20,
              fontFamily: 'Syne, sans-serif', fontSize: 13, fontWeight: 700,
              padding: '10px 24px', cursor: loading ? 'default' : 'pointer',
              display: 'flex', alignItems: 'center', gap: 8,
            }}
          >
            {loading ? <><Spinner size={16} /> Auditing…</> : 'Run Audit →'}
          </button>
        </div>

        {/* Loading */}
        {loading && (
          <Card style={{ textAlign: 'center', padding: '28px' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 14 }}>
              <Spinner size={32} />
            </div>
            <div style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--muted)' }}>{loadMsg}</div>
          </Card>
        )}

        {/* Error */}
        {error && !loading && (
          <div style={{
            background: 'rgba(224,91,139,0.08)', border: '1px solid rgba(224,91,139,0.2)',
            borderRadius: 10, padding: '12px 16px', fontSize: 13, color: 'var(--accent2)', marginBottom: 16,
          }}>
            {error}
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="fade-up">
            {/* Overall */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 16,
              background: 'var(--surface)', border: '1px solid var(--border)',
              borderRadius: 14, padding: '20px 24px', marginBottom: 20,
            }}>
              <div>
                <div style={{ fontFamily: 'monospace', fontSize: 10, color: 'var(--muted)', letterSpacing: 1, marginBottom: 4, textTransform: 'uppercase' }}>
                  Overall Ethics Score
                </div>
                <div style={{
                  fontSize: 52, fontWeight: 800, lineHeight: 1,
                  color: result.overall_score >= 70 ? 'var(--accent3)' : result.overall_score >= 45 ? 'var(--amber)' : 'var(--accent2)',
                }}>
                  {result.overall_score}
                </div>
                <div style={{ height: 3, width: 100, background: 'var(--surface2)', borderRadius: 2, overflow: 'hidden', marginTop: 8 }}>
                  <div className="bar-animate" style={{
                    height: '100%', borderRadius: 2,
                    background: result.overall_score >= 70 ? 'var(--accent3)' : result.overall_score >= 45 ? 'var(--amber)' : 'var(--accent2)',
                    width: `${result.overall_score}%`,
                  }} />
                </div>
              </div>
              <div style={{ flex: 1, paddingLeft: 20, borderLeft: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <VerdictBadge verdict={result.verdict} />
                  <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace' }}>{result.domain}</span>
                </div>
                <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6 }}>{result.verdict_reason}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace', marginBottom: 4 }}>IMPROVEMENT</div>
                <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--accent3)' }}>+{result.improvement_pct}%</div>
                <div style={{ fontSize: 11, color: 'var(--muted)' }}>after mitigation</div>
              </div>
            </div>

            {/* Dimension scores */}
            <SectionLabel>Dimension scores</SectionLabel>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 10, marginBottom: 24 }}>
              {Object.entries(result.scores).map(([k, v]) => (
                <ScoreCard key={k} label={SCORE_LABELS[k] || k} value={v} />
              ))}
            </div>

            {/* Findings */}
            <SectionLabel>Key findings</SectionLabel>
            <Card style={{ marginBottom: 20 }}>
              {result.findings.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 12, padding: '12px 0',
                  borderBottom: i < result.findings.length - 1 ? '1px solid var(--border)' : 'none',
                }}>
                  <div style={{ color: SEVERITY_COLOR[f.severity], fontSize: 16, flexShrink: 0, marginTop: 2 }}>
                    {SEVERITY_ICON[f.severity]}
                  </div>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: SEVERITY_COLOR[f.severity], marginBottom: 3 }}>
                      {f.title}
                      {f.framework && (
                        <span style={{
                          fontSize: 9, padding: '2px 6px', borderRadius: 4,
                          background: 'var(--surface2)', color: 'var(--muted)',
                          marginLeft: 8, fontWeight: 400, fontFamily: 'monospace',
                        }}>
                          {f.framework}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>{f.detail}</div>
                  </div>
                </div>
              ))}
            </Card>

            {/* Recommendations */}
            <SectionLabel>Recommendations</SectionLabel>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 20 }}>
              {result.recommendations.map((r, i) => (
                <div key={i} style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 10, padding: '14px 16px',
                }}>
                  <div style={{ fontFamily: 'monospace', fontSize: 18, fontWeight: 700, color: 'var(--accent)', marginBottom: 4 }}>
                    R{i + 1}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>{r}</div>
                </div>
              ))}
            </div>

            {/* Regulatory flags */}
            {result.regulatory_flags?.length > 0 && (
              <>
                <SectionLabel>Regulatory flags</SectionLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {result.regulatory_flags.map((flag, i) => (
                    <div key={i} style={{
                      background: 'rgba(245,166,35,0.07)', borderLeft: '3px solid var(--amber)',
                      borderRadius: '0 8px 8px 0', padding: '10px 14px',
                      fontSize: 12, color: 'var(--muted)', lineHeight: 1.6,
                    }}>
                      {flag}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
