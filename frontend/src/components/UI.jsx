import React from 'react'

export function ScoreCard({ label, value, delta, colorFn }) {
  const color = colorFn
    ? colorFn(value)
    : value >= 70 ? 'var(--accent3)' : value >= 45 ? 'var(--amber)' : 'var(--accent2)'

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 12, padding: '14px 16px',
    }}>
      <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)', letterSpacing: '0.8px', marginBottom: 8, textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>
        {typeof value === 'number' ? Math.round(value) : value}
      </div>
      <div style={{ height: 3, background: 'var(--surface2)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
        <div className="bar-animate" style={{ height: '100%', borderRadius: 2, background: color, width: `${Math.min(100, value)}%` }} />
      </div>
      {delta && (
        <div style={{ fontSize: 11, color: 'var(--accent3)', marginTop: 5 }}>{delta}</div>
      )}
    </div>
  )
}

export function VerdictBadge({ verdict }) {
  const colors = {
    PASS:    { bg: 'rgba(62,207,176,0.12)', color: 'var(--accent3)', border: 'rgba(62,207,176,0.3)' },
    CAUTION: { bg: 'rgba(245,166,35,0.12)', color: 'var(--amber)',   border: 'rgba(245,166,35,0.3)' },
    FAIL:    { bg: 'rgba(224,91,139,0.12)', color: 'var(--accent2)', border: 'rgba(224,91,139,0.3)' },
  }
  const s = colors[verdict] || colors.CAUTION
  return (
    <span style={{
      ...s, border: `1px solid ${s.border}`,
      borderRadius: 20, fontSize: 11, padding: '3px 12px',
      fontFamily: 'monospace', fontWeight: 700, letterSpacing: 1,
    }}>
      {verdict}
    </span>
  )
}

export function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace',
      letterSpacing: '1.5px', textTransform: 'uppercase', marginBottom: 10,
    }}>
      {children}
    </div>
  )
}

export function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 14, padding: '18px 20px',
      ...style,
    }}>
      {children}
    </div>
  )
}

export function PageHeader({ icon, title, subtitle, badge }) {
  return (
    <div style={{
      padding: '28px 32px 20px',
      borderBottom: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', gap: 14,
    }}>
      <div style={{
        width: 42, height: 42, borderRadius: 12, flexShrink: 0,
        background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 20,
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-0.5px' }}>{title}</div>
        {subtitle && <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>{subtitle}</div>}
      </div>
      {badge && (
        <div style={{
          marginLeft: 'auto',
          background: 'rgba(124,109,240,0.12)',
          color: 'var(--accent)',
          border: '1px solid rgba(124,109,240,0.25)',
          borderRadius: 20, fontSize: 10, padding: '4px 12px',
          fontFamily: 'monospace', letterSpacing: 0.5,
        }}>
          {badge}
        </div>
      )}
    </div>
  )
}

export function BarRow({ label, value, max = 1, color }) {
  const pct = Math.round((value / max) * 100)
  const c = color || (value / max >= 0.7 ? 'var(--accent3)' : value / max >= 0.5 ? 'var(--amber)' : 'var(--accent2)')
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
      <div style={{ fontSize: 12, color: 'var(--muted)', minWidth: 90, flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, height: 8, background: 'var(--surface2)', borderRadius: 4, overflow: 'hidden' }}>
        <div className="bar-animate" style={{ height: '100%', borderRadius: 4, background: c, width: `${pct}%` }} />
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, minWidth: 36, textAlign: 'right' }}>
        {typeof value === 'number' && value <= 1 ? value.toFixed(2) : Math.round(value)}
      </div>
    </div>
  )
}

export function Spinner({ size = 28 }) {
  return (
    <div style={{
      width: size, height: size,
      border: '2px solid var(--border)',
      borderTopColor: 'var(--accent)',
      borderRadius: '50%',
    }} className="spin" />
  )
}
