import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import AuditPage from './pages/AuditPage'
import DashboardPage from './pages/DashboardPage'
import FairnessPage from './pages/FairnessPage'
import ShapPage from './pages/ShapPage'
import ReportsPage from './pages/ReportsPage'

const NAV = [
  { to: '/',          icon: '⊞', label: 'Dashboard'  },
  { to: '/audit',     icon: '⚖', label: 'Run Audit'  },
  { to: '/fairness',  icon: '◈', label: 'Fairness'   },
  { to: '/shap',      icon: '◉', label: 'SHAP'       },
  { to: '/reports',   icon: '▤', label: 'Reports'    },
]

export default function App() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{
        width: 220, flexShrink: 0,
        background: 'var(--surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        padding: '24px 0',
      }}>
        {/* Logo */}
        <div style={{ padding: '0 20px 24px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 34, height: 34, borderRadius: 10,
              background: 'linear-gradient(135deg, #7c6df0, #e05b8b)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 17,
            }}>⚖</div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 800, letterSpacing: '-0.3px' }}>
                Ethical AI
              </div>
              <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>
                Auditor v1.0
              </div>
            </div>
          </div>
        </div>

        {/* Nav links */}
        <nav style={{ padding: '16px 12px', flex: 1 }}>
          {NAV.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              style={({ isActive }) => ({
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 12px', borderRadius: 8,
                textDecoration: 'none', marginBottom: 4,
                fontSize: 13, fontWeight: 600,
                color: isActive ? 'var(--text)' : 'var(--muted)',
                background: isActive ? 'var(--surface2)' : 'transparent',
                transition: 'all 0.15s',
              })}
            >
              <span style={{ fontSize: 14 }}>{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace', lineHeight: 1.6 }}>
            Powered by Claude AI<br />
            NIST AI RMF · EU AI Act<br />
            IEEE 7000 Aligned
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        <Routes>
          <Route path="/"         element={<DashboardPage />} />
          <Route path="/audit"    element={<AuditPage />} />
          <Route path="/fairness" element={<FairnessPage />} />
          <Route path="/shap"     element={<ShapPage />} />
          <Route path="/reports"  element={<ReportsPage />} />
        </Routes>
      </main>
    </div>
  )
}
