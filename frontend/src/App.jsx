import React, { useEffect, useState } from 'react'
import {
  Check,
  ChevronDown,
  ChevronUp,
  Download,
  MessageSquare,
  RefreshCw,
  Send,
  X,
} from 'lucide-react'
import WaferInspector from './components/WaferInspector.jsx'
import {
  DistributionChart,
  ForecastChart,
  LotTrendChart,
  ParetoChart,
  ProbeSiteChart,
  ShapBars,
  TestTimeBars,
} from './components/SvgCharts.jsx'

const TABS = [
  { id: 'actions', label: 'Action Plan' },
  { id: 'wafers', label: 'Wafer Maps' },
  { id: 'trends', label: 'Trends & Causes' },
  { id: 'risk_speed', label: 'Hidden Risk & Speed' },
  { id: 'models', label: 'Model Transparency' },
]

const SAMPLE_QUESTIONS = [
  'Why did yield drop?',
  'Which probe card has a contact issue?',
  'Which tests can we safely sample?',
  'How do we catch hidden burn-in failures?',
]

export default function App() {
  const [data, setData] = useState(null)
  const [activeTab, setActiveTab] = useState('actions')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedEvidence, setExpandedEvidence] = useState({})
  const [question, setQuestion] = useState('')
  const [qaResult, setQaResult] = useState(null)
  const [asking, setAsking] = useState(false)

  const loadPreloaded = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/preloaded')
      const payload = await res.json()
      if (!res.ok) throw new Error(payload.detail || 'Failed to load preloaded dataset')
      setData(payload)
    } catch (err) {
      setError(err.message || 'Could not load preloaded dataset.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPreloaded()
  }, [])

  const handleActionDecision = async (actionId, status) => {
    if (!data) return
    try {
      const res = await fetch(`/api/session/${data.session_id}/action/${actionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      if (res.ok) {
        setData((prev) => ({
          ...prev,
          actions: prev.actions.map((a) => (a.id === actionId ? { ...a, status } : a)),
        }))
      }
    } catch {
      // ignore network blip
    }
  }

  const handleAsk = async (qText) => {
    const query = (qText ?? question).trim()
    if (!query || !data) return
    setQuestion(query)
    setAsking(true)
    try {
      const res = await fetch(`/api/session/${data.session_id}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      })
      if (res.ok) {
        setQaResult(await res.json())
      }
    } finally {
      setAsking(false)
    }
  }

  if (loading && !data) {
    return (
      <div className="page-shell">
        <div className="upload-container" style={{ textAlign: 'center', marginTop: 96 }}>
          <div className="card" style={{ padding: '44px 28px' }}>
            <h1 style={{ fontSize: 22, fontWeight: 600, margin: '0 0 8px 0' }}>Yield Advisor</h1>
            <p style={{ fontSize: 14, color: '#4b5563', margin: '0 0 18px 0' }}>
              Running ML models on the preloaded AX7-SOC chip-test dataset...
            </p>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#6b7280' }}>
              <span
                className="spinner"
                style={{ borderColor: '#cbd5e1', borderTopColor: '#1e3a5f' }}
              />
              <span>Preparing wafer maps, trend alerts, root causes, and action plan</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="page-shell">
        <div className="upload-container" style={{ marginTop: 80 }}>
          <div className="card">
            <h2 className="card-title">Could not initialize dataset</h2>
            <div className="error-banner">{error}</div>
            <button type="button" className="btn btn-primary" style={{ marginTop: 14 }} onClick={loadPreloaded}>
              <RefreshCw size={14} /> Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  const { summary, actions, pareto, excursion, root_cause, outliers, test_optimizer, yield_forecast, models_used, quality_report } = data

  return (
    <div className="page-shell">
      {/* Top Bar */}
      <header className="topbar">
        <div className="brand">
          <h1 className="brand-title">Yield Advisor</h1>
          <p className="brand-sub tabular">
            Preloaded AX7-SOC · {summary.total_lots} lots · {summary.total_wafers.toLocaleString()} wafers ·{' '}
            {summary.total_chips.toLocaleString()} chips analyzed in {data.runtime_seconds}s
          </p>
        </div>

        <div className="topbar-actions">
          <a
            href={`/api/session/${data.session_id}/export`}
            className="btn btn-primary"
            style={{ textDecoration: 'none' }}
          >
            <Download size={14} />
            Export Summary
          </a>
        </div>
      </header>

      {/* Top 4 KPIs */}
      <section className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Final Yield</div>
          <div className="kpi-value tabular">{summary.final_yield_pct}%</div>
          <div className="kpi-sub tabular">
            First pass {summary.first_pass_yield_pct}%
            {summary.retest_recovered_chips > 0 ? ` · +${summary.retest_recovered_chips.toLocaleString()} retest saved` : ''}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Issues Flagged</div>
          <div className="kpi-value tabular">{summary.active_alerts_count}</div>
          <div className="kpi-sub">
            {data.wafer_patterns_summary?.non_clean_wafers || 0} wafers with spatial patterns
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Hidden Risk Chips</div>
          <div className="kpi-value tabular">
            {summary.flagged_risk_pct !== null ? `${summary.flagged_risk_pct}%` : '—'}
          </div>
          <div className="kpi-sub tabular">
            {summary.flagged_risk_pct !== null
              ? `${summary.flagged_risk_count.toLocaleString()} passing chips flagged for risk`
              : 'Needs die-level test values'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Test Time Savings</div>
          <div className="kpi-value tabular">
            {summary.test_time_saved_pct !== null ? `${summary.test_time_saved_pct}%` : '—'}
          </div>
          <div className="kpi-sub tabular">
            {summary.test_time_saved_pct !== null
              ? `~${summary.test_time_saved_ms} ms faster per chip`
              : 'Needs test timing values'}
          </div>
        </div>
      </section>

      {/* Navigation Tabs */}
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Tab 1: Action Plan */}
      {activeTab === 'actions' && (
        <div className="grid-main-side">
          <div>
            <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <h2 className="card-title" style={{ fontSize: 16 }}>Recommended Actions</h2>
              <span style={{ fontSize: 12, color: '#6b7280' }}>
                Ranked by impact &amp; confidence · Engineer approval required
              </span>
            </div>

            {actions.length === 0 ? (
              <div className="card">
                <p className="card-desc" style={{ margin: 0 }}>
                  No major anomalies detected in this dataset. Yield appears stable.
                </p>
              </div>
            ) : (
              <div className="action-list">
                {actions.map((act) => {
                  const isExp = !!expandedEvidence[act.id]
                  return (
                    <div key={act.id} className={`action-card ${act.status}`}>
                      <div className="action-head">
                        <div className="action-title-row">
                          <span className="action-rank">{act.rank}</span>
                          <h3 className="action-title">{act.title}</h3>
                          <span className="pill pill-neutral">{act.category}</span>
                        </div>

                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <span className={`pill ${act.confidence_label === 'High' ? 'pill-ok' : 'pill-warn'}`}>
                            {act.confidence_label} ({act.confidence_pct}%)
                          </span>
                        </div>
                      </div>

                      <p className="action-why">{act.why}</p>

                      <div className="action-meta">
                        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className="pill pill-accent">{act.expected_benefit}</span>
                          <span style={{ color: '#6b7280', fontSize: 12 }}>Owner: {act.approval_owner}</span>
                        </div>

                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <button
                            type="button"
                            className="btn btn-sm btn-ghost"
                            onClick={() => setExpandedEvidence((p) => ({ ...p, [act.id]: !p[act.id] }))}
                          >
                            {isExp ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                            Evidence
                          </button>

                          {act.status === 'accepted' ? (
                            <span className="pill pill-ok">Approved</span>
                          ) : act.status === 'dismissed' ? (
                            <span className="pill pill-neutral">Dismissed</span>
                          ) : (
                            <>
                              <button
                                type="button"
                                className="btn btn-sm"
                                onClick={() => handleActionDecision(act.id, 'accepted')}
                              >
                                <Check size={13} /> Approve
                              </button>
                              <button
                                type="button"
                                className="btn btn-sm btn-ghost"
                                onClick={() => handleActionDecision(act.id, 'dismissed')}
                              >
                                <X size={13} />
                              </button>
                            </>
                          )}
                        </div>
                      </div>

                      {isExp && <div className="action-evidence">{act.evidence}</div>}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Side Column: Top Failure Reasons + Grounded Q&A */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div className="card">
              <h3 className="card-title">Where Chips Fail Most</h3>
              <p className="card-desc">Top failure reasons across all tested dies.</p>
              <ParetoChart data={pareto} />
            </div>

            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
                <MessageSquare size={15} color="#1e3a5f" />
                <h3 className="card-title" style={{ margin: 0 }}>Ask the Advisor</h3>
              </div>
              <p className="card-desc">
                Ask a plain question — answers come strictly from the calculated numbers.
              </p>

              <div className="filter-pills">
                {SAMPLE_QUESTIONS.map((sq) => (
                  <button
                    key={sq}
                    type="button"
                    className="filter-pill"
                    onClick={() => handleAsk(sq)}
                  >
                    {sq}
                  </button>
                ))}
              </div>

              <form
                className="qa-input-row"
                onSubmit={(e) => {
                  e.preventDefault()
                  handleAsk()
                }}
              >
                <input
                  type="text"
                  className="qa-input"
                  placeholder="e.g. Tell me about lot L262226..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
                <button type="submit" className="btn btn-primary" disabled={asking}>
                  <Send size={13} />
                </button>
              </form>

              {qaResult && (
                <div className="qa-answer">
                  <div>{qaResult.answer}</div>
                  {qaResult.sources?.length > 0 && (
                    <div style={{ fontSize: 11.5, color: '#4b5563', marginTop: 6 }}>
                      Sources: {qaResult.sources.join(' · ')}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Wafer Maps */}
      {activeTab === 'wafers' && (
        <WaferInspector
          sessionId={data.session_id}
          waferTable={data.wafer_table}
          patternSummary={data.wafer_patterns_summary}
        />
      )}

      {/* Tab 3: Trends & Root Causes */}
      {activeTab === 'trends' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="card">
            <h3 className="card-title">Lot Yield Over Time</h3>
            <p className="card-desc">
              Smoothed trend (EWMA) catches small sustained yield drops earlier than single-lot threshold checks.
            </p>
            <LotTrendChart trend={excursion.lot_trend} baseline={excursion.baseline_yield_pct} />

            {excursion.excursions?.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 10, marginTop: 14 }}>
                {excursion.excursions.map((ex, i) => (
                  <div key={i} className="info-banner">
                    <div style={{ fontWeight: 600, color: '#b42318' }}>{ex.title}</div>
                    <div>
                      Peak drop: <strong>{ex.peak_yield_drop_pts} pts</strong> · Main drivers: {ex.top_drivers}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="grid-2">
            <div className="card">
              <h3 className="card-title">Top Yield-Loss Drivers (ML Attribution)</h3>
              <p className="card-desc">
                Gradient-boosted model (LightGBM + SHAP) ranking equipment, wafer electrical tests, and time windows.
              </p>
              <ShapBars items={root_cause.shap_feature_importance} />
            </div>

            <div className="card">
              <h3 className="card-title">Statistical Commonality Verification</h3>
              <p className="card-desc">
                Equipment and process factors verified with Fisher / Mann-Whitney significance tests.
              </p>
              {root_cause.ranked_causes?.length > 0 ? (
                <table className="clean-table">
                  <thead>
                    <tr>
                      <th>Factor</th>
                      <th>Pattern / Effect</th>
                      <th style={{ textAlign: 'right' }}>Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {root_cause.ranked_causes.map((rc, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 500, color: '#111827' }}>{rc.factor}</td>
                        <td>{rc.signature} ({rc.wafers_affected} wafers)</td>
                        <td className="tabular" style={{ textAlign: 'right', color: '#b42318', fontWeight: 500 }}>
                          -{rc.est_dataset_yield_impact_pts} pts
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="card-desc">No single equipment factor showed significant yield separation.</p>
              )}
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <h3 className="card-title">Probe Card vs. Silicon Check</h3>
              <p className="card-desc">
                Comparing the 4 probe sites on each card reveals whether failures come from worn test needles or real wafer defects.
              </p>
              <ProbeSiteChart siteSummary={root_cause.probe_site_summary} />

              {root_cause.retest_summary?.length > 0 && (
                <div className="info-banner" style={{ marginTop: 12 }}>
                  {root_cause.retest_summary.map((r) => (
                    <div key={r.soft_bin}>
                      <strong>{r.bin_name} retest recovery:</strong> {r.recovery_rate_pct}% passed on retest (
                      {r.recovered_dies.toLocaleString()} of {r.retested_dies.toLocaleString()} chips recovered).
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="card">
              <h3 className="card-title">Pre-Sort Yield Forecast</h3>
              <p className="card-desc">
                {yield_forecast.available
                  ? `Predicted from early electrical measurements (WAT) before wafer testing · Average error (MAE): ${yield_forecast.mae_yield_pts} yield pts.`
                  : yield_forecast.reason}
              </p>
              {yield_forecast.available && <ForecastChart forecasts={yield_forecast.lot_forecasts} />}
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Hidden Risk & Speed */}
      {activeTab === 'risk_speed' && (
        <div className="grid-2">
          <div className="card">
            <h3 className="card-title">Hidden Risk Screening (Passing Chips)</h3>
            <p className="card-desc">
              Some chips pass static specification limits yet behave very differently from their neighbors on the wafer.
            </p>

            {outliers.available ? (
              <>
                <DistributionChart bins={outliers.iddq_distribution} />

                {outliers.burnin_validation && (
                  <div className="info-banner" style={{ marginTop: 14, background: '#ecfdf3', borderColor: '#abefc6' }}>
                    <div style={{ fontWeight: 600, color: '#1f6f4a', marginBottom: 4 }}>
                      Validated Against Burn-In Failures
                    </div>
                    <div>
                      Chips flagged by Neighbor Spike (NNR) failed burn-in at{' '}
                      <strong>{outliers.burnin_validation.nnr_fail_rate_pct}%</strong> vs{' '}
                      <strong>{outliers.burnin_validation.baseline_fail_rate_pct}%</strong> for normal passing chips —
                      catching <strong>{outliers.burnin_validation.burnin_recall_pct}%</strong> of all burn-in failures at only{' '}
                      <strong>{outliers.burnin_validation.yield_cost_pct}%</strong> yield cost.
                    </div>
                  </div>
                )}

                <table className="clean-table" style={{ marginTop: 14 }}>
                  <thead>
                    <tr>
                      <th>Screening Method</th>
                      <th style={{ textAlign: 'right' }}>Flagged</th>
                    </tr>
                  </thead>
                  <tbody>
                    {outliers.screens.map((s) => (
                      <tr key={s.key}>
                        <td>
                          <div style={{ fontWeight: 500, color: '#111827' }}>{s.name}</div>
                          <div style={{ fontSize: 12, color: '#6b7280' }}>{s.plain_desc}</div>
                        </td>
                        <td className="tabular" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                          {s.flagged_count.toLocaleString()} ({s.flagged_pct}%)
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : (
              <p className="card-desc">{outliers.reason}</p>
            )}
          </div>

          <div className="card">
            <h3 className="card-title">Faster Testing Opportunities</h3>
            <p className="card-desc">
              {test_optimizer.available
                ? `Identifies redundant or near-zero-fail tests to move to 5% audit sampling — saving ${test_optimizer.total_saved_ms} ms/chip (${test_optimizer.pct_time_reduction}% faster).`
                : test_optimizer.reason}
            </p>

            {test_optimizer.available && (
              <>
                <TestTimeBars items={test_optimizer.test_items} />

                {test_optimizer.candidates?.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                      Safe Sampling / Removal Candidates
                    </div>
                    <table className="clean-table">
                      <thead>
                        <tr>
                          <th>Test Item</th>
                          <th>Why Safe</th>
                          <th style={{ textAlign: 'right' }}>95% Risk Bound</th>
                        </tr>
                      </thead>
                      <tbody>
                        {test_optimizer.candidates.map((c) => (
                          <tr key={c.test_name}>
                            <td style={{ fontWeight: 500, color: '#111827' }}>{c.test_name}</td>
                            <td style={{ fontSize: 12 }}>{c.reason}</td>
                            <td className="tabular" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                              &le; {c.escape_bound_dppm} DPPM
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Tab 5: Model Transparency */}
      {activeTab === 'models' && (
        <div className="grid-2">
          <div className="card">
            <h3 className="card-title">Machine Learning Models Used</h3>
            <p className="card-desc">
              Every insight is produced dynamically by specialized models — no hardcoded rules.
            </p>
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Model / Method</th>
                  <th>Validation</th>
                </tr>
              </thead>
              <tbody>
                {models_used.map((m) => (
                  <tr key={m.engine}>
                    <td style={{ fontWeight: 500, color: '#111827' }}>{m.engine}</td>
                    <td>{m.model}</td>
                    <td className="tabular">{m.metric}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h3 className="card-title">Preloaded Dataset &amp; Quality Check</h3>
            <p className="card-desc">
              Automatic schema reconciliation and ground-truth isolation checks.
            </p>
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Role</th>
                  <th style={{ textAlign: 'right' }}>Rows</th>
                </tr>
              </thead>
              <tbody>
                {quality_report.files_loaded.map((f, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 500, color: '#111827' }}>{f.file}</td>
                    <td>{f.role}</td>
                    <td className="tabular" style={{ textAlign: 'right' }}>{f.rows.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="info-banner" style={{ marginTop: 14 }}>
              <div>Retest rows reconciled: <strong>{quality_report.retest_rows_detected.toLocaleString()}</strong></div>
              <div>Invalid coordinate rows dropped: <strong>{quality_report.invalid_rows_dropped}</strong></div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
