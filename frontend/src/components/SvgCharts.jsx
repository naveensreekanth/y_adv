import React, { useState } from 'react'

export function ParetoChart({ data }) {
  if (!data || data.length === 0) {
    return <p className="card-desc">No failure breakdown available for this upload.</p>
  }

  const maxCnt = Math.max(...data.map((d) => d.fail_count), 1)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {data.slice(0, 8).map((item) => {
        const pctWidth = Math.max(2, (item.fail_count / maxCnt) * 100)
        return (
          <div key={item.soft_bin} style={{ display: 'grid', gridTemplateColumns: '92px 1fr 112px', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 12.5, fontWeight: 500, color: '#374151', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {item.name}
            </span>
            <div style={{ height: 10, background: '#f1f5f9', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{ width: `${pctWidth}%`, height: '100%', background: '#1e3a5f', borderRadius: 99 }} />
            </div>
            <span className="tabular" style={{ fontSize: 12, color: '#6b7280', textAlign: 'right' }}>
              {item.fail_count.toLocaleString()} ({item.share_pct}%)
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function LotTrendChart({ trend, baseline }) {
  const [hoverIdx, setHoverIdx] = useState(null)
  if (!trend || trend.length === 0) return null

  const W = 680
  const H = 240
  const pad = { top: 20, right: 20, bottom: 34, left: 44 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom

  const allY = trend.flatMap((d) => [d.yield_pct, d.ewma_pct, d.ewma_lcl])
  const ymin = Math.max(0, Math.floor(Math.min(...allY) - 2))
  const ymax = Math.min(100, Math.ceil(Math.max(...allY) + 1.5))
  const spanY = Math.max(1, ymax - ymin)

  const xOf = (i) => pad.left + (trend.length === 1 ? innerW / 2 : (i / (trend.length - 1)) * innerW)
  const yOf = (v) => pad.top + ((ymax - v) / spanY) * innerH

  const rawPoints = trend.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.yield_pct).toFixed(1)}`).join(' ')
  const ewmaPoints = trend.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.ewma_pct).toFixed(1)}`).join(' ')
  const lclY = yOf(trend[0].ewma_lcl)

  const yTicks = [ymin, Math.round((ymin + ymax) / 2), ymax]
  const hovered = hoverIdx !== null ? trend[hoverIdx] : null

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={yOf(t)} y2={yOf(t)} stroke="#e5e7eb" strokeWidth="1" />
            <text x={pad.left - 8} y={yOf(t) + 4} textAnchor="end" fontSize="11" fill="#6b7280">
              {t}%
            </text>
          </g>
        ))}

        {/* Early-warning lower threshold */}
        <line
          x1={pad.left}
          x2={W - pad.right}
          y1={lclY}
          y2={lclY}
          stroke="#d97706"
          strokeWidth="1.2"
          strokeDasharray="4 4"
        />
        <text x={W - pad.right} y={lclY - 5} textAnchor="end" fontSize="10.5" fill="#b45309">
          Early-warning threshold ({trend[0].ewma_lcl}%)
        </text>

        {/* Raw lot yield */}
        <polyline fill="none" stroke="#94a3b8" strokeWidth="1.6" points={rawPoints} />

        {/* Smoothed EWMA line */}
        <polyline fill="none" stroke="#1e3a5f" strokeWidth="2.4" points={ewmaPoints} />

        {/* Points & hover zones */}
        {trend.map((d, i) => {
          const cx = xOf(i)
          const cy = yOf(d.yield_pct)
          const showLabel = i === 0 || i === trend.length - 1 || i % Math.max(1, Math.floor(trend.length / 6)) === 0
          return (
            <g key={d.lot_id}>
              {d.ewma_alert && (
                <circle cx={cx} cy={cy} r={6} fill="#fef2f2" stroke="#b42318" strokeWidth="1.5" />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={hoverIdx === i ? 5 : 3}
                fill={d.ewma_alert ? '#b42318' : '#1e3a5f'}
              />
              {showLabel && (
                <text x={cx} y={H - 10} textAnchor="middle" fontSize="10" fill="#6b7280">
                  {d.lot_id.replace('L26', '')}
                </text>
              )}
              <rect
                x={cx - innerW / Math.max(2, trend.length * 2)}
                y={pad.top}
                width={innerW / Math.max(1, trend.length)}
                height={innerH}
                fill="transparent"
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(null)}
              />
            </g>
          )
        })}
      </svg>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8, minHeight: 22, fontSize: 12.5, color: '#4b5563' }}>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <span><strong style={{ color: '#94a3b8' }}>—</strong> Raw Lot Yield</span>
          <span><strong style={{ color: '#1e3a5f' }}>—</strong> Smoothed Trend (EWMA)</span>
          <span><strong style={{ color: '#b42318' }}>●</strong> Shift Flagged</span>
        </div>
        {hovered && (
          <span className="tabular" style={{ fontWeight: 500, color: '#111827' }}>
            {hovered.lot_id}: {hovered.yield_pct}% (Trend {hovered.ewma_pct}% · baseline {baseline}%)
          </span>
        )}
      </div>
    </div>
  )
}

export function ShapBars({ items }) {
  if (!items || items.length === 0) {
    return <p className="card-desc">Upload equipment or electrical context to see feature attribution.</p>
  }
  const maxImp = Math.max(...items.map((d) => d.mean_abs_shap_pts), 0.1)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {items.slice(0, 6).map((item) => {
        const w = Math.max(3, (item.mean_abs_shap_pts / maxImp) * 100)
        return (
          <div key={item.feature}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
              <span style={{ fontWeight: 500, color: '#111827' }}>{item.display_name}</span>
              <span className="tabular" style={{ color: '#4b5563', fontSize: 12 }}>
                {item.share_pct}% share (±{item.mean_abs_shap_pts.toFixed(2)} pts)
              </span>
            </div>
            <div style={{ height: 8, background: '#f1f5f9', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{ width: `${w}%`, height: '100%', background: '#1e3a5f', borderRadius: 99 }} />
            </div>
            {item.detail && (
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 3 }}>{item.detail}</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function ProbeSiteChart({ siteSummary }) {
  if (!siteSummary || siteSummary.length === 0) {
    return <p className="card-desc">No multi-site probe card data in this upload.</p>
  }

  // Group by probe_card_id
  const byCard = {}
  for (const row of siteSummary) {
    byCard[row.probe_card_id] = byCard[row.probe_card_id] || []
    byCard[row.probe_card_id].push(row)
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 14 }}>
      {Object.entries(byCard).map(([pc, sites]) => {
        const med = sites.map((s) => s.yield_pct).sort((a, b) => a - b)[Math.floor(sites.length / 2)]
        return (
          <div key={pc} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: '12px 14px' }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>{pc}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {sites.sort((a, b) => a.site - b.site).map((s) => {
                const lagging = med - s.yield_pct >= 0.8
                const barW = Math.max(10, Math.min(100, s.yield_pct))
                return (
                  <div key={s.site} style={{ display: 'grid', gridTemplateColumns: '48px 1fr 48px', alignItems: 'center', gap: 8, fontSize: 12 }}>
                    <span style={{ color: lagging ? '#b42318' : '#4b5563', fontWeight: lagging ? 600 : 400 }}>
                      Site {s.site}
                    </span>
                    <div style={{ height: 8, background: '#f1f5f9', borderRadius: 99, overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${barW}%`,
                          height: '100%',
                          background: lagging ? '#b42318' : '#2e7d5b',
                          borderRadius: 99,
                        }}
                      />
                    </div>
                    <span className="tabular" style={{ textAlign: 'right', color: lagging ? '#b42318' : '#374151', fontWeight: lagging ? 600 : 400 }}>
                      {s.yield_pct.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function DistributionChart({ bins }) {
  if (!bins || bins.length === 0) return null
  const maxN = Math.max(...bins.map((b) => b.normal_count), 1)
  const W = 520
  const H = 160
  const pad = { top: 14, right: 12, bottom: 26, left: 36 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom
  const barW = innerW / bins.length - 1.5

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {bins.map((b, i) => {
        const x = pad.left + i * (innerW / bins.length)
        const hNorm = (b.normal_count / maxN) * innerH
        const hOut = Math.min(innerH, (b.outlier_count / maxN) * innerH * 12) // scaled slightly for visibility of 0.3% rare spikes
        return (
          <g key={i}>
            <rect
              x={x}
              y={pad.top + innerH - hNorm}
              width={Math.max(2, barW)}
              height={hNorm}
              fill="#cbd5e1"
              rx="1.5"
            />
            {b.outlier_count > 0 && (
              <rect
                x={x}
                y={pad.top + innerH - Math.max(3, hOut)}
                width={Math.max(2, barW)}
                height={Math.max(3, hOut)}
                fill="#b42318"
                rx="1.5"
              />
            )}
            {i % 5 === 0 && (
              <text x={x + barW / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="#6b7280">
                {b.bin_center}
              </text>
            )}
          </g>
        )
      })}
      <text x={pad.left} y={11} fontSize="10" fill="#6b7280">
        Gray: Normal passing chips · Red: Neighbor-spike outliers (uA)
      </text>
    </svg>
  )
}

export function TestTimeBars({ items }) {
  if (!items || items.length === 0) return null
  const sorted = [...items].sort((a, b) => b.test_time_ms - a.test_time_ms)
  const maxMs = Math.max(...sorted.map((t) => t.test_time_ms), 1)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      {sorted.map((t) => {
        const w = Math.max(3, (t.test_time_ms / maxMs) * 100)
        return (
          <div key={t.test_name} style={{ display: 'grid', gridTemplateColumns: '130px 1fr 125px', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 12.5, fontWeight: 500, color: t.candidate_for_sampling ? '#a15c07' : '#374151' }}>
              {t.test_name}
            </span>
            <div style={{ height: 9, background: '#f1f5f9', borderRadius: 99, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${w}%`,
                  height: '100%',
                  background: t.candidate_for_sampling ? '#d97706' : '#1e3a5f',
                  borderRadius: 99,
                }}
              />
            </div>
            <span className="tabular" style={{ fontSize: 12, color: '#6b7280', textAlign: 'right' }}>
              {t.test_time_ms} ms · {t.unique_fails} fails
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function ForecastChart({ forecasts }) {
  if (!forecasts || forecasts.length === 0) return null
  const W = 680
  const H = 200
  const pad = { top: 18, right: 20, bottom: 30, left: 44 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom

  const allY = forecasts.flatMap((d) => [d.actual_yield_pct, d.pred_p10_pct, d.pred_p90_pct])
  const ymin = Math.max(0, Math.floor(Math.min(...allY) - 2))
  const ymax = Math.min(100, Math.ceil(Math.max(...allY) + 1.5))
  const spanY = Math.max(1, ymax - ymin)

  const xOf = (i) => pad.left + (forecasts.length === 1 ? innerW / 2 : (i / (forecasts.length - 1)) * innerW)
  const yOf = (v) => pad.top + ((ymax - v) / spanY) * innerH

  const actualPts = forecasts.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.actual_yield_pct).toFixed(1)}`).join(' ')
  const predPts = forecasts.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.pred_p50_pct).toFixed(1)}`).join(' ')
  const bandPts =
    forecasts.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.pred_p90_pct).toFixed(1)}`).join(' ') +
    ' ' +
    [...forecasts].reverse().map((d, revI) => {
      const i = forecasts.length - 1 - revI
      return `${xOf(i).toFixed(1)},${yOf(d.pred_p10_pct).toFixed(1)}`
    }).join(' ')

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[ymin, Math.round((ymin + ymax) / 2), ymax].map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={yOf(t)} y2={yOf(t)} stroke="#e5e7eb" strokeWidth="1" />
            <text x={pad.left - 8} y={yOf(t) + 4} textAnchor="end" fontSize="11" fill="#6b7280">{t}%</text>
          </g>
        ))}
        <polygon points={bandPts} fill="#e2e8f0" opacity="0.65" />
        <polyline fill="none" stroke="#1e3a5f" strokeWidth="2" strokeDasharray="4 3" points={predPts} />
        <polyline fill="none" stroke="#1f6f4a" strokeWidth="2" points={actualPts} />
      </svg>
      <div style={{ display: 'flex', gap: 14, fontSize: 12.5, color: '#4b5563', marginTop: 6 }}>
        <span><strong style={{ color: '#1f6f4a' }}>—</strong> Actual Sort Yield</span>
        <span><strong style={{ color: '#1e3a5f' }}>- -</strong> Pre-Sort Forecast (Median P50)</span>
        <span><strong style={{ color: '#94a3b8' }}>■</strong> Uncertainty Range (P10–P90)</span>
      </div>
    </div>
  )
}
