import React, { useEffect, useState } from 'react'

export default function WaferInspector({ sessionId, waferTable, patternSummary }) {
  const [patternFilter, setPatternFilter] = useState('ALL')
  const [lotSearch, setLotSearch] = useState('')
  const [selectedKey, setSelectedKey] = useState(null)
  const [waferDetail, setWaferDetail] = useState(null)
  const [loadingMap, setLoadingMap] = useState(false)
  const [overlayMode, setOverlayMode] = useState('fail') // 'fail' | 'gradcam' | 'outlier'
  const [selectedDie, setSelectedDie] = useState(null)

  const counts = patternSummary?.counts || {}
  const patternOptions = ['ALL', ...Object.keys(counts).sort((a, b) => (a === 'Clean' ? 1 : b === 'Clean' ? -1 : (counts[b] || 0) - (counts[a] || 0)))]

  const filteredWafers = (waferTable || []).filter((w) => {
    if (patternFilter !== 'ALL' && w.primary_pattern !== patternFilter) return false
    if (lotSearch.trim() && !w.lot_id.toLowerCase().includes(lotSearch.trim().toLowerCase())) return false
    return true
  })

  // Auto-select the first interesting non-clean wafer on first open
  useEffect(() => {
    if (!selectedKey && waferTable && waferTable.length > 0) {
      const interesting = waferTable.find((w) => w.primary_pattern !== 'Clean') || waferTable[0]
      setSelectedKey({ lot_id: interesting.lot_id, wafer_id: interesting.wafer_id })
    }
  }, [waferTable, selectedKey])

  useEffect(() => {
    if (!selectedKey || !sessionId) return
    let active = true
    setLoadingMap(true)
    setSelectedDie(null)
    fetch(`/api/session/${sessionId}/wafer/${encodeURIComponent(selectedKey.lot_id)}/${selectedKey.wafer_id}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('Map unavailable'))))
      .then((data) => {
        if (active) setWaferDetail(data)
      })
      .catch(() => {
        if (active) setWaferDetail(null)
      })
      .finally(() => {
        if (active) setLoadingMap(false)
      })
    return () => {
      active = false
    }
  }, [sessionId, selectedKey])

  if (!patternSummary?.available) {
    return (
      <div className="card">
        <h3 className="card-title">Wafer Maps</h3>
        <p className="card-desc">
          Upload a die-level CSV or Excel file containing chip coordinates (`die_x`, `die_y`) to view spatial wafer maps and AI pattern detection.
        </p>
      </div>
    )
  }

  return (
    <div className="grid-main-side">
      {/* Left: Interactive Wafer Map & Diagnosis */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
          <div>
            <h3 className="card-title">
              {selectedKey ? `${selectedKey.lot_id} · Wafer ${selectedKey.wafer_id}` : 'Wafer Map'}
            </h3>
            <p className="card-desc" style={{ marginBottom: 0 }}>
              {waferDetail?.pattern?.plain_pattern_name || 'Click any wafer on the right to inspect'}
              {waferDetail?.pattern?.primary_pattern !== 'Clean' && waferDetail?.pattern?.pattern_confidence
                ? ` (${Math.round(waferDetail.pattern.pattern_confidence * 100)}% AI confidence)`
                : ''}
            </p>
          </div>

          <div className="wafer-controls">
            <button
              type="button"
              className={`btn btn-sm ${overlayMode === 'fail' ? 'btn-primary' : ''}`}
              onClick={() => setOverlayMode('fail')}
            >
              Pass / Fail Map
            </button>
            <button
              type="button"
              className={`btn btn-sm ${overlayMode === 'gradcam' ? 'btn-primary' : ''}`}
              onClick={() => setOverlayMode('gradcam')}
            >
              AI Focus (Grad-CAM)
            </button>
            <button
              type="button"
              className={`btn btn-sm ${overlayMode === 'outlier' ? 'btn-primary' : ''}`}
              onClick={() => setOverlayMode('outlier')}
            >
              Hidden Risk Outliers
            </button>
          </div>
        </div>

        <div className="wafer-stage">
          {loadingMap ? (
            <div style={{ padding: 80, color: '#6b7280', fontSize: 13 }}>Rendering wafer map...</div>
          ) : waferDetail ? (
            <WaferSvg
              detail={waferDetail}
              overlayMode={overlayMode}
              selectedDie={selectedDie}
              onSelectDie={setSelectedDie}
            />
          ) : (
            <div style={{ padding: 60, color: '#6b7280', fontSize: 13 }}>Die coordinates unavailable for this wafer.</div>
          )}

          {/* Legend */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#4b5563' }}>
            <span><strong style={{ color: '#dbe4ee' }}>■</strong> Pass</span>
            <span><strong style={{ color: '#c8372d' }}>■</strong> Fail</span>
            {overlayMode === 'gradcam' && (
              <span><strong style={{ color: '#d97706' }}>■</strong> High AI Attention Region</span>
            )}
            {overlayMode === 'outlier' && (
              <span><strong style={{ color: '#7c3aed' }}>■</strong> Passed Spec, Flagged Hidden Risk</span>
            )}
          </div>
        </div>

        {/* Plain-English Diagnosis & Selected Die Readout */}
        {waferDetail && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 12, marginTop: 18 }}>
            <div className="info-banner">
              <div style={{ fontWeight: 600, color: '#111827', marginBottom: 4 }}>What the AI sees</div>
              <div>{waferDetail.pattern.cause_hint}</div>
              {waferDetail.pattern.reticle_detail && (
                <div style={{ marginTop: 6, fontWeight: 500 }}>
                  Repeating {waferDetail.pattern.reticle_detail.field_size} field at position {waferDetail.pattern.reticle_detail.field_pos}:{' '}
                  {Math.round(waferDetail.pattern.reticle_detail.pos_fail_rate * 100)}% fail rate vs{' '}
                  {Math.round(waferDetail.pattern.reticle_detail.other_fail_rate * 100)}% elsewhere.
                </div>
              )}
              {waferDetail.pattern.est_pattern_lost_dies > 0 && (
                <div style={{ marginTop: 6, color: '#b42318', fontWeight: 500 }}>
                  Estimated ~{waferDetail.pattern.est_pattern_lost_dies} extra chips lost to this pattern.
                </div>
              )}
              <div style={{ marginTop: 6, color: '#1e3a5f', fontWeight: 500 }}>
                Next step: {waferDetail.pattern.first_check}
              </div>
            </div>

            <div className="info-banner">
              <div style={{ fontWeight: 600, color: '#111827', marginBottom: 4 }}>
                {selectedDie ? `Chip (${selectedDie.x}, ${selectedDie.y})` : 'Click any chip on the map'}
              </div>
              {selectedDie ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <div>
                    Result:{' '}
                    <strong style={{ color: selectedDie.pass ? '#1f6f4a' : '#b42318' }}>
                      {selectedDie.pass ? 'PASS' : `FAIL (${selectedDie.bin_name})`}
                    </strong>{' '}
                    · Probe Site {selectedDie.site}
                  </div>
                  {selectedDie.outlier_reason && (
                    <div style={{ color: '#7c3aed', fontWeight: 500 }}>
                      Outlier flag: {selectedDie.outlier_reason}
                    </div>
                  )}
                  <div className="tabular" style={{ fontSize: 12, marginTop: 3 }}>
                    {Object.entries(selectedDie.params || {})
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(' · ') || 'No parametric values recorded'}
                  </div>
                </div>
              ) : (
                <div>Select a square on the wafer to see its exact failure reason, probe site, and test measurements.</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Right: Filter & Wafer Browser */}
      <div className="card">
        <h3 className="card-title">Browse Wafers</h3>
        <p className="card-desc">Filter by AI-detected defect signature or search by lot.</p>

        <div className="filter-pills">
          {patternOptions.map((pat) => (
            <button
              key={pat}
              type="button"
              className={`filter-pill ${patternFilter === pat ? 'active' : ''}`}
              onClick={() => setPatternFilter(pat)}
            >
              {pat === 'ALL' ? `All (${waferTable.length})` : `${pat} (${counts[pat] || 0})`}
            </button>
          ))}
        </div>

        <input
          type="text"
          className="qa-input"
          placeholder="Search lot ID (e.g. L262238)..."
          value={lotSearch}
          onChange={(e) => setLotSearch(e.target.value)}
          style={{ width: '100%', marginBottom: 12 }}
        />

        <div style={{ maxHeight: 420, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 8 }}>
          <table className="clean-table">
            <thead>
              <tr>
                <th>Lot / Wafer</th>
                <th>Pattern</th>
                <th style={{ textAlign: 'right' }}>Yield</th>
              </tr>
            </thead>
            <tbody>
              {filteredWafers.slice(0, 120).map((w) => {
                const isSel = selectedKey?.lot_id === w.lot_id && selectedKey?.wafer_id === w.wafer_id
                return (
                  <tr
                    key={`${w.lot_id}-${w.wafer_id}`}
                    className={`clickable ${isSel ? 'selected-row' : ''}`}
                    onClick={() => setSelectedKey({ lot_id: w.lot_id, wafer_id: w.wafer_id })}
                  >
                    <td className="tabular" style={{ fontWeight: 500, color: '#111827' }}>
                      {w.lot_id} · W{String(w.wafer_id).padStart(2, '0')}
                    </td>
                    <td>
                      <span className={`pill ${w.primary_pattern === 'Clean' ? 'pill-neutral' : 'pill-warn'}`}>
                        {w.primary_pattern}
                      </span>
                    </td>
                    <td className="tabular" style={{ textAlign: 'right' }}>
                      {w.final_yield_pct}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {filteredWafers.length > 120 && (
          <div style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>
            Showing first 120 of {filteredWafers.length} matching wafers. Narrow by lot search above.
          </div>
        )}
      </div>
    </div>
  )
}

function WaferSvg({ detail, overlayMode, selectedDie, onSelectDie }) {
  const { bounds, dies } = detail
  const cols = Math.max(1, bounds.xmax - bounds.xmin + 1)
  const rows = Math.max(1, bounds.ymax - bounds.ymin + 1)
  const size = 380
  const pad = 16
  const cell = (size - pad * 2) / Math.max(cols, rows)

  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 380, height: 'auto', display: 'block' }}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={(size - pad * 1.4) / 2}
        fill="#f8fafc"
        stroke="#cbd5e1"
        strokeWidth="1.5"
      />
      {dies.map((d) => {
        const px = pad + (d.x - bounds.xmin) * cell
        const py = pad + (d.y - bounds.ymin) * cell
        let fill = d.pass ? '#dbe4ee' : '#c8372d'
        if (overlayMode === 'gradcam') {
          const g = d.gradcam || 0
          if (g > 0.25) {
            fill = g > 0.65 ? '#d97706' : '#fbbf24'
          } else {
            fill = d.pass ? '#edf2f7' : '#e17066'
          }
        } else if (overlayMode === 'outlier') {
          if (d.outlier_reason) {
            fill = '#7c3aed'
          } else {
            fill = d.pass ? '#e2e8f0' : '#f3b7b3'
          }
        }
        const isSel = selectedDie && selectedDie.x === d.x && selectedDie.y === d.y
        return (
          <rect
            key={`${d.x}-${d.y}`}
            x={px + 0.7}
            y={py + 0.7}
            width={Math.max(2, cell - 1.4)}
            height={Math.max(2, cell - 1.4)}
            rx={1.5}
            fill={fill}
            stroke={isSel ? '#111827' : 'none'}
            strokeWidth={isSel ? 1.8 : 0}
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectDie(d)}
          />
        )
      })}
    </svg>
  )
}
