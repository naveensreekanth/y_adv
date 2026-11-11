import React, { useRef, useState } from 'react'
import { UploadCloud, FileSpreadsheet, Play, Download, X } from 'lucide-react'

export default function UploadScreen({
  onUploadFiles,
  onLoadSample,
  loading,
  error,
  availableSamples,
  hasFullDataset,
}) {
  const inputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const [matchContext, setMatchContext] = useState(true)

  const handleSelected = (fileList) => {
    const picked = Array.from(fileList || []).filter((f) =>
      /\.(csv|xlsx|xlsm|xls|tsv)$/i.test(f.name),
    )
    if (picked.length > 0) {
      setFiles((prev) => {
        const seen = new Set(prev.map((p) => p.name + p.size))
        return [...prev, ...picked.filter((p) => !seen.has(p.name + p.size))]
      })
    }
  }

  const removeFile = (idx) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  return (
    <div className="upload-container">
      <div className="upload-header">
        <h1>Yield Advisor</h1>
        <p>
          Drop your semiconductor chip-test CSV or Excel file to find defect patterns,
          root causes, hidden quality risks, and faster testing opportunities.
        </p>
      </div>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          handleSelected(e.dataTransfer.files)
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv,.xlsx,.xlsm,.xls,.tsv"
          style={{ display: 'none' }}
          onChange={(e) => handleSelected(e.target.files)}
        />
        <div className="dropzone-icon">
          <UploadCloud size={22} />
        </div>
        <h3>Drop CSV or Excel (.xlsx) file(s) here</h3>
        <p>
          Supports single test files, multi-sheet Excel workbooks, or multiple CSV tables dropped together.
        </p>

        {files.length > 0 && (
          <div className="file-chips" onClick={(e) => e.stopPropagation()}>
            {files.map((f, idx) => (
              <span key={f.name + idx} className="file-chip">
                <FileSpreadsheet size={13} />
                <span>{f.name}</span>
                <span className="tabular">({(f.size / (1024 * 1024)).toFixed(2)} MB)</span>
                <button type="button" onClick={() => removeFile(idx)} aria-label="Remove file">
                  <X size={13} />
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="upload-options">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={matchContext}
            onChange={(e) => setMatchContext(e.target.checked)}
          />
          <span>
            Auto-match fab equipment &amp; electrical context if uploading sort-only AX7-SOC data
          </span>
        </label>

        <button
          type="button"
          className="btn btn-primary"
          disabled={loading || files.length === 0}
          onClick={() => onUploadFiles(files, matchContext)}
        >
          {loading ? (
            <>
              <span className="spinner" />
              Running ML Models...
            </>
          ) : (
            <>
              <Play size={14} />
              Run Analysis
            </>
          )}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="sample-strip">
        <div className="sample-strip-title">Or test right away with the AX7-SOC dataset</div>
        <div className="sample-buttons">
          <button
            type="button"
            className="btn"
            disabled={loading}
            onClick={() => onLoadSample('full')}
          >
            <Play size={14} />
            {hasFullDataset
              ? 'Run Full AX7-SOC Dataset (40 Lots · 1,000 Wafers)'
              : 'Run Built-in Demo Dataset (8 Lots · 200 Wafers)'}
          </button>

          {availableSamples.map((name) => (
            <a
              key={name}
              href={`/api/samples/${encodeURIComponent(name)}`}
              className="btn btn-sm"
              style={{ textDecoration: 'none' }}
            >
              <Download size={13} />
              {name}
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}
