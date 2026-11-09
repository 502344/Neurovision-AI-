import { useEffect, useState } from 'react'
import Disclaimer from '../components/Disclaimer.jsx'
import StatusBanner from '../components/StatusBanner.jsx'
import UploadZone from '../components/UploadZone.jsx'
import { api, predict } from '../services/api.js'

export default function Analysis({ onResult, go, notify }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | uploading | analysing
  const [progress, setProgress] = useState(0)
  const [modelState, setModelState] = useState(null)

  useEffect(() => { api.health().then(setModelState).catch(() => setModelState({ offline: true })) }, [])
  useEffect(() => {
    if (!file) return setPreview(null)
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const modelsMissing = modelState && !modelState.offline && modelState.models_available === false
  const busy = phase !== 'idle'

  const run = async () => {
    setPhase('uploading'); setProgress(0)
    try {
      const result = await predict(file, (p) => { setProgress(p); if (p >= 1) setPhase('analysing') })
      onResult(result); go('results')
    } catch (e) {
      notify(e.message, 'error')
    } finally { setPhase('idle') }
  }

  return (
    <main className="page">
      <h1 className="page-title">Analysis</h1>
      <Disclaimer compact />
      {modelState?.offline && <StatusBanner tone="error">The backend is not reachable. Start it with <code>uvicorn backend.main:app --reload</code>.</StatusBanner>}
      {modelsMissing && (
        <StatusBanner>
          The trained model is not available. Please complete model training before using prediction.
          <br />MODEL STATUS: Training dataset required.
        </StatusBanner>
      )}
      <div className="analysis-grid">
        <section className="card">
          <h2>Upload retinal fundus image</h2>
          <UploadZone onFile={setFile} onReject={(m) => notify(m, 'error')} disabled={busy} />
        </section>
        <section className="card preview">
          <h2>Image preview</h2>
          {preview ? <img src={preview} alt="Selected retinal fundus" /> : <div className="placeholder">No image selected</div>}
          {file && <p className="muted">{(file.size / 1024).toFixed(0)} KB · {file.type}</p>}
          {busy && (
            <div className="progress" aria-live="polite">
              <div className={`bar ${phase === 'analysing' ? 'indeterminate' : ''}`} style={phase === 'uploading' ? { width: `${Math.round(progress * 100)}%` } : undefined} />
              <span>{phase === 'uploading' ? `Uploading… ${Math.round(progress * 100)}%` : 'Analysing image (validation → features → classification → Grad-CAM → report)…'}</span>
            </div>
          )}
          <div className="actions">
            <button className="primary" disabled={!file || busy || modelsMissing || modelState?.offline} onClick={run}>Analyze</button>
            <button className="ghost" disabled={!file || busy} onClick={() => setFile(null)}>Clear</button>
          </div>
        </section>
      </div>
    </main>
  )
}
