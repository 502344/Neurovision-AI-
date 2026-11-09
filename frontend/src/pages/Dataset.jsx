import StatusBanner from '../components/StatusBanner.jsx'
import { useApi } from '../hooks/useApi.js'
import { api } from '../services/api.js'

function Bars({ data }) {
  const entries = Object.entries(data || {})
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1
  return (
    <div className="bars">{entries.map(([k, v]) => (
      <div className="barrow" key={k}><span>{k}</span>
        <div className="track"><div className="fill" style={{ width: `${(100 * v) / total}%` }} /></div>
        <span>{v} ({((100 * v) / total).toFixed(1)}%)</span></div>))}
    </div>
  )
}

export default function Dataset() {
  const { data, error, loading } = useApi(api.datasetInfo)
  return (
    <main className="page">
      <h1 className="page-title">Dataset</h1>
      {loading && <p className="muted">Loading…</p>}
      {error && <StatusBanner tone="error">{error}</StatusBanner>}
      {data && !data.available && <StatusBanner>{data.message}</StatusBanner>}
      {data?.available && (<>
        {data.warnings?.map((w) => <StatusBanner key={w}>{w}</StatusBanner>)}
        <section className="metrics-row">
          <div className="metric"><span className="metric-value">{data.valid_images}</span><span className="metric-label">valid images</span></div>
          <div className="metric"><span className="metric-value">{data.ASD_count}</span><span className="metric-label">ASD</span></div>
          <div className="metric"><span className="metric-value">{data.Non_ASD_count}</span><span className="metric-label">Non-ASD</span></div>
          <div className="metric"><span className="metric-value">{data.patient_count ?? 'n/a'}</span><span className="metric-label">patients</span></div>
        </section>
        <section className="grid2">
          <div className="card"><h2>Class distribution</h2><Bars data={{ ASD: data.ASD_count, 'Non-ASD': data.Non_ASD_count }} /></div>
          <div className="card"><h2>Severity distribution (ASD images)</h2>
            <Bars data={data.severity_distribution} />
            {(!data.severity_distribution || Object.keys(data.severity_distribution).every((k) => k === 'Unlabelled')) &&
              <p className="muted small">No validated severity labels: severity prediction is unavailable.</p>}</div>
        </section>
        <section className="card">
          <h2>Data integrity</h2>
          <ul className="plain">
            <li>Images listed: {data.total_images_listed}; corrupted: {data.corrupted_images}; duplicates removed: {data.duplicate_images_removed}; missing files: {data.missing_files}</li>
            <li>Image dimensions (min / median / max): {['min', 'median', 'max'].map((k) => data.image_dimensions?.[k]?.join('×') ?? '—').join(' / ')}</li>
            <li>Split strategy: {data.split_strategy}</li>
            <li>Split counts: {Object.entries(data.split_counts || {}).map(([s, c]) => `${s}: ASD ${c.ASD} / Non-ASD ${c.Non_ASD}`).join(' · ')}</li>
          </ul>
        </section>
      </>)}
    </main>
  )
}
