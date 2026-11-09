export const pct = (v) => (v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(2)}%`)
export const num = (v) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(3))

export default function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}
