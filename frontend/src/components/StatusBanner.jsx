export default function StatusBanner({ children, tone = 'warn' }) {
  return <div className={`status ${tone}`} role="status">{children}</div>
}

export const TRAINING_REQUIRED =
  'MODEL STATUS: Training dataset required. No experimental performance values are displayed until real labelled data are supplied and evaluated.'
