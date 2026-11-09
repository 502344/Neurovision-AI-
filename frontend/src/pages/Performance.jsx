import Metric, { num, pct } from '../components/Metric.jsx'
import StatusBanner, { TRAINING_REQUIRED } from '../components/StatusBanner.jsx'
import { useApi } from '../hooks/useApi.js'
import { abs, api } from '../services/api.js'

function ConfusionMatrix({ cm }) {
  const cells = [['True Non-ASD', cm.tn, cm.fp], ['True ASD', cm.fn, cm.tp]]
  const max = Math.max(cm.tn, cm.fp, cm.fn, cm.tp, 1)
  return (
    <table className="cm">
      <thead><tr><th /><th>Pred. Non-ASD</th><th>Pred. ASD</th></tr></thead>
      <tbody>{cells.map(([l, a, b]) => (
        <tr key={l}><th>{l}</th>{[a, b].map((v, i) => (
          <td key={i} style={{ background: `rgba(15,58,69,${0.08 + 0.55 * v / max})`, color: v / max > 0.5 ? '#fff' : 'inherit' }}>{v}</td>))}</tr>))}
      </tbody>
    </table>
  )
}

export default function Performance() {
  const { data, error, loading } = useApi(api.metrics)
  return (
    <main className="page">
      <h1 className="page-title">Model Performance</h1>
      {loading && <p className="muted">Loading…</p>}
      {error && <StatusBanner tone="error">{error}</StatusBanner>}
      {data && !data.available && <StatusBanner>{data.message || TRAINING_REQUIRED}</StatusBanner>}
      {data?.available && (() => {
        const f = data.focus_metrics, ev = data.evaluation
        return (<>
          <p className="muted">Held-out test set: n = {ev.test_n} (ASD = {ev.test_ASD}, Non-ASD = {ev.test_Non_ASD}); test set not re-balanced; identical for every model. Focus model: {data.focus_model}.</p>
          {ev.dataset_report?.warnings?.map((w) => <StatusBanner key={w}>{w}</StatusBanner>)}
          {f && <section className="metrics-row">
            <Metric label="Accuracy" value={pct(f.Accuracy)} /><Metric label="Precision" value={pct(f.Precision)} />
            <Metric label="Recall" value={pct(f.Recall)} /><Metric label="F1-score" value={pct(f.F1)} />
            <Metric label="ROC-AUC" value={num(f['ROC-AUC'])} />
          </section>}
          <section className="card">
            <h2>Model comparison</h2>
            <div className="scroll"><table className="table"><thead><tr>
              {['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'ROC-AUC 95% CI'].map((h) => <th key={h}>{h}</th>)}</tr></thead>
              <tbody>{data.comparison.map((r) => { const ci = ev.roc_auc_bootstrap_95ci?.[r.Model]; return (
                <tr key={r.Model} className={r.Model === data.focus_model ? 'focus' : ''}>
                  <td>{r.Model}</td><td>{pct(r.Accuracy)}</td><td>{pct(r.Precision)}</td><td>{pct(r.Recall)}</td>
                  <td>{pct(r.F1)}</td><td>{num(r['ROC-AUC'])}</td><td>{ci ? `${num(ci[0])} – ${num(ci[1])}` : '—'}</td></tr>) })}</tbody></table></div>
            <p className="muted small">{ev.caveats?.join(' ')}</p>
          </section>
          <section className="grid2">
            {data.confusion_matrix && <div className="card"><h2>Confusion matrix</h2><ConfusionMatrix cm={data.confusion_matrix} /></div>}
            {data.plots['roc_curve.png'] && <div className="card"><h2>ROC curve</h2><img className="plot" src={abs(data.plots['roc_curve.png'])} alt="ROC curves" /></div>}
            {data.plots['precision_recall_curve.png'] && <div className="card"><h2>Precision–recall</h2><img className="plot" src={abs(data.plots['precision_recall_curve.png'])} alt="PR curves" /></div>}
            {data.plots['training_history.png'] && <div className="card"><h2>CNN training history</h2><img className="plot" src={abs(data.plots['training_history.png'])} alt="Training history" /></div>}
          </section>
        </>)
      })()}
    </main>
  )
}
