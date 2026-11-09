import Disclaimer from '../components/Disclaimer.jsx'
import FeatureChart from '../components/FeatureChart.jsx'
import { abs } from '../services/api.js'

export default function Results({ result, go }) {
  if (!result) {
    return (
      <main className="page">
        <h1 className="page-title">Results</h1>
        <div className="card"><p>No analysis has been run in this session.</p>
          <button className="primary" onClick={() => go('analysis')}>Go to Analysis</button></div>
      </main>
    )
  }
  const sev = result.severity_available && result.severity ? result.severity : 'Not Available'
  return (
    <main className="page">
      <h1 className="page-title">Results</h1>
      <p className="muted">AI-assisted screening prediction · processed in {result.processing_time}s · model: {result.model}</p>
      <section className="result-summary card">
        <div><span className="k">Prediction</span><span className={`v big ${result.prediction === 'ASD' ? 'asd' : 'non'}`}>{result.prediction}</span></div>
        <div><span className="k">Confidence</span><span className="v big">{(result.confidence * 100).toFixed(2)}%</span></div>
        <div><span className="k">Severity</span><span className="v big sev">{sev}</span></div>
      </section>
      <p className="statement">{result.prediction_statement}</p>
      {!result.severity_available && <p className="muted">{result.severity_note}</p>}
      {result.severity_available && !result.severity && <p className="muted">{result.severity_note}</p>}
      {result.quality_warnings?.length > 0 && (
        <div className="status warn">Image quality notes: {result.quality_warnings.join(' ')}</div>
      )}
      <section className="card">
        <div className="pair">
          <figure><img src={result.original_b64} alt="Preprocessed input" /><figcaption>Original image (preprocessed)</figcaption></figure>
          <figure><img src={result.gradcam_overlay_b64} alt="Grad-CAM overlay" /><figcaption>Grad-CAM overlay (CNN component)</figcaption></figure>
        </div>
        <p className="muted small">{result.gradcam_note}</p>
        <p className="muted small">CNN head P(ASD) = {(result.cnn_head_probability_asd * 100).toFixed(2)}% · fused-model P(ASD) = {(result.probability_asd * 100).toFixed(2)}%</p>
      </section>
      <FeatureChart features={result.shap_features} />
      <div className="actions">
        {result.report_url && <a className="primary as-button" href={abs(result.report_url)} target="_blank" rel="noreferrer">Download PDF report</a>}
        <button className="ghost" onClick={() => go('analysis')}>Analyse another image</button>
      </div>
      <Disclaimer />
    </main>
  )
}
