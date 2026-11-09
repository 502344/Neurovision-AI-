import Disclaimer from '../components/Disclaimer.jsx'
import { useApi } from '../hooks/useApi.js'
import { api } from '../services/api.js'

export default function About() {
  const { data } = useApi(api.modelInfo)
  return (
    <main className="page">
      <h1 className="page-title">About</h1>
      <Disclaimer />
      <section className="card"><h2>Research gap</h2>
        <p>Existing ASD screening depends on behavioural assessment. Retinal imaging is non-invasive and has been
          explored as a possible correlate, but published work is limited by small datasets, single-architecture models
          and little interpretability. NEUROVISION AI studies whether combining CNN and Vision Transformer features with
          classical machine-learning classifiers, plus visual explanation, can support screening research.
          The project specification document was not supplied to the developer of this software; this text is drawn from the
          project brief and should be replaced with your own literature-review wording.</p></section>
      <section className="card"><h2>Methodology</h2>
        <p>{data?.architecture ?? 'Retinal Fundus Image → Preprocessing → CNN + ViT → Fusion → Random Forest → XGBoost → ASD/Non-ASD → Severity → Grad-CAM → Report'}</p>
        {data?.pipeline && <p className="muted small">Trained pipeline: image size {data.pipeline.image_size}, fusion “{data.pipeline.fusion_method}”, {data.pipeline.selected_k} of {data.pipeline.fused_dim} fused features selected, training set {data.pipeline.train_size} images, random state {data.pipeline.random_state}.</p>}
        {data && !data.available && <p className="muted small">Models are not trained yet.</p>}</section>
      <section className="card"><h2>Limitations</h2>
        <ul className="plain">
          <li>Research prototype for screening research — never a diagnosis and never a substitute for clinical assessment.</li>
          <li>Performance depends entirely on the supplied labelled dataset; small or single-site data limit generalisation.</li>
          <li>Without patient IDs, patient-level leakage cannot be completely excluded.</li>
          <li>No external or clinical validation has been performed.</li>
          <li>Grad-CAM shows where the CNN component looked; it does not identify clinically established biomarkers.</li>
          <li>Feature importance and SHAP values describe the model’s mathematics, not medical causes.</li>
          <li>Severity is estimated only when validated severity labels were available for training.</li>
        </ul></section>
    </main>
  )
}
