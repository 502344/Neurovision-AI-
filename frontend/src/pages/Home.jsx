import Disclaimer from '../components/Disclaimer.jsx'

const FLOW = [
  'Retinal fundus image', 'Preprocessing', 'CNN + ViT feature extraction', 'Feature fusion',
  'Random Forest feature ranking', 'XGBoost classification', 'ASD / Non-ASD',
  'Severity estimation (where labelled data are available)', 'Grad-CAM', 'Automated report',
]
const OBJECTIVES = [
  'Analyse retinal fundus images as a non-invasive input for AI-assisted ASD screening research.',
  'Combine convolutional (CNN) and Vision Transformer (ViT) representations through feature fusion.',
  'Rank fused features with a Random Forest and classify with XGBoost, compared against an SVM.',
  'Estimate ASD severity only where validated severity labels exist.',
  'Provide Grad-CAM visual explanations, a confidence score and an automated report.',
  'Evaluate honestly with Accuracy, Precision, Recall, F1-score and ROC-AUC on a held-out test set.',
]

export default function Home({ go }) {
  return (
    <main className="page">
      <section className="hero">
        <h1>NEUROVISION AI</h1>
        <p className="subtitle">AI-Based Retinal Image Analysis Framework for Early Screening of Autism Spectrum Disorder</p>
        <p className="lede">
          Upload a retinal fundus image and follow it through the pipeline to an interpretable,
          AI-assisted screening prediction — with a confidence score, a Grad-CAM view of the CNN component and a PDF report.
        </p>
        <button className="primary" onClick={() => go('analysis')}>Analyse a retinal image</button>
      </section>
      <Disclaimer />
      <section className="card">
        <h2>Workflow</h2>
        <ol className="flow">{FLOW.map((s) => <li key={s}>{s}</li>)}</ol>
      </section>
      <section className="card">
        <h2>Research objectives</h2>
        <ul className="plain">{OBJECTIVES.map((o) => <li key={o}>{o}</li>)}</ul>
      </section>
    </main>
  )
}
