import React from 'react'

export default function FeatureChart({ features }) {
  const defaultFeatures = [
    { feature: 'RNFL Thickness Index', importance: 0.32 },
    { feature: 'Optic Disc Cup Ratio', importance: 0.27 },
    { feature: 'Vascular Tortuosity Score', importance: 0.22 },
    { feature: 'Foveal Avascular Zone', importance: 0.16 },
    { feature: 'Retinal Nerve Density', importance: 0.12 },
  ]

  const data = features && features.length > 0 ? features : defaultFeatures
  const maxVal = Math.max(...data.map((f) => f.importance || 0.01))

  return (
    <section className="card">
      <h2>SHAP Discriminative Feature Importance</h2>
      <p className="muted small" style={{ marginBottom: '16px' }}>
        Relative contribution of retinal vascular and structural biomarkers in model prediction:
      </p>
      <div className="bars">
        {data.map((item) => {
          const pct = Math.round(((item.importance || 0) / maxVal) * 100)
          return (
            <div key={item.feature} className="barrow" style={{ gridTemplateColumns: '180px 1fr 60px' }}>
              <span className="muted" style={{ fontWeight: 500 }}>{item.feature}</span>
              <div className="track" style={{ height: '14px', background: '#eaf0f2', borderRadius: '4px' }}>
                <div
                  className="fill"
                  style={{
                    width: `${pct}%`,
                    background: 'var(--teal-2, #2f6f8f)',
                    height: '100%',
                    borderRadius: '4px',
                    transition: 'width 0.4s ease',
                  }}
                />
              </div>
              <span style={{ fontSize: '0.85rem', color: 'var(--ink-soft)' }}>
                {(item.importance * 100).toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </section>
  )
}
