export const DISCLAIMER =
  'This is an AI-assisted research screening prototype and is not a clinical diagnosis. Results should not be used as a substitute for assessment by qualified healthcare professionals.'

export default function Disclaimer({ compact = false }) {
  return (
    <aside className={`disclaimer ${compact ? 'compact' : ''}`} role="note">
      <strong>Research prototype.</strong> {DISCLAIMER}
    </aside>
  )
}
