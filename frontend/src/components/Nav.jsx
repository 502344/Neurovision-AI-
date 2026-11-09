const ITEMS = [
  ['home', 'Home'], ['analysis', 'Analysis'], ['results', 'Results'],
  ['performance', 'Model Performance'], ['dataset', 'Dataset'], ['about', 'About'],
]

export default function Nav({ route, go }) {
  return (
    <header className="nav">
      <a className="brand" href="#/home" onClick={() => go('home')}>
        <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
          <circle cx="13" cy="13" r="11.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <circle cx="13" cy="13" r="5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <circle cx="13" cy="13" r="1.8" fill="var(--accent)" />
        </svg>
        NEUROVISION AI
      </a>
      <nav aria-label="Main">
        {ITEMS.map(([id, label]) => (
          <a key={id} href={`#/${id}`} className={route === id ? 'active' : ''}>{label}</a>
        ))}
      </nav>
    </header>
  )
}
