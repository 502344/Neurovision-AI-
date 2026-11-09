import { useCallback, useState } from 'react'
import Nav from './components/Nav.jsx'
import Toasts from './components/Toasts.jsx'
import { useRoute } from './hooks/useApi.js'
import About from './pages/About.jsx'
import Analysis from './pages/Analysis.jsx'
import Dataset from './pages/Dataset.jsx'
import Home from './pages/Home.jsx'
import Performance from './pages/Performance.jsx'
import Results from './pages/Results.jsx'
import Disclaimer from './components/Disclaimer.jsx'

export default function App() {
  const [route, go] = useRoute('home')
  const [result, setResult] = useState(null)
  const [toasts, setToasts] = useState([])

  const notify = useCallback((message, kind = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t, { id, message, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 8000)
  }, [])

  const pages = {
    home: <Home go={go} />,
    analysis: <Analysis onResult={setResult} go={go} notify={notify} />,
    results: <Results result={result} go={go} />,
    performance: <Performance />,
    dataset: <Dataset />,
    about: <About />,
  }
  return (
    <>
      <Nav route={route} go={go} />
      {pages[route] ?? pages.home}
      <footer className="footer"><Disclaimer compact /></footer>
      <Toasts items={toasts} dismiss={(id) => setToasts((t) => t.filter((x) => x.id !== id))} />
    </>
  )
}
