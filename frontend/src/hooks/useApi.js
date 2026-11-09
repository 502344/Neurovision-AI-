import { useEffect, useState } from 'react'

/** Fetch once on mount; returns { data, error, loading }. */
export function useApi(fn) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    fn().then((data) => alive && setState({ data, error: null, loading: false }))
        .catch((error) => alive && setState({ data: null, error: error.message, loading: false }))
    return () => { alive = false }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return state
}

/** Minimal hash router: returns the current route name. */
export function useRoute(defaultRoute = 'home') {
  const read = () => window.location.hash.replace('#/', '') || defaultRoute
  const [route, setRoute] = useState(read)
  useEffect(() => {
    const on = () => setRoute(read())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return [route, (r) => { window.location.hash = `#/${r}` }]
}
