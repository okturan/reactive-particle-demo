import { useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

function ReactiveParticleApp() {
  const bootedRef = useRef(false);

  useEffect(() => {
    if (bootedRef.current) {
      return;
    }

    bootedRef.current = true;
    let cancelled = false;

    import('./main.js').then((engine) => {
      if (!cancelled) {
        engine.bootParticleEngine();
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  return <App />;
}

createRoot(document.querySelector('#root')).render(<ReactiveParticleApp />);
