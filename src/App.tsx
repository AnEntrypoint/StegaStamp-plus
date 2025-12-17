import React, { useState } from 'react';
import { Encoder } from './components/Encoder';
import { Decoder } from './components/Decoder';
import './styles.css';

function App() {
  const [mode, setMode] = useState<'encoder' | 'decoder'>('encoder');
  const [useWebGPU, setUseWebGPU] = useState(false);

  return (
    <div className="app">
      <header className="header">
        <div className="container header-content">
          <div>
            <h1>StegaStamp</h1>
            <p>Invisible Hyperlinks in Physical Photographs</p>
          </div>
          <div className="header-controls">
            <label>
              <input
                type="checkbox"
                checked={useWebGPU}
                onChange={(e) => setUseWebGPU(e.target.checked)}
              />
              Use WebGPU
            </label>
          </div>
        </div>
      </header>

      <nav className="nav">
        <div className="container">
          <button
            onClick={() => setMode('encoder')}
            className={`nav-btn ${mode === 'encoder' ? 'active' : ''}`}
          >
            Encoder
          </button>
          <button
            onClick={() => setMode('decoder')}
            className={`nav-btn ${mode === 'decoder' ? 'active' : ''}`}
          >
            Decoder
          </button>
        </div>
      </nav>

      <main className="container main-content">
        {mode === 'encoder' ? <Encoder useWebGPU={useWebGPU} /> : <Decoder useWebGPU={useWebGPU} />}
      </main>

      <footer className="footer">
        <div className="container">
          <p>
            StegaStamp: Invisible Hyperlinks in Physical Photographs (CVPR 2020)
          </p>
          <p style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.5rem' }}>
            Matthew Tancik, Ben Mildenhall, Ren Ng - UC Berkeley
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
