import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import './styles/App.css';

// API service
import { fetchSymbols } from './services/api';

function App() {
  const [symbols, setSymbols] = useState([]);
  const [currentSymbol, setCurrentSymbol] = useState('BTC/USD');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Base URL for API
  const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    const loadSymbols = async () => {
      try {
        setIsLoading(true);
        const data = await fetchSymbols(apiBaseUrl);
        
        if (data.success && data.symbols) {
          setSymbols(data.symbols);
          // Set default symbol if available
          if (data.symbols.length > 0 && !currentSymbol) {
            setCurrentSymbol(data.symbols[0].symbol);
          }
        } else {
          setError('Failed to load symbols');
        }
      } catch (err) {
        console.error('Error loading symbols:', err);
        setError('Failed to connect to API');
      } finally {
        setIsLoading(false);
      }
    };

    loadSymbols();
  }, [apiBaseUrl, currentSymbol]);

  const handleSymbolChange = (newSymbol) => {
    setCurrentSymbol(newSymbol);
  };

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Crypto Trading Platform</h1>
          <nav className="App-nav">
            <ul>
              <li>
                <Link to="/">Dashboard</Link>
              </li>
              <li>
                <Link to="/backtest">Backtesting</Link>
              </li>
              <li>
                <Link to="/portfolio">Portfolio</Link>
              </li>
              <li>
                <Link to="/settings">Settings</Link>
              </li>
            </ul>
          </nav>
        </header>
        <main className="App-main">
          {error && <div className="error-message">{error}</div>}
          
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  symbols={symbols}
                  currentSymbol={currentSymbol}
                  onSymbolChange={handleSymbolChange}
                  apiBaseUrl={apiBaseUrl}
                  isLoading={isLoading}
                />
              } 
            />
            <Route path="/backtest" element={<div>Backtest Page (Coming Soon)</div>} />
            <Route path="/portfolio" element={<div>Portfolio Page (Coming Soon)</div>} />
            <Route path="/settings" element={<div>Settings Page (Coming Soon)</div>} />
          </Routes>
        </main>
        <footer className="App-footer">
          <p>Crypto Trading Platform &copy; {new Date().getFullYear()}</p>
          <p>Powered by FastAPI and React</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;