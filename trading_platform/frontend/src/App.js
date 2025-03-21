import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './styles/App.css';

// Import components
import Dashboard from './pages/Dashboard';

// API URLs
const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  // State for app-wide data
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [symbols, setSymbols] = useState([]);
  const [currentSymbol, setCurrentSymbol] = useState('BTC/USD');

  // Fetch available symbols when app loads
  useEffect(() => {
    const fetchSymbols = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/crypto/symbols`);
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        if (data.success && data.symbols) {
          setSymbols(data.symbols);
          // Set default symbol if available
          if (data.symbols.length > 0 && !currentSymbol) {
            setCurrentSymbol(data.symbols[0].symbol);
          }
        } else {
          throw new Error('Failed to fetch symbols');
        }
      } catch (error) {
        setError(error.message);
        console.error('Error fetching symbols:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSymbols();
  }, [currentSymbol]);

  // Handle symbol change
  const handleSymbolChange = (symbol) => {
    setCurrentSymbol(symbol);
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
                <Link to="/portfolio">Portfolio</Link>
              </li>
              <li>
                <Link to="/backtest">Backtest</Link>
              </li>
              <li>
                <Link to="/optimize">Optimize</Link>
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
                  apiBaseUrl={API_BASE_URL}
                  isLoading={isLoading}
                />
              } 
            />
            <Route path="/portfolio" element={<div>Portfolio Page (Coming Soon)</div>} />
            <Route path="/backtest" element={<div>Backtest Page (Coming Soon)</div>} />
            <Route path="/optimize" element={<div>Optimization Page (Coming Soon)</div>} />
          </Routes>
        </main>
        
        <footer className="App-footer">
          <p>&copy; {new Date().getFullYear()} Crypto Trading Platform</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;