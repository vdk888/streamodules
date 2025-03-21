import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="logo">
            <Link to="/">Crypto Trading Platform</Link>
          </div>
          <ul className="nav-links">
            <li>
              <Link to="/">Dashboard</Link>
            </li>
            <li>
              <Link to="/backtest">Backtest</Link>
            </li>
            <li>
              <Link to="/portfolio">Portfolio</Link>
            </li>
          </ul>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/backtest" element={<Dashboard />} />
            <Route path="/portfolio" element={<Dashboard />} />
          </Routes>
        </main>
        
        <footer className="footer">
          <p>&copy; {new Date().getFullYear()} Crypto Trading Platform</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;