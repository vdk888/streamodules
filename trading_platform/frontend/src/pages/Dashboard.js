import React, { useEffect, useState } from 'react';
import api from '../services/api';
import PriceChart from '../components/PriceChart';
import SymbolSelector from '../components/SymbolSelector';
import BacktestResults from '../components/BacktestResults';

const Dashboard = () => {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [lookbackDays, setLookbackDays] = useState(15);
  const [priceData, setPriceData] = useState([]);
  const [signalsData, setSignalsData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [wsConnection, setWsConnection] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [isBacktesting, setIsBacktesting] = useState(false);
  
  // Fetch available symbols
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const response = await api.getCryptoSymbols();
        setSymbols(response.data.symbols);
        
        // Set default selected symbol if available
        if (response.data.symbols.length > 0) {
          setSelectedSymbol(response.data.symbols[0].symbol);
        }
      } catch (err) {
        console.error('Error fetching symbols:', err);
        setError('Failed to load symbols. Please try again later.');
      }
    };
    
    fetchSymbols();
  }, []);
  
  // Fetch price data for selected symbol
  useEffect(() => {
    if (!selectedSymbol) return;
    
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await api.getCryptoData(selectedSymbol, timeframe, lookbackDays);
        
        if (response.data.success) {
          setPriceData(response.data.price_data);
          setSignalsData(response.data.signals_data);
        } else {
          setError(response.data.error || 'Failed to fetch data');
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
    
    // Set up WebSocket connection for real-time updates
    const ws = api.connectWebSocket(selectedSymbol, (data) => {
      if (data.success) {
        setPriceData(data.price_data);
        setSignalsData(data.signals_data);
      }
    });
    
    setWsConnection(ws);
    
    // Clean up WebSocket connection when component unmounts or symbol changes
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, [selectedSymbol, timeframe, lookbackDays]);
  
  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
    setBacktestResults(null); // Reset backtest results when symbol changes
  };
  
  const handleTimeframeChange = (e) => {
    setTimeframe(e.target.value);
  };
  
  const handleLookbackDaysChange = (e) => {
    setLookbackDays(parseInt(e.target.value, 10));
  };
  
  const handleRunBacktest = async () => {
    setIsBacktesting(true);
    setError(null);
    
    try {
      const response = await api.runCryptoBacktest(selectedSymbol, lookbackDays);
      
      if (response.data.success) {
        setBacktestResults(response.data.backtest_result);
      } else {
        setError(response.data.error || 'Failed to run backtest');
      }
    } catch (err) {
      console.error('Error running backtest:', err);
      setError('Failed to run backtest. Please try again later.');
    } finally {
      setIsBacktesting(false);
    }
  };
  
  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Crypto Trading Platform</h1>
      </header>
      
      <div className="controls-section">
        <div className="symbol-control">
          <SymbolSelector
            symbols={symbols}
            selectedSymbol={selectedSymbol}
            onSymbolChange={handleSymbolChange}
          />
        </div>
        
        <div className="timeframe-control">
          <label htmlFor="timeframe-select">Timeframe:</label>
          <select
            id="timeframe-select"
            value={timeframe}
            onChange={handleTimeframeChange}
            className="form-select"
          >
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>
        
        <div className="lookback-control">
          <label htmlFor="lookback-input">Lookback Days:</label>
          <input
            id="lookback-input"
            type="number"
            min="1"
            max="90"
            value={lookbackDays}
            onChange={handleLookbackDaysChange}
            className="form-input"
          />
        </div>
        
        <div className="actions-control">
          <button
            className="btn backtest-btn"
            onClick={handleRunBacktest}
            disabled={isBacktesting}
          >
            {isBacktesting ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </div>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="chart-container">
        {isLoading ? (
          <div className="loading">Loading price data...</div>
        ) : (
          <PriceChart
            priceData={priceData}
            signals={signalsData}
            symbol={selectedSymbol}
          />
        )}
      </div>
      
      {backtestResults && (
        <div className="backtest-section">
          <BacktestResults results={backtestResults} />
        </div>
      )}
    </div>
  );
};

export default Dashboard;