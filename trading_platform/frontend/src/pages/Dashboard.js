import React, { useState, useEffect } from 'react';
import SymbolSelector from '../components/SymbolSelector';
import PriceChart from '../components/PriceChart';
import BacktestResults from '../components/BacktestResults';

function Dashboard({ symbols, currentSymbol, onSymbolChange, apiBaseUrl, isLoading }) {
  // State variables
  const [timeframe, setTimeframe] = useState('1h');
  const [lookbackDays, setLookbackDays] = useState(15);
  const [priceData, setPriceData] = useState([]);
  const [signalsData, setSignalsData] = useState([]);
  const [isDataLoading, setIsDataLoading] = useState(false);
  const [error, setError] = useState(null);
  const [performanceRanking, setPerformanceRanking] = useState([]);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [webSocket, setWebSocket] = useState(null);

  // Fetch cryptocurrency data
  useEffect(() => {
    const fetchData = async () => {
      if (!currentSymbol) return;

      setIsDataLoading(true);
      try {
        const response = await fetch(
          `${apiBaseUrl}/crypto/data/${encodeURIComponent(currentSymbol)}?timeframe=${timeframe}&lookback_days=${lookbackDays}`
        );

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
          setPriceData(data.price_data || []);
          setSignalsData(data.signals_data || []);
        } else {
          throw new Error(data.error || 'Failed to fetch cryptocurrency data');
        }
      } catch (err) {
        setError(err.message);
        console.error('Error fetching cryptocurrency data:', err);
      } finally {
        setIsDataLoading(false);
      }
    };

    fetchData();
  }, [currentSymbol, timeframe, lookbackDays, apiBaseUrl]);

  // Fetch performance ranking
  useEffect(() => {
    const fetchRanking = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/crypto/ranking?lookback_days=${lookbackDays}`);

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success && data.ranking) {
          setPerformanceRanking(data.ranking);
        } else {
          throw new Error(data.error || 'Failed to fetch performance ranking');
        }
      } catch (err) {
        console.error('Error fetching performance ranking:', err);
      }
    };

    fetchRanking();
  }, [lookbackDays, apiBaseUrl]);

  // Setup WebSocket connection
  useEffect(() => {
    if (!currentSymbol) return;

    // Close existing connection if any
    if (webSocket) {
      webSocket.close();
    }

    // WebSocket Host - replace with your actual domain in production
    const wsHost = window.location.hostname === 'localhost' 
      ? 'ws://localhost:5000' 
      : `wss://${window.location.host}`;
    
    const ws = new WebSocket(`${wsHost}/ws/crypto/${encodeURIComponent(currentSymbol)}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsWebSocketConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.success) {
          setPriceData(data.price_data || []);
          setSignalsData(data.signals_data || []);
        } else {
          console.error('WebSocket error:', data.error);
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsWebSocketConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsWebSocketConnected(false);
    };

    setWebSocket(ws);

    // Cleanup on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [currentSymbol]);

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
  };

  // Handle lookback days change
  const handleLookbackDaysChange = (newLookbackDays) => {
    setLookbackDays(newLookbackDays);
  };

  // Find current symbol ranking
  const currentSymbolRanking = performanceRanking.find(
    (item) => item.symbol === currentSymbol
  );

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <SymbolSelector
          symbols={symbols}
          currentSymbol={currentSymbol}
          onSymbolChange={onSymbolChange}
          isLoading={isLoading}
        />
        
        <div className="timeframe-selector">
          <label htmlFor="timeframe">Timeframe:</label>
          <select
            id="timeframe"
            value={timeframe}
            onChange={(e) => handleTimeframeChange(e.target.value)}
          >
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>
        
        <div className="lookback-selector">
          <label htmlFor="lookback">Lookback Days:</label>
          <select
            id="lookback"
            value={lookbackDays}
            onChange={(e) => handleLookbackDaysChange(parseInt(e.target.value))}
          >
            <option value="7">7 Days</option>
            <option value="15">15 Days</option>
            <option value="30">30 Days</option>
          </select>
        </div>
        
        <div className="connection-status">
          <span className={`status-indicator ${isWebSocketConnected ? 'connected' : 'disconnected'}`}></span>
          <span className="status-text">{isWebSocketConnected ? 'Live' : 'Offline'}</span>
        </div>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="dashboard-content">
        <div className="chart-container">
          <h2>Price Chart: {currentSymbol}</h2>
          {isDataLoading ? (
            <div className="loading-indicator">Loading chart data...</div>
          ) : (
            <PriceChart
              priceData={priceData}
              signalsData={signalsData}
              symbol={currentSymbol}
              timeframe={timeframe}
            />
          )}
        </div>
        
        <div className="performance-container">
          <h2>Performance Metrics</h2>
          {currentSymbolRanking ? (
            <div className="performance-metrics">
              <div className="metric">
                <span className="metric-label">Rank:</span>
                <span className="metric-value">{currentSymbolRanking.rank}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Performance:</span>
                <span className={`metric-value ${currentSymbolRanking.performance >= 0 ? 'positive' : 'negative'}`}>
                  {currentSymbolRanking.performance.toFixed(2)}%
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Current Price:</span>
                <span className="metric-value">${currentSymbolRanking.end_price.toFixed(2)}</span>
              </div>
            </div>
          ) : (
            <div className="loading-indicator">Loading performance data...</div>
          )}
        </div>
      </div>
      
      <div className="trading-signals">
        <h2>Trading Signals</h2>
        {signalsData && signalsData.length > 0 ? (
          <div className="signals-table-container">
            <table className="signals-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Price</th>
                  <th>Signal</th>
                  <th>Indicator Value</th>
                </tr>
              </thead>
              <tbody>
                {signalsData.slice(-10).map((signal, index) => {
                  // Determine signal type
                  let signalType = 'None';
                  let signalClass = '';
                  
                  if (signal.buy_signal) {
                    signalType = 'Buy';
                    signalClass = 'buy-signal';
                  } else if (signal.sell_signal) {
                    signalType = 'Sell';
                    signalClass = 'sell-signal';
                  } else if (signal.potential_buy) {
                    signalType = 'Potential Buy';
                    signalClass = 'potential-buy';
                  } else if (signal.potential_sell) {
                    signalType = 'Potential Sell';
                    signalClass = 'potential-sell';
                  }
                  
                  return (
                    <tr key={index} className={signalClass}>
                      <td>{new Date(signal.timestamp || signal.date).toLocaleString()}</td>
                      <td>${signal.price.toFixed(2)}</td>
                      <td className={signalClass}>{signalType}</td>
                      <td>{signal.daily_composite.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="no-data">No signal data available</div>
        )}
      </div>
      
      <div className="ranking-table">
        <h2>Performance Ranking</h2>
        {performanceRanking && performanceRanking.length > 0 ? (
          <div className="ranking-table-container">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Symbol</th>
                  <th>Performance</th>
                  <th>Current Price</th>
                </tr>
              </thead>
              <tbody>
                {performanceRanking
                  .sort((a, b) => a.rank - b.rank)
                  .slice(0, 10)
                  .map((item) => (
                    <tr 
                      key={item.symbol} 
                      className={item.symbol === currentSymbol ? 'current-symbol' : ''}
                      onClick={() => onSymbolChange(item.symbol)}
                    >
                      <td>{item.rank}</td>
                      <td>{item.symbol}</td>
                      <td className={item.performance >= 0 ? 'positive' : 'negative'}>
                        {item.performance.toFixed(2)}%
                      </td>
                      <td>${item.end_price.toFixed(2)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="loading-indicator">Loading ranking data...</div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;