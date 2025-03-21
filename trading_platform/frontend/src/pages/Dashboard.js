import React, { useState, useEffect, useRef } from 'react';
import { fetchSymbolData, fetchRanking, setupWebSocket } from '../services/api';
import PriceChart from '../components/PriceChart';
import SymbolSelector from '../components/SymbolSelector';

function Dashboard({ symbols, currentSymbol, onSymbolChange, apiBaseUrl, isLoading }) {
  const [timeframe, setTimeframe] = useState('1h');
  const [lookbackDays, setLookbackDays] = useState(15);
  const [priceData, setPriceData] = useState([]);
  const [signalsData, setSignalsData] = useState([]);
  const [rankingData, setRankingData] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [dataLoading, setDataLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  const webSocketRef = useRef(null);

  // Load initial data
  useEffect(() => {
    if (!currentSymbol || !apiBaseUrl) return;

    const loadData = async () => {
      try {
        setDataLoading(true);
        setError(null);

        // Fetch symbol data
        const data = await fetchSymbolData(apiBaseUrl, currentSymbol, timeframe, lookbackDays);
        
        if (data.success) {
          setPriceData(data.price_data || []);
          setSignalsData(data.signals_data || []);
          setLastUpdated(new Date());
        } else {
          setError(data.error || 'Failed to load price data');
        }

        // Fetch ranking data
        const rankingResponse = await fetchRanking(apiBaseUrl, lookbackDays);
        
        if (rankingResponse.success) {
          setRankingData(rankingResponse.ranking || []);
        }
      } catch (err) {
        console.error('Error loading dashboard data:', err);
        setError('Failed to load data. Please try again later.');
      } finally {
        setDataLoading(false);
      }
    };

    loadData();
  }, [apiBaseUrl, currentSymbol, timeframe, lookbackDays]);

  // Setup WebSocket for real-time updates
  useEffect(() => {
    if (!currentSymbol || !apiBaseUrl) return;

    // Disconnect previous WebSocket
    if (webSocketRef.current) {
      webSocketRef.current.close();
    }

    // Setup new WebSocket
    const ws = setupWebSocket(
      apiBaseUrl,
      currentSymbol,
      (data) => {
        if (data.success) {
          setPriceData(data.price_data || []);
          setSignalsData(data.signals_data || []);
          setLastUpdated(new Date());
        } else if (data.error) {
          console.error('WebSocket error:', data.error);
        }
      },
      (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      },
      () => {
        setIsConnected(false);
      }
    );

    webSocketRef.current = ws;
    setIsConnected(true);

    // Cleanup on unmount
    return () => {
      if (webSocketRef.current) {
        webSocketRef.current.close();
      }
    };
  }, [apiBaseUrl, currentSymbol]);

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
  };

  // Handle lookback days change
  const handleLookbackDaysChange = (days) => {
    setLookbackDays(days);
  };

  // Get current symbol ranking
  const getCurrentRanking = () => {
    if (!rankingData.length || !currentSymbol) return null;
    
    const symbolRanking = rankingData.find(item => item.symbol === currentSymbol);
    return symbolRanking || null;
  };

  // Format price with commas for thousands
  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', { 
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  // Calculate price change
  const calculatePriceChange = () => {
    if (!priceData || priceData.length < 2) return { value: 0, percentage: 0 };
    
    const latest = priceData[priceData.length - 1];
    const previous = priceData[priceData.length - 2];
    
    if (!latest || !previous) return { value: 0, percentage: 0 };
    
    const change = latest.close - previous.close;
    const percentage = (change / previous.close) * 100;
    
    return {
      value: change,
      percentage
    };
  };

  // Get latest signals
  const getLatestSignals = () => {
    if (!signalsData || !signalsData.length) return [];
    
    // Get last 10 signals or less if fewer are available
    return signalsData.slice(-10).reverse();
  };

  const currentRanking = getCurrentRanking();
  const priceChange = calculatePriceChange();
  const latestSignals = getLatestSignals();
  const latestPrice = priceData.length > 0 ? priceData[priceData.length - 1] : null;

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
          <label>Timeframe:</label>
          <select 
            value={timeframe}
            onChange={(e) => handleTimeframeChange(e.target.value)}
            disabled={dataLoading}
          >
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>
        
        <div className="lookback-selector">
          <label>Lookback:</label>
          <select 
            value={lookbackDays}
            onChange={(e) => handleLookbackDaysChange(parseInt(e.target.value))}
            disabled={dataLoading}
          >
            <option value="5">5 Days</option>
            <option value="10">10 Days</option>
            <option value="15">15 Days</option>
            <option value="30">30 Days</option>
          </select>
        </div>
        
        <div className="connection-status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="dashboard-content">
        <div className="chart-container">
          {dataLoading ? (
            <div className="loading-indicator">
              <div className="loading-indicator-small"></div>
              <span>Loading data...</span>
            </div>
          ) : (
            <>
              <PriceChart 
                priceData={priceData} 
                signalsData={signalsData}
                symbol={currentSymbol}
                timeframe={timeframe}
              />
              
              <div className="signals-table-container">
                <h3>Recent Signals</h3>
                {latestSignals.length > 0 ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Price</th>
                        <th>Signal</th>
                        <th>Indicator</th>
                      </tr>
                    </thead>
                    <tbody>
                      {latestSignals.map((signal, index) => {
                        let signalType = "None";
                        let rowClass = "";
                        
                        if (signal.buy_signal) {
                          signalType = "BUY";
                          rowClass = "buy-signal";
                        } else if (signal.sell_signal) {
                          signalType = "SELL";
                          rowClass = "sell-signal";
                        } else if (signal.potential_buy) {
                          signalType = "Potential Buy";
                          rowClass = "potential-buy";
                        } else if (signal.potential_sell) {
                          signalType = "Potential Sell";
                          rowClass = "potential-sell";
                        }
                        
                        const timestamp = new Date(signal.timestamp);
                        
                        return (
                          <tr key={index} className={rowClass}>
                            <td>{timestamp.toLocaleString()}</td>
                            <td>${formatPrice(signal.price)}</td>
                            <td>{signalType}</td>
                            <td>{signal.daily_composite.toFixed(4)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <p>No recent signals available.</p>
                )}
              </div>
            </>
          )}
        </div>
        
        <div className="performance-container">
          <h3>Market Stats</h3>
          
          {latestPrice ? (
            <>
              <div className="crypto-symbol">{currentSymbol}</div>
              <div className="crypto-price">${formatPrice(latestPrice.close)}</div>
              
              <div className={priceChange.value >= 0 ? 'positive' : 'negative'}>
                {priceChange.value >= 0 ? '▲' : '▼'} 
                ${Math.abs(priceChange.value).toFixed(2)} 
                ({Math.abs(priceChange.percentage).toFixed(2)}%)
              </div>
              
              <div className="performance-metrics">
                <div className="metric">
                  <div className="metric-label">High</div>
                  <div className="metric-value">
                    ${formatPrice(Math.max(...priceData.map(d => d.high)))}
                  </div>
                </div>
                
                <div className="metric">
                  <div className="metric-label">Low</div>
                  <div className="metric-value">
                    ${formatPrice(Math.min(...priceData.map(d => d.low)))}
                  </div>
                </div>
                
                <div className="metric">
                  <div className="metric-label">24h Volume</div>
                  <div className="metric-value">
                    ${formatPrice(priceData.slice(-24).reduce((sum, d) => sum + d.volume, 0))}
                  </div>
                </div>
                
                <div className="metric">
                  <div className="metric-label">Updated</div>
                  <div className="metric-value">
                    {lastUpdated ? lastUpdated.toLocaleTimeString() : 'N/A'}
                  </div>
                </div>
              </div>
              
              {currentRanking && (
                <>
                  <h3>Performance Ranking</h3>
                  <div style={{ marginBottom: '1rem' }}>
                    <div>Rank: {currentRanking.rank} of {rankingData.length}</div>
                    <div>Performance: {currentRanking.performance.toFixed(2)}%</div>
                    <div className="metric">
                      <div className="metric-label">Position Sizing</div>
                      <div className="metric-value">
                        Buy: {(0.5 * Math.exp(-2.5 * (currentRanking.rank / rankingData.length)) + 0.05).toFixed(2) * 100}%
                        <br />
                        Sell: {(0.1 + 0.9 * (currentRanking.rank / rankingData.length)).toFixed(2) * 100}%
                      </div>
                    </div>
                  </div>
                </>
              )}
              
              <h3>Top Performers</h3>
              <div className="ranking-table-container">
                {rankingData.length > 0 ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Symbol</th>
                        <th>Performance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankingData
                        .sort((a, b) => a.rank - b.rank)
                        .slice(0, 5)
                        .map((item, index) => (
                          <tr 
                            key={index}
                            className={item.symbol === currentSymbol ? 'current-symbol' : ''}
                          >
                            <td>{item.rank}</td>
                            <td>{item.symbol}</td>
                            <td className={item.performance >= 0 ? 'positive' : 'negative'}>
                              {item.performance.toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                ) : (
                  <p>Ranking data not available.</p>
                )}
              </div>
            </>
          ) : (
            <div className="loading-indicator">
              <div className="loading-indicator-small"></div>
              <span>Loading market data...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;