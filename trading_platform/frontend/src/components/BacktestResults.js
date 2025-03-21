import React from 'react';

function BacktestResults({ results, symbol }) {
  if (!results || Object.keys(results).length === 0) {
    return <div className="no-results">No backtest results available</div>;
  }

  if (results.error) {
    return <div className="error-message">{results.error}</div>;
  }

  // Format numbers for display
  const formatNumber = (num, precision = 2) => {
    return parseFloat(num).toFixed(precision);
  };

  // Format percentage
  const formatPercent = (num) => {
    return `${formatNumber(num)}%`;
  };

  // Format currency
  const formatCurrency = (num) => {
    return `$${formatNumber(num)}`;
  };

  // Color based on performance
  const getPerformanceColor = (value) => {
    return value >= 0 ? 'positive' : 'negative';
  };

  return (
    <div className="backtest-results">
      <h3>Backtest Results: {symbol}</h3>
      
      <div className="results-summary">
        <div className="results-metric">
          <span className="metric-label">Initial Capital:</span>
          <span className="metric-value">{formatCurrency(results.initial_capital)}</span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Final Value:</span>
          <span className="metric-value">{formatCurrency(results.final_value)}</span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Return:</span>
          <span className={`metric-value ${getPerformanceColor(results.return)}`}>
            {formatPercent(results.return)}
          </span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Buy & Hold Return:</span>
          <span className={`metric-value ${getPerformanceColor(results.buy_hold_return)}`}>
            {formatPercent(results.buy_hold_return)}
          </span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Outperformance:</span>
          <span className={`metric-value ${getPerformanceColor(results.outperformance)}`}>
            {formatPercent(results.outperformance)}
          </span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Max Drawdown:</span>
          <span className="metric-value negative">{formatPercent(results.max_drawdown)}</span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Number of Trades:</span>
          <span className="metric-value">{results.num_trades}</span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Win Rate:</span>
          <span className="metric-value">{formatPercent(results.win_rate * 100)}</span>
        </div>
        
        <div className="results-metric">
          <span className="metric-label">Sharpe Ratio:</span>
          <span className="metric-value">{formatNumber(results.sharpe_ratio, 3)}</span>
        </div>
      </div>
      
      {results.trades && results.trades.length > 0 && (
        <div className="trades-table">
          <h4>Recent Trades</h4>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Action</th>
                <th>Price</th>
                <th>Size</th>
                <th>Value</th>
                <th>P/L</th>
              </tr>
            </thead>
            <tbody>
              {results.trades.map((trade, index) => (
                <tr key={index} className={trade.action.toLowerCase()}>
                  <td>{new Date(trade.timestamp).toLocaleString()}</td>
                  <td>{trade.action}</td>
                  <td>{formatCurrency(trade.price)}</td>
                  <td>{formatNumber(trade.size, 4)}</td>
                  <td>
                    {trade.action === 'BUY' 
                      ? formatCurrency(trade.cost)
                      : formatCurrency(trade.revenue || 0)}
                  </td>
                  <td className={trade.pl_pct >= 0 ? 'positive' : 'negative'}>
                    {trade.pl_pct ? formatPercent(trade.pl_pct) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {results.params && (
        <div className="parameters">
          <h4>Strategy Parameters</h4>
          <div className="parameters-grid">
            {Object.entries(results.params).map(([key, value]) => (
              <div key={key} className="parameter">
                <span className="param-name">{key}:</span>
                <span className="param-value">
                  {Array.isArray(value) 
                    ? JSON.stringify(value)
                    : typeof value === 'number' 
                      ? formatNumber(value, 4)
                      : value.toString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default BacktestResults;