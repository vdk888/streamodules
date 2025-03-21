import React from 'react';

const BacktestResults = ({ results }) => {
  if (!results) {
    return null;
  }
  
  // Check if there's an error
  if (results.error) {
    return (
      <div className="backtest-results error">
        <h3>Backtest Error</h3>
        <p className="error-message">{results.error}</p>
      </div>
    );
  }
  
  return (
    <div className="backtest-results">
      <h3>Backtest Results for {results.symbol}</h3>
      
      <div className="metrics-grid">
        <div className="metric">
          <div className="metric-label">Initial Capital</div>
          <div className="metric-value">${results.initial_capital.toLocaleString()}</div>
        </div>
        
        <div className="metric">
          <div className="metric-label">Final Value</div>
          <div className="metric-value">${results.final_value.toLocaleString()}</div>
        </div>
        
        <div className="metric">
          <div className="metric-label">ROI</div>
          <div className={`metric-value ${results.roi_percent >= 0 ? 'positive' : 'negative'}`}>
            {results.roi_percent.toFixed(2)}%
          </div>
        </div>
        
        <div className="metric">
          <div className="metric-label">Annualized ROI</div>
          <div className={`metric-value ${results.annualized_roi >= 0 ? 'positive' : 'negative'}`}>
            {results.annualized_roi.toFixed(2)}%
          </div>
        </div>
        
        <div className="metric">
          <div className="metric-label">Max Drawdown</div>
          <div className="metric-value negative">{results.max_drawdown.toFixed(2)}%</div>
        </div>
        
        <div className="metric">
          <div className="metric-label">Number of Trades</div>
          <div className="metric-value">{results.n_trades}</div>
        </div>
        
        <div className="metric">
          <div className="metric-label">Test Period</div>
          <div className="metric-value">{results.n_days} days</div>
        </div>
      </div>
      
      {results.trades && results.trades.length > 0 && (
        <div className="trades-section">
          <h4>Trades</h4>
          <div className="trades-table-container">
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Action</th>
                  <th>Price</th>
                  <th>Amount</th>
                  <th>Shares</th>
                  <th>Fees</th>
                </tr>
              </thead>
              <tbody>
                {results.trades.map((trade, index) => (
                  <tr key={index} className={trade.action === 'BUY' ? 'buy-row' : 'sell-row'}>
                    <td>{new Date(trade.timestamp).toLocaleString()}</td>
                    <td>{trade.action}</td>
                    <td>${trade.price.toLocaleString()}</td>
                    <td>${trade.amount.toLocaleString()}</td>
                    <td>{trade.shares.toLocaleString()}</td>
                    <td>${trade.fees.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default BacktestResults;