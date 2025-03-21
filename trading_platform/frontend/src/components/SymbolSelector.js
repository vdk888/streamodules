import React from 'react';

function SymbolSelector({ symbols, currentSymbol, onSymbolChange, isLoading }) {
  const handleSymbolChange = (e) => {
    onSymbolChange(e.target.value);
  };

  return (
    <div className="symbol-selector">
      <label htmlFor="symbol-select">Select Cryptocurrency:</label>
      <select
        id="symbol-select"
        value={currentSymbol}
        onChange={handleSymbolChange}
        disabled={isLoading}
      >
        {isLoading ? (
          <option value="">Loading symbols...</option>
        ) : (
          symbols.map((symbol) => (
            <option key={symbol.symbol} value={symbol.symbol}>
              {symbol.name} ({symbol.symbol})
            </option>
          ))
        )}
      </select>
      
      {isLoading && <span className="loading-indicator-small"></span>}
    </div>
  );
}

export default SymbolSelector;