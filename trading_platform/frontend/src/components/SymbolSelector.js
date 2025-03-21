import React from 'react';

function SymbolSelector({ symbols, currentSymbol, onSymbolChange, isLoading }) {
  const handleChange = (event) => {
    const selectedSymbol = event.target.value;
    onSymbolChange(selectedSymbol);
  };

  return (
    <div className="symbol-selector">
      <label>Symbol:</label>
      <select 
        value={currentSymbol} 
        onChange={handleChange}
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
      {isLoading && (
        <div className="loading-indicator-small"></div>
      )}
    </div>
  );
}

export default SymbolSelector;