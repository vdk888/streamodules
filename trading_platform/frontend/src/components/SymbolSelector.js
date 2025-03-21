import React from 'react';

const SymbolSelector = ({ symbols, selectedSymbol, onSymbolChange }) => {
  return (
    <div className="symbol-selector">
      <label htmlFor="symbol-select">Select Cryptocurrency:</label>
      <select
        id="symbol-select"
        value={selectedSymbol}
        onChange={(e) => onSymbolChange(e.target.value)}
        className="form-select"
      >
        {symbols.map((symbol) => (
          <option key={symbol.symbol} value={symbol.symbol}>
            {symbol.name} ({symbol.symbol})
          </option>
        ))}
      </select>
    </div>
  );
};

export default SymbolSelector;