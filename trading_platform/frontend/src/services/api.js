/**
 * API service for interacting with the cryptocurrency trading platform backend
 */

/**
 * Fetch available cryptocurrency symbols
 * @param {string} baseUrl - API base URL
 * @returns {Promise} - Promise resolving to JSON data containing symbols
 */
export const fetchSymbols = async (baseUrl) => {
  try {
    console.log(`Fetching symbols from: ${baseUrl}/api/crypto/symbols`);
    const response = await fetch(`${baseUrl}/api/crypto/symbols`);
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error response text:', errorText);
      throw new Error(`HTTP error! Status: ${response.status}, Text: ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Parsed response data:', data);
    return data;
  } catch (error) {
    console.error('Error fetching symbols:', error);
    throw error;
  }
};

/**
 * Fetch data for a specific cryptocurrency
 * @param {string} baseUrl - API base URL
 * @param {string} symbol - Cryptocurrency symbol
 * @param {string} timeframe - Timeframe for data (e.g., '1h', '4h', '1d')
 * @param {number} lookbackDays - Number of days to look back
 * @returns {Promise} - Promise resolving to JSON data containing price and signals
 */
export const fetchSymbolData = async (baseUrl, symbol, timeframe = '1h', lookbackDays = 15) => {
  try {
    const response = await fetch(
      `${baseUrl}/api/crypto/data/${symbol}?timeframe=${timeframe}&lookback_days=${lookbackDays}`
    );
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching data for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Fetch performance ranking of cryptocurrencies
 * @param {string} baseUrl - API base URL
 * @param {number} lookbackDays - Number of days to look back
 * @returns {Promise} - Promise resolving to JSON data containing ranking
 */
export const fetchRanking = async (baseUrl, lookbackDays = 15) => {
  try {
    const response = await fetch(`${baseUrl}/api/crypto/ranking?lookback_days=${lookbackDays}`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching ranking:', error);
    throw error;
  }
};

/**
 * Run backtest for a cryptocurrency
 * @param {string} baseUrl - API base URL
 * @param {string} symbol - Cryptocurrency symbol
 * @param {number} days - Number of days to backtest
 * @param {Object} params - Optional parameters for backtest
 * @returns {Promise} - Promise resolving to JSON data containing backtest results
 */
export const runBacktest = async (baseUrl, symbol, days = 15, params = null) => {
  try {
    const response = await fetch(`${baseUrl}/api/crypto/backtest/${symbol}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        days,
        params,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error running backtest for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Optimize parameters for a cryptocurrency
 * @param {string} baseUrl - API base URL
 * @param {string} symbol - Cryptocurrency symbol
 * @param {number} days - Number of days for optimization
 * @returns {Promise} - Promise resolving to JSON data containing optimal parameters
 */
export const optimizeParameters = async (baseUrl, symbol, days = 15) => {
  try {
    const response = await fetch(`${baseUrl}/api/crypto/optimize/${symbol}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        days,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error optimizing parameters for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Simulate portfolio with multiple cryptocurrencies
 * @param {string} baseUrl - API base URL
 * @param {Array} symbols - Array of cryptocurrency symbols
 * @param {number} days - Number of days for simulation
 * @param {number} initialCapital - Initial capital for simulation
 * @returns {Promise} - Promise resolving to JSON data containing portfolio simulation results
 */
export const simulatePortfolio = async (baseUrl, symbols = null, days = 15, initialCapital = 100000) => {
  try {
    const response = await fetch(`${baseUrl}/api/crypto/portfolio/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbols,
        days,
        initial_capital: initialCapital,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error simulating portfolio:', error);
    throw error;
  }
};

/**
 * Setup WebSocket connection for real-time data
 * @param {string} baseUrl - API base URL
 * @param {string} symbol - Cryptocurrency symbol
 * @param {function} onMessage - Callback function for handling messages
 * @param {function} onError - Callback function for handling errors
 * @param {function} onClose - Callback function for handling connection close
 * @returns {WebSocket} - WebSocket instance
 */
export const setupWebSocket = (baseUrl, symbol, onMessage, onError, onClose) => {
  // Convert http(s) to ws(s)
  const wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  const ws = new WebSocket(`${wsBaseUrl}/ws/crypto/${symbol}`);

  ws.onopen = () => {
    console.log(`WebSocket connection established for ${symbol}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (onError) {
      onError(error);
    }
  };

  ws.onclose = (event) => {
    console.log(`WebSocket connection closed for ${symbol}:`, event.code, event.reason);
    if (onClose) {
      onClose(event);
    }
  };

  return ws;
};