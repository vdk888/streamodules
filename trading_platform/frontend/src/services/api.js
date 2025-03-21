// API service for the trading platform

// Base API URL
const API_BASE_URL = 'http://localhost:5000/api';

// Helper function for fetch requests
const fetchWithTimeout = async (url, options = {}, timeout = 10000) => {
  const controller = new AbortController();
  const { signal } = controller;
  
  // Set timeout to abort the request if it takes too long
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, { ...options, signal });
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error.name === 'AbortError') {
      throw new Error('Request timed out');
    }
    
    throw error;
  }
};

// API methods
const apiService = {
  // Cryptocurrency endpoints
  
  // Get available symbols
  getSymbols: async () => {
    try {
      return await fetchWithTimeout(`${API_BASE_URL}/crypto/symbols`);
    } catch (error) {
      console.error('Error fetching symbols:', error);
      throw error;
    }
  },
  
  // Get data for a specific symbol
  getSymbolData: async (symbol, timeframe = '1h', lookbackDays = 15) => {
    try {
      return await fetchWithTimeout(
        `${API_BASE_URL}/crypto/data/${encodeURIComponent(symbol)}?timeframe=${timeframe}&lookback_days=${lookbackDays}`
      );
    } catch (error) {
      console.error(`Error fetching data for ${symbol}:`, error);
      throw error;
    }
  },
  
  // Get performance ranking
  getPerformanceRanking: async (lookbackDays = 15) => {
    try {
      return await fetchWithTimeout(
        `${API_BASE_URL}/crypto/ranking?lookback_days=${lookbackDays}`
      );
    } catch (error) {
      console.error('Error fetching performance ranking:', error);
      throw error;
    }
  },
  
  // Run backtest for a specific symbol
  runBacktest: async (symbol, days = 15, params = null) => {
    try {
      return await fetchWithTimeout(
        `${API_BASE_URL}/crypto/backtest/${encodeURIComponent(symbol)}`, 
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ days, params }),
        }
      );
    } catch (error) {
      console.error(`Error running backtest for ${symbol}:`, error);
      throw error;
    }
  },
  
  // Optimize parameters for a specific symbol
  optimizeParameters: async (symbol, days = 15) => {
    try {
      return await fetchWithTimeout(
        `${API_BASE_URL}/crypto/optimize/${encodeURIComponent(symbol)}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ days }),
        }
      );
    } catch (error) {
      console.error(`Error optimizing parameters for ${symbol}:`, error);
      throw error;
    }
  },
  
  // Simulate portfolio performance
  simulatePortfolio: async (symbols = null, days = 15, initialCapital = 100000.0) => {
    try {
      return await fetchWithTimeout(
        `${API_BASE_URL}/crypto/portfolio/simulate`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ symbols, days, initial_capital: initialCapital }),
        }
      );
    } catch (error) {
      console.error('Error simulating portfolio:', error);
      throw error;
    }
  },
  
  // Health check
  healthCheck: async () => {
    try {
      return await fetchWithTimeout(`${API_BASE_URL}/health`);
    } catch (error) {
      console.error('Error checking API health:', error);
      throw error;
    }
  },
};

export default apiService;