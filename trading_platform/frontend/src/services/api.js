import axios from 'axios';

// Create axios instance with base URL
const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
const endpoints = {
  // Health check
  health: () => api.get('/health'),
  
  // Crypto endpoints
  getCryptoSymbols: () => api.get('/crypto/symbols'),
  getCryptoData: (symbol, timeframe = '1h', lookback_days = 15) => 
    api.get(`/crypto/data/${symbol}`, { params: { timeframe, lookback_days } }),
  getCryptoRanking: (lookback_days = 15) => 
    api.get('/crypto/ranking', { params: { lookback_days } }),
  runCryptoBacktest: (symbol, days = 15, params = null) => 
    api.post(`/crypto/backtest/${symbol}`, { days, params }),
  optimizeCryptoParameters: (symbol, days = 15) => 
    api.post(`/crypto/optimize/${symbol}`, { days }),
  simulateCryptoPortfolio: (symbols = null, days = 15, initial_capital = 100000.0) => 
    api.post('/crypto/portfolio/simulate', { symbols, days, initial_capital }),
};

// WebSocket connection handler
const connectWebSocket = (symbol, onMessage) => {
  const ws = new WebSocket(`ws://${window.location.host}/ws/crypto/${symbol}`);
  
  ws.onopen = () => {
    console.log(`WebSocket connected for ${symbol}`);
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket data:', error);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log(`WebSocket disconnected for ${symbol}`);
  };
  
  // Return object with close method
  return {
    close: () => {
      ws.close();
    }
  };
};

export default {
  ...endpoints,
  connectWebSocket,
};