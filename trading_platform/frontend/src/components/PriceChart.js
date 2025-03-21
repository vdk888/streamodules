import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const PriceChart = ({ priceData, signals, symbol }) => {
  const [chartData, setChartData] = useState(null);
  
  useEffect(() => {
    if (!priceData || priceData.length === 0) return;
    
    // Prepare data for chart
    const labels = priceData.map(d => new Date(d.timestamp));
    const prices = priceData.map(d => d.Close);
    
    // Create buy and sell point datasets
    let buyPoints = [];
    let sellPoints = [];
    
    if (signals && signals.length > 0) {
      signals.forEach((signal, index) => {
        if (signal.signal === 1) {  // Buy signal
          buyPoints.push({
            x: new Date(signal.timestamp),
            y: signal.price,
          });
        } else if (signal.signal === -1) {  // Sell signal
          sellPoints.push({
            x: new Date(signal.timestamp),
            y: signal.price,
          });
        }
      });
    }
    
    // Create chart data
    const data = {
      labels,
      datasets: [
        {
          label: `${symbol} Price`,
          data: prices,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1,
          pointRadius: 0,
        },
        {
          label: 'Buy Signals',
          data: buyPoints,
          borderColor: 'rgb(0, 255, 0)',
          backgroundColor: 'rgb(0, 255, 0)',
          pointRadius: 5,
          pointHoverRadius: 7,
          showLine: false,
        },
        {
          label: 'Sell Signals',
          data: sellPoints,
          borderColor: 'rgb(255, 0, 0)',
          backgroundColor: 'rgb(255, 0, 0)',
          pointRadius: 5,
          pointHoverRadius: 7,
          showLine: false,
        },
      ],
    };
    
    setChartData(data);
  }, [priceData, signals, symbol]);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${symbol} Price Chart`,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          tooltipFormat: 'MMM d, yyyy h:mm a',
          displayFormats: {
            day: 'MMM d',
          },
        },
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Price',
        },
      },
    },
    interaction: {
      mode: 'nearest',
      intersect: false,
    },
  };
  
  if (!chartData) {
    return <div>Loading chart data...</div>;
  }
  
  return (
    <div style={{ height: '400px' }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default PriceChart;