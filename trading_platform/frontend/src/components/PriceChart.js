import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

function PriceChart({ priceData, signalsData, symbol, timeframe }) {
  const chartRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!priceData || priceData.length === 0) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();

    // Setup dimensions
    const width = chartRef.current.clientWidth;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(chartRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Parse dates
    const parseDate = d3.timeParse('%Y-%m-%dT%H:%M:%S');
    const formatDate = d3.timeFormat('%b %d, %Y %H:%M');
    
    const formattedData = priceData.map(d => ({
      ...d,
      date: parseDate(d.timestamp.split('.')[0]), // Remove milliseconds if present
      open: +d.open,
      high: +d.high,
      low: +d.low,
      close: +d.close,
      volume: +d.volume
    }));

    // X scale - time
    const xScale = d3.scaleTime()
      .domain(d3.extent(formattedData, d => d.date))
      .range([0, innerWidth]);

    // Y scale - price
    const yScale = d3.scaleLinear()
      .domain([
        d3.min(formattedData, d => d.low) * 0.99, 
        d3.max(formattedData, d => d.high) * 1.01
      ])
      .range([innerHeight, 0]);

    // Add X axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .ticks(width > 700 ? 10 : 5)
        .tickFormat(d3.timeFormat(timeframe === '1d' ? '%b %d' : '%b %d %H:%M')));

    // Add Y axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));

    // Add Y axis label
    g.append('text')
      .attr('class', 'y-axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('y', -margin.left)
      .attr('x', -innerHeight / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Price ($)');

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat(''));

    // Candlestick chart
    g.selectAll('.candle')
      .data(formattedData)
      .enter()
      .append('g')
      .attr('class', 'candle')
      .each(function(d) {
        const candle = d3.select(this);
        
        // Determine if it's an up or down day
        const isUp = d.close >= d.open;
        
        // Draw the wick (high to low)
        candle.append('line')
          .attr('class', 'wick')
          .attr('x1', xScale(d.date))
          .attr('x2', xScale(d.date))
          .attr('y1', yScale(d.high))
          .attr('y2', yScale(d.low))
          .attr('stroke', 'black')
          .attr('stroke-width', 1);
        
        // Draw the body (open to close)
        const candleWidth = Math.max(innerWidth / formattedData.length * 0.7, 1);
        candle.append('rect')
          .attr('class', isUp ? 'up-candle' : 'down-candle')
          .attr('x', xScale(d.date) - candleWidth / 2)
          .attr('y', yScale(Math.max(d.open, d.close)))
          .attr('width', candleWidth)
          .attr('height', Math.max(1, Math.abs(yScale(d.open) - yScale(d.close))))
          .attr('fill', isUp ? '#28a745' : '#dc3545');
      });

    // Process signals data if available
    if (signalsData && signalsData.length > 0) {
      const formattedSignals = signalsData.map(d => ({
        ...d,
        date: parseDate(d.timestamp.split('.')[0]), // Remove milliseconds if present
        price: +d.price
      }));

      // Add buy signals
      g.selectAll('.buy-signal')
        .data(formattedSignals.filter(d => d.buy_signal))
        .enter()
        .append('path')
        .attr('class', 'buy-signal')
        .attr('d', d3.symbol().type(d3.symbolTriangle).size(100))
        .attr('transform', d => `translate(${xScale(d.date)},${yScale(d.price - d.price * 0.005)}) rotate(180)`)
        .attr('fill', 'green')
        .attr('stroke', 'darkgreen')
        .attr('stroke-width', 1.5);

      // Add sell signals
      g.selectAll('.sell-signal')
        .data(formattedSignals.filter(d => d.sell_signal))
        .enter()
        .append('path')
        .attr('class', 'sell-signal')
        .attr('d', d3.symbol().type(d3.symbolTriangle).size(100))
        .attr('transform', d => `translate(${xScale(d.date)},${yScale(d.price + d.price * 0.005)})`)
        .attr('fill', 'red')
        .attr('stroke', 'darkred')
        .attr('stroke-width', 1.5);
    }

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text(`${symbol} - ${timeframe}`);

    // Create tooltip
    const tooltip = d3.select(tooltipRef.current)
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background-color', 'white')
      .style('border', '1px solid #ddd')
      .style('border-radius', '4px')
      .style('padding', '10px')
      .style('pointer-events', 'none');

    // Add mouseover behavior
    g.selectAll('.candle')
      .on('mouseover', function(event, d) {
        tooltip.transition()
          .duration(200)
          .style('opacity', 0.9);
        
        tooltip.html(`
          <strong>Date:</strong> ${formatDate(d.date)}<br>
          <strong>Open:</strong> $${d.open.toFixed(2)}<br>
          <strong>High:</strong> $${d.high.toFixed(2)}<br>
          <strong>Low:</strong> $${d.low.toFixed(2)}<br>
          <strong>Close:</strong> $${d.close.toFixed(2)}<br>
          <strong>Volume:</strong> $${d.volume.toFixed(2)}
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        tooltip.transition()
          .duration(500)
          .style('opacity', 0);
      });

  }, [priceData, signalsData, symbol, timeframe]);

  return (
    <div className="price-chart">
      <div ref={chartRef} style={{ width: '100%', height: '100%' }}></div>
      <div ref={tooltipRef}></div>
    </div>
  );
}

export default PriceChart;