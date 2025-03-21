import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

function PriceChart({ priceData, signalsData, symbol, timeframe }) {
  const svgRef = useRef();
  const tooltipRef = useRef();

  useEffect(() => {
    if (!priceData || priceData.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Setup dimensions
    const margin = { top: 20, right: 50, bottom: 30, left: 50 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3
      .select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Parse dates
    const parseDate = d3.timeParse('%Y-%m-%dT%H:%M:%S.%LZ');
    const formattedData = priceData.map(d => ({
      ...d,
      date: parseDate(d.timestamp) || new Date(d.timestamp)
    }));

    // Sort data by date
    formattedData.sort((a, b) => a.date - b.date);

    // Set up scales
    const x = d3.scaleTime()
      .domain(d3.extent(formattedData, d => d.date))
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([
        d3.min(formattedData, d => d.low * 0.998),
        d3.max(formattedData, d => d.high * 1.002)
      ])
      .range([height, 0]);

    // Format price with appropriate precision
    const formatPrice = d3.format(',.2f');

    // Create line generator for price
    const line = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.close));

    // Draw axes
    const xAxis = svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(width / 100).tickFormat(date => {
        const format = timeframe === '1d' ? '%Y-%m-%d' : '%m-%d %H:%M';
        return d3.timeFormat(format)(date);
      }));

    const yAxis = svg.append('g')
      .call(d3.axisLeft(y).tickFormat(d => `$${formatPrice(d)}`));

    // Add X axis label
    svg.append('text')
      .attr('transform', `translate(${width / 2}, ${height + margin.top + 10})`)
      .style('text-anchor', 'middle')
      .text(`Time (${timeframe})`);

    // Add Y axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Price (USD)');

    // Draw price line
    svg.append('path')
      .datum(formattedData)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 1.5)
      .attr('d', line);

    // Create a hover effect with tooltip
    const tooltip = d3.select(tooltipRef.current)
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background-color', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('pointer-events', 'none');

    // Draw signals if available
    if (signalsData && signalsData.length > 0) {
      // Format signal data
      const formattedSignals = signalsData.map(d => ({
        ...d,
        date: parseDate(d.timestamp) || new Date(d.timestamp)
      }));

      // Draw buy signals
      svg.selectAll('.buy-signal')
        .data(formattedSignals.filter(d => d.buy_signal))
        .enter()
        .append('circle')
        .attr('class', 'buy-signal')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.price))
        .attr('r', 5)
        .attr('fill', 'green')
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .on('mouseover', (event, d) => {
          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip.html(`
            <strong>Buy Signal</strong><br/>
            Date: ${d.date.toLocaleString()}<br/>
            Price: $${formatPrice(d.price)}<br/>
            Indicator: ${(d.daily_composite).toFixed(4)}
          `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', () => {
          tooltip.transition().duration(500).style('opacity', 0);
        });

      // Draw sell signals
      svg.selectAll('.sell-signal')
        .data(formattedSignals.filter(d => d.sell_signal))
        .enter()
        .append('circle')
        .attr('class', 'sell-signal')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.price))
        .attr('r', 5)
        .attr('fill', 'red')
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .on('mouseover', (event, d) => {
          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip.html(`
            <strong>Sell Signal</strong><br/>
            Date: ${d.date.toLocaleString()}<br/>
            Price: $${formatPrice(d.price)}<br/>
            Indicator: ${(d.daily_composite).toFixed(4)}
          `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', () => {
          tooltip.transition().duration(500).style('opacity', 0);
        });

      // Draw potential buy signals (smaller dots)
      svg.selectAll('.potential-buy')
        .data(formattedSignals.filter(d => d.potential_buy && !d.buy_signal))
        .enter()
        .append('circle')
        .attr('class', 'potential-buy')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.price))
        .attr('r', 3)
        .attr('fill', 'rgba(0, 128, 0, 0.5)')
        .attr('stroke', 'white')
        .attr('stroke-width', 0.5);

      // Draw potential sell signals (smaller dots)
      svg.selectAll('.potential-sell')
        .data(formattedSignals.filter(d => d.potential_sell && !d.sell_signal))
        .enter()
        .append('circle')
        .attr('class', 'potential-sell')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.price))
        .attr('r', 3)
        .attr('fill', 'rgba(255, 0, 0, 0.5)')
        .attr('stroke', 'white')
        .attr('stroke-width', 0.5);
    }

    // Add hover overlay for price information
    const bisectDate = d3.bisector(d => d.date).left;
    
    const overlay = svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .style('fill', 'none')
      .style('pointer-events', 'all')
      .on('mousemove', function(event) {
        const x0 = x.invert(d3.pointer(event)[0]);
        const i = bisectDate(formattedData, x0, 1);
        const d0 = formattedData[i - 1];
        const d1 = formattedData[i];
        if (!d0 || !d1) return;
        
        const d = x0 - d0.date > d1.date - x0 ? d1 : d0;
        
        tooltip.transition().duration(200).style('opacity', 0.9);
        tooltip.html(`
          <strong>${symbol}</strong><br/>
          Date: ${d.date.toLocaleString()}<br/>
          Open: $${formatPrice(d.open)}<br/>
          High: $${formatPrice(d.high)}<br/>
          Low: $${formatPrice(d.low)}<br/>
          Close: $${formatPrice(d.close)}<br/>
          Volume: ${d3.format(',')(Math.round(d.volume))}
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', () => {
        tooltip.transition().duration(500).style('opacity', 0);
      });

    // Add chart title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0 - (margin.top / 2))
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text(`${symbol} Price Chart (${timeframe})`);

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - 150}, 10)`);

    // Price line legend
    legend.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 20)
      .attr('y2', 0)
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 1.5);

    legend.append('text')
      .attr('x', 25)
      .attr('y', 4)
      .text('Price');

    // Buy signal legend
    legend.append('circle')
      .attr('cx', 10)
      .attr('cy', 20)
      .attr('r', 5)
      .attr('fill', 'green');

    legend.append('text')
      .attr('x', 25)
      .attr('y', 24)
      .text('Buy Signal');

    // Sell signal legend
    legend.append('circle')
      .attr('cx', 10)
      .attr('cy', 40)
      .attr('r', 5)
      .attr('fill', 'red');

    legend.append('text')
      .attr('x', 25)
      .attr('y', 44)
      .text('Sell Signal');

  }, [priceData, signalsData, symbol, timeframe]);

  return (
    <div className="price-chart">
      <svg ref={svgRef} width="100%" height="400"></svg>
      <div ref={tooltipRef} className="tooltip"></div>
    </div>
  );
}

export default PriceChart;