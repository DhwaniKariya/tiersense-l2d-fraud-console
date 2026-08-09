/* Minimal dependency-free SVG chart helpers, styled to match the dark console theme
 * (replaces the matplotlib figures from the Streamlit prototype). */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function niceMax(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  let step;
  if (norm <= 1) step = 1;
  else if (norm <= 2) step = 2;
  else if (norm <= 5) step = 5;
  else step = 10;
  return step * mag;
}

/**
 * Grouped bar chart. series: [{label, values (per category), color}]
 * categories: array of x-axis labels.
 */
function renderGroupedBarChart(container, opts) {
  const { title, categories, series, yLabel, valueFmt = (v) => v.toFixed(2), height = 260 } = opts;
  const width = container.clientWidth || 340;
  const padL = 44;
  const padR = 14;
  const padT = title ? 34 : 14;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const allVals = series.flatMap((s) => s.values);
  const maxVal = niceMax(Math.max(...allVals, 0.0001) * 1.18);

  const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "ts-chart" });

  if (title) {
    const t = svgEl("text", { x: width / 2, y: 18, "text-anchor": "middle", class: "chart-title" });
    t.textContent = title;
    svg.appendChild(t);
  }

  // gridlines + y-axis labels
  const gridN = 4;
  for (let i = 0; i <= gridN; i++) {
    const val = (maxVal / gridN) * i;
    const y = padT + plotH - (val / maxVal) * plotH;
    svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: y, y2: y, class: "chart-grid" }));
    const lbl = svgEl("text", { x: padL - 6, y: y + 3, "text-anchor": "end", class: "chart-tick" });
    lbl.textContent = valueFmt(val);
    svg.appendChild(lbl);
  }

  if (yLabel) {
    const yl = svgEl("text", {
      x: 12,
      y: padT + plotH / 2,
      "text-anchor": "middle",
      class: "chart-axis-label",
      transform: `rotate(-90, 12, ${padT + plotH / 2})`,
    });
    yl.textContent = yLabel;
    svg.appendChild(yl);
  }

  const groupW = plotW / categories.length;
  const barGap = 4;
  const barW = Math.min(30, (groupW - barGap * (series.length + 1)) / series.length);

  categories.forEach((cat, ci) => {
    const groupX = padL + ci * groupW;
    const totalBarsW = barW * series.length + barGap * (series.length - 1);
    let bx = groupX + (groupW - totalBarsW) / 2;

    series.forEach((s) => {
      const val = s.values[ci];
      const barH = (val / maxVal) * plotH;
      const y = padT + plotH - barH;
      svg.appendChild(
        svgEl("rect", { x: bx, y, width: barW, height: Math.max(barH, 0), fill: s.color, rx: 1.5 })
      );
      const vt = svgEl("text", { x: bx + barW / 2, y: y - 4, "text-anchor": "middle", class: "chart-bar-label" });
      vt.textContent = valueFmt(val);
      svg.appendChild(vt);
      bx += barW + barGap;
    });

    const xt = svgEl("text", { x: groupX + groupW / 2, y: padT + plotH + 18, "text-anchor": "middle", class: "chart-tick" });
    xt.textContent = cat;
    svg.appendChild(xt);
  });

  // x-axis baseline
  svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: padT + plotH, y2: padT + plotH, class: "chart-axis" }));

  container.innerHTML = "";
  container.appendChild(svg);

  if (opts.legend) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    opts.legend.forEach(({ label, color }) => {
      const item = document.createElement("span");
      item.className = "chart-legend-item";
      item.innerHTML = `<span class="chart-legend-swatch" style="background:${color}"></span>${label}`;
      legend.appendChild(item);
    });
    container.appendChild(legend);
  }
}

/** Scatter chart with a dashed mean line. */
function renderScatterChart(container, opts) {
  const { title, xValues, yValues, color, xLabel, yLabel, height = 260 } = opts;
  const width = container.clientWidth || 340;
  const padL = 44;
  const padR = 14;
  const padT = title ? 34 : 14;
  const padB = 40;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const xMin = Math.min(...xValues) - 0.6;
  const xMax = Math.max(...xValues) + 0.6;
  const yMax = niceMax(Math.max(...yValues) * 1.2);
  const yMin = 0;
  const mean = yValues.reduce((a, b) => a + b, 0) / yValues.length;

  const xScale = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
  const yScale = (y) => padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "ts-chart" });

  if (title) {
    const t = svgEl("text", { x: width / 2, y: 18, "text-anchor": "middle", class: "chart-title" });
    t.textContent = title;
    svg.appendChild(t);
  }

  const gridN = 4;
  for (let i = 0; i <= gridN; i++) {
    const val = (yMax / gridN) * i;
    const y = yScale(val);
    svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: y, y2: y, class: "chart-grid" }));
    const lbl = svgEl("text", { x: padL - 6, y: y + 3, "text-anchor": "end", class: "chart-tick" });
    lbl.textContent = val.toFixed(1) + "x";
    svg.appendChild(lbl);
  }

  // mean dashed line
  const my = yScale(mean);
  svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: my, y2: my, class: "chart-mean-line" }));
  const meanLbl = svgEl("text", { x: padL + plotW - 4, y: my - 5, "text-anchor": "end", class: "chart-tick" });
  meanLbl.textContent = `mean ${mean.toFixed(1)}x`;
  svg.appendChild(meanLbl);

  xValues.forEach((x, i) => {
    const cx = xScale(x);
    const cy = yScale(yValues[i]);
    svg.appendChild(svgEl("circle", { cx, cy, r: 5.5, fill: color }));
    const xt = svgEl("text", { x: cx, y: padT + plotH + 18, "text-anchor": "middle", class: "chart-tick" });
    xt.textContent = x;
    svg.appendChild(xt);
  });

  svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: padT + plotH, y2: padT + plotH, class: "chart-axis" }));

  if (xLabel) {
    const xl = svgEl("text", { x: padL + plotW / 2, y: height - 4, "text-anchor": "middle", class: "chart-axis-label" });
    xl.textContent = xLabel;
    svg.appendChild(xl);
  }

  container.innerHTML = "";
  container.appendChild(svg);
}
