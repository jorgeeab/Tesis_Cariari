function sbnsFmt(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(digits);
}

function sbnsClamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sbnsSetText(id, value) {
  document.getElementById(id).textContent = value;
}

function sbnsGetCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function sbnsCreateControls(containerId, config, state, onUpdate) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  config.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "control";

    const head = document.createElement("div");
    head.className = "control-head";

    const label = document.createElement("label");
    label.textContent = item.label;
    label.setAttribute("for", `${containerId}-${item.key}`);
    head.appendChild(label);

    if (item.type === "select") {
      const select = document.createElement("select");
      select.id = `${containerId}-${item.key}`;
      item.options.forEach((option) => {
        const element = document.createElement("option");
        element.value = option.value;
        element.textContent = option.label;
        if (option.value === item.value) element.selected = true;
        select.appendChild(element);
      });
      select.addEventListener("change", (event) => {
        state[item.key] = event.target.value;
        onUpdate();
      });
      head.appendChild(select);
      wrapper.appendChild(head);
      container.appendChild(wrapper);
      return;
    }

    const number = document.createElement("input");
    number.type = "number";
    number.id = `${containerId}-${item.key}`;
    number.min = item.min;
    number.max = item.max;
    number.step = item.step;
    number.value = item.value;
    head.appendChild(number);
    wrapper.appendChild(head);

    const range = document.createElement("input");
    range.type = "range";
    range.min = item.min;
    range.max = item.max;
    range.step = item.step;
    range.value = item.value;
    wrapper.appendChild(range);

    function sync(raw) {
      let value = Number(raw);
      if (!Number.isFinite(value)) value = Number(item.value);
      value = sbnsClamp(value, Number(item.min), Number(item.max));
      state[item.key] = value;
      number.value = value;
      range.value = value;
      onUpdate();
    }

    number.addEventListener("input", (event) => sync(event.target.value));
    range.addEventListener("input", (event) => sync(event.target.value));
    container.appendChild(wrapper);
  });
}

function sbnsRenderEquations(containerId, cards) {
  const container = document.getElementById(containerId);
  container.innerHTML = cards.map((card) => `
    <div class="equation-card">
      <strong>${card.title}</strong>
      <code>${card.expr}</code>
    </div>
  `).join("");
}

function sbnsCreateAxes(width, height, margin, maxY, maxX, labelsY = 4, yTitle = "Volumen (m3)") {
  let html = "";
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  for (let i = 0; i <= labelsY; i += 1) {
    const y = margin.top + (innerHeight * i) / labelsY;
    const value = maxY - (maxY * i) / labelsY;
    html += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(34,49,40,0.12)" stroke-dasharray="4 4"></line>`;
    html += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#66756b">${sbnsFmt(value, maxY < 10 ? 2 : 1)}</text>`;
  }

  html += `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#35443a" stroke-width="1.5"></line>`;
  html += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#35443a" stroke-width="1.5"></line>`;
  html += `<text x="${width / 2}" y="${height - 10}" text-anchor="middle" font-size="11" fill="#66756b">Tiempo (min)</text>`;
  html += `<text x="18" y="${height / 2}" text-anchor="middle" font-size="11" fill="#66756b" transform="rotate(-90 18 ${height / 2})">${yTitle}</text>`;

  for (let i = 0; i <= 4; i += 1) {
    const x = margin.left + (innerWidth * i) / 4;
    const value = (maxX * i) / 4;
    html += `<text x="${x}" y="${height - margin.bottom + 18}" text-anchor="middle" font-size="11" fill="#66756b">${sbnsFmt(value, 1)}</text>`;
  }

  return html;
}

function sbnsPointsFromSeries(series, width, height, margin, maxX, maxY, accessor) {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  return series.map((item) => {
    const x = margin.left + (item.minute / Math.max(maxX, 1e-9)) * innerWidth;
    const y = margin.top + innerHeight - (accessor(item) / Math.max(maxY, 1e-9)) * innerHeight;
    return `${x},${y}`;
  }).join(" ");
}

function sbnsRenderTimeline(svgId, series, options) {
  const svg = document.getElementById(svgId);
  const width = options.width || 560;
  const height = options.height || 300;
  const margin = options.margin || { top: 22, right: 22, bottom: 44, left: 62 };
  const maxX = options.maxX;
  const maxY = Math.max(...series.map((item) => Math.max(options.storage(item), options.line(item))), 1);
  const areaPoints = sbnsPointsFromSeries(series, width, height, margin, maxX, maxY, options.storage);
  const linePoints = sbnsPointsFromSeries(series, width, height, margin, maxX, maxY, options.line);
  const baseY = height - margin.bottom;
  const firstX = margin.left;
  const lastX = width - margin.right;

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fffdf7"></rect>
    ${sbnsCreateAxes(width, height, margin, maxY, maxX, 4, options.yTitle || "Volumen (m3)")}
    <polygon points="${firstX},${baseY} ${areaPoints} ${lastX},${baseY}" fill="${options.areaFill || "rgba(127, 182, 217, 0.38)"}"></polygon>
    <polyline points="${areaPoints}" fill="none" stroke="${options.areaStroke || "#4f87ad"}" stroke-width="2.5"></polyline>
    <polyline points="${linePoints}" fill="none" stroke="${options.lineStroke || "#2f6f4f"}" stroke-width="3"></polyline>
  `;
}

function sbnsRenderIndicatorBars(svgId, bars) {
  const svg = document.getElementById(svgId);
  const width = 560;
  const height = 300;
  const margin = { top: 22, right: 22, bottom: 52, left: 62 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const slot = innerWidth / bars.length;
  const barWidth = slot * 0.52;

  let html = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fffdf7"></rect>
    <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#35443a" stroke-width="1.5"></line>
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#35443a" stroke-width="1.5"></line>
  `;

  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (innerHeight * i) / 4;
    const value = 100 - 25 * i;
    html += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(34,49,40,0.12)" stroke-dasharray="4 4"></line>`;
    html += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#66756b">${value}</text>`;
  }

  bars.forEach((bar, index) => {
    const x = margin.left + slot * index + (slot - barWidth) / 2;
    const barHeight = (bar.value / 100) * innerHeight;
    const y = height - margin.bottom - barHeight;
    html += `
      <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="12" fill="${bar.color}"></rect>
      <text x="${x + barWidth / 2}" y="${height - margin.bottom + 18}" text-anchor="middle" font-size="11" fill="#66756b">${bar.label}</text>
      <text x="${x + barWidth / 2}" y="${y - 8}" text-anchor="middle" font-size="11" fill="#223128">${bar.raw}</text>
    `;
  });

  svg.innerHTML = html;
}
