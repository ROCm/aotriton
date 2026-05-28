// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Theoretical peak matrix TFLOPS (dense, no sparsity).
// Source: AMD datasheets and third-party benchmarks (see visperf_rev1.md).
const THEORETICAL_PEAK_TFLOPS = {
  gfx942:  { float16: 1307.4, bfloat16: 1307.4, float32: 163.4 },
  gfx950:  { float16: 2880,   bfloat16: 2880,   float32: 157   },
  gfx90a:  { float16: 383.0,  bfloat16: 383.0,  float32: 95.7  },
  gfx1100: { float16: 90.4,   bfloat16: 90.4,   float32: 45.2  },
  gfx1201: { float16: 191,    bfloat16: 191,     float32: 47.8  },
};

// Descriptor registry — populated by vis_descriptors/*.js files loaded before this script.
const DESCRIPTORS = {};
function registerDescriptor(desc) { DESCRIPTORS[desc.id] = desc; }

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

const _HUE_STOPS = [0,   30,  60,  120, 220];
const _SAT_STOPS = [90,  90,  90,  70,  80];
const _LIT_STOPS = [45,  50,  50,  40,  45];

function _interpStops(stops, t) {
  const i = Math.min(Math.floor(t), stops.length - 2);
  const f = t - i;
  return stops[i] + f * (stops[i + 1] - stops[i]);
}

// Returns {h, s, l} for a fraction in [0,1].
function _hslComponents(fraction) {
  const t = Math.max(0, Math.min(1, fraction)) * (_HUE_STOPS.length - 1);
  return {
    h: _interpStops(_HUE_STOPS, t),
    s: _interpStops(_SAT_STOPS, t),
    l: _interpStops(_LIT_STOPS, t),
  };
}

// Background color for a performance fraction.
function perfColor(fraction) {
  const { h, s, l } = _hslComponents(fraction);
  return `hsl(${h.toFixed(1)},${s.toFixed(1)}%,${l.toFixed(1)}%)`;
}

// Text color (black or white) for readability on the given performance fraction background.
// Uses WCAG luminance approximation: lightness > 0.35 → black text.
function cellTextColor(fraction) {
  const { l } = _hslComponents(fraction);
  return (l / 100) > 0.35 ? '#111' : '#fff';
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

let state = {
  descriptor: null,
  data: null,           // raw API response {arch, kernel, axes, rows}
  colDims: [],
  rowDims: [],
  fixed: {},
  // filter: per-dim allowed value sets for grid dims (col/row).
  // null = all values shown. A Set means only those values are shown.
  filter: {},
  displayMode: 'heatmap',   // 'heatmap' | '3d' | 'autozoom'
  autozoom: { drilldown: null },  // drilldown = {rowCombo, colCombo} or null for overview
  seqlenRange: [0, 65536],
  scale: {
    mode: 'max_observed',   // 'max_observed' | 'theoretical' | 'user'
    userValue: null,
  },
  arch: '',
  kernel: '',
};

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------

async function fetchData(arch, kernel, mode) {
  const params = new URLSearchParams({
    arch,
    kernel,
    mode: mode || 'kernel',
    seqlen_min: state.seqlenRange[0],
    seqlen_max: state.seqlenRange[1],
  });
  const resp = await fetch(`/api/perf/data?${params}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();

  // Annotate rows with the kernel name for the tflops() function.
  data.rows.forEach(r => { r._kernel = kernel; });
  return data;
}

// ---------------------------------------------------------------------------
// Layout builder
// ---------------------------------------------------------------------------

// Returns the Cartesian product of arrays.
function cartesian(arrays) {
  if (!arrays.length) return [[]];
  const [head, ...tail] = arrays;
  const rest = cartesian(tail);
  return head.flatMap(v => rest.map(t => [v, ...t]));
}

// Enumerate distinct dim-value tuples present in rows (for a given set of dim keys).
function distinctCombos(rows, dimKeys) {
  const seen = new Set();
  const result = [];
  for (const r of rows) {
    const key = dimKeys.map(k => r[k]).join('|');
    if (!seen.has(key)) {
      seen.add(key);
      const obj = {};
      dimKeys.forEach(k => { obj[k] = r[k]; });
      result.push(obj);
    }
  }
  return result;
}

// Apply fixed filters and per-dim value filters to rows.
function applyFixed(rows, fixed, filter) {
  return rows.filter(r => {
    for (const [k, v] of Object.entries(fixed)) {
      if (r[k] !== v) return false;
    }
    if (filter) {
      for (const [k, allowed] of Object.entries(filter)) {
        if (allowed !== null && !allowed.has(String(r[k]))) return false;
      }
    }
    return true;
  });
}

// Build nested layout: array of {rowCombo, colCombo, matrix} objects.
function buildLayout(state) {
  if (!state.data) return [];
  const { rows, axes } = state.data;

  const filtered = applyFixed(rows, state.fixed, state.filter);

  const rowCombos = distinctCombos(filtered, state.rowDims);
  const colCombos = distinctCombos(filtered, state.colDims);
  const seqQ = axes.seqlen_q || [];
  const seqK = axes.seqlen_k || [];

  const cells = [];
  for (const rowCombo of rowCombos) {
    for (const colCombo of colCombos) {
      // Build index of (seqlen_q, seqlen_k) -> row for this combo.
      const index = new Map();
      for (const r of filtered) {
        const matchRow = state.rowDims.every(k => r[k] === rowCombo[k]);
        const matchCol = state.colDims.every(k => r[k] === colCombo[k]);
        if (matchRow && matchCol) {
          index.set(`${r.seqlen_q}|${r.seqlen_k}`, r);
        }
      }
      cells.push({ rowCombo, colCombo, seqQ, seqK, index });
    }
  }
  return { rowCombos, colCombos, seqQ, seqK, cells };
}

// ---------------------------------------------------------------------------
// Scale anchor computation
// ---------------------------------------------------------------------------

function computeAnchor(data, state) {
  if (!data) return 1;
  const desc = state.descriptor;
  if (!desc) return 1;

  if (state.scale.mode === 'user' && state.scale.userValue > 0) {
    return state.scale.userValue;
  }
  if (state.scale.mode === 'theoretical') {
    const arch = state.arch;
    const dtype = state.fixed.dtype || (data.axes.dtype && data.axes.dtype[0]) || 'float16';
    const peak = THEORETICAL_PEAK_TFLOPS[arch];
    if (peak && peak[dtype]) return peak[dtype];
    // Fall through to max_observed if not found.
  }
  // max_observed: max TFLOPS across all currently displayed rows.
  const filtered = applyFixed(data.rows, state.fixed, state.filter);
  let maxT = 0;
  for (const r of filtered) {
    if (r.median_ms > 0) {
      const t = desc.tflops(r);
      if (t > maxT) maxT = t;
    }
  }
  return maxT || 1;
}

// ---------------------------------------------------------------------------
// Rendering: heatmap
// ---------------------------------------------------------------------------

function renderHeatmap(container, seqQ, seqK, index, desc, anchor) {
  const tbl = document.createElement('table');
  tbl.style.cssText = 'border-collapse:collapse;font-size:0.8em;font-family:monospace;width:auto;';

  // Header row: empty corner + one th per seqlen_k value.
  const thead = tbl.createTHead();
  const hrow = thead.insertRow();
  const corner = document.createElement('th');
  corner.style.cssText = 'padding:2px 6px;opacity:0.5;font-weight:normal;';
  corner.textContent = 'sq \\ sk';
  hrow.appendChild(corner);
  seqK.forEach(k => {
    const th = document.createElement('th');
    th.style.cssText = 'padding:2px 4px;text-align:center;opacity:0.7;font-weight:normal;white-space:nowrap;';
    th.textContent = k;
    hrow.appendChild(th);
  });

  // Data rows: one per seqlen_q.
  const tbody = tbl.createTBody();
  seqQ.forEach(q => {
    const tr = tbody.insertRow();

    // Row header.
    const th = document.createElement('th');
    th.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.7;font-weight:normal;white-space:nowrap;width:1px;';
    th.textContent = q;
    tr.appendChild(th);

    seqK.forEach(k => {
      const td = tr.insertCell();
      td.style.cssText = 'padding:0;width:4.5rem;height:2.2rem;text-align:center;vertical-align:middle;';

      const row = index.get(`${q}|${k}`);
      if (!row || !(row.median_ms > 0)) {
        td.style.background = 'rgba(128,128,128,0.15)';
        return;
      }

      const tflops = desc.tflops(row);
      const frac   = Math.min(1, tflops / anchor);
      td.style.background = perfColor(frac);
      td.style.color      = cellTextColor(frac);
      td.style.cursor     = 'default';

      const pct = Math.round(frac * 100);
      td.innerHTML = `<div style="line-height:1.2">${tflops.toFixed(1)}<br><span style="opacity:0.8">${pct}%</span></div>`;

      // Tooltip.
      if (desc.tooltip) {
        const lines = desc.tooltip(row);
        lines.unshift(`TFLOPS (matrix): ${tflops.toFixed(2)}`);
        td.title = lines.join('\n');
      }
    });
  });

  container.appendChild(tbl);
}

// ---------------------------------------------------------------------------
// Rendering: 3-D bars via Plotly
// ---------------------------------------------------------------------------

function render3D(container, seqQ, seqK, index, desc, anchor) {
  if (typeof Plotly === 'undefined') {
    container.textContent = 'Plotly.js not loaded — 3D view unavailable.';
    return;
  }
  const z = seqQ.map(q =>
    seqK.map(k => {
      const row = index.get(`${q}|${k}`);
      return (row && row.median_ms > 0) ? desc.tflops(row) : null;
    })
  );
  const div = document.createElement('div');
  div.style.cssText = 'width:320px;height:280px;';
  container.appendChild(div);

  Plotly.newPlot(div, [{
    type: 'surface',
    x: seqK,
    y: seqQ,
    z,
    colorscale: [
      [0,    'hsl(0,90%,45%)'],
      [0.25, 'hsl(30,90%,50%)'],
      [0.5,  'hsl(60,90%,50%)'],
      [0.75, 'hsl(120,70%,40%)'],
      [1,    'hsl(220,80%,45%)'],
    ],
    cmin: 0,
    cmax: anchor,
    showscale: false,
    hovertemplate: 'sk=%{x}<br>sq=%{y}<br>TFLOPS=%{z:.2f}<extra></extra>',
  }], {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    scene: {
      xaxis: { title: 'seqlen_k' },
      yaxis: { title: 'seqlen_q' },
      zaxis: { title: 'TFLOPS' },
    },
  }, { responsive: true, displayModeBar: false });
}

// ---------------------------------------------------------------------------
// Autozoom rendering
// ---------------------------------------------------------------------------

// Returns the max TFLOPS across all seqlen_q×seqlen_k entries in the index.
function _maxTflops(index, desc) {
  let max = 0;
  for (const row of index.values()) {
    if (row && row.median_ms > 0) {
      const t = desc.tflops(row);
      if (t > max) max = t;
    }
  }
  return max;
}

// Overview table: rows = rowDims combos, cols = colDims combos.
// Each cell shows max TFLOPS across seqlen_q×seqlen_k; click to drill in.
function renderAutozoom(grid, layout, anchor) {
  const desc = state.descriptor;
  const { rowCombos, colCombos, seqQ, seqK, cells } = layout;
  const numRowDims = state.rowDims.length;
  const numColDims = state.colDims.length;

  const table = document.createElement('table');
  table.style.cssText = 'border-collapse:separate;border-spacing:4px;';

  // Column header rows.
  for (let di = 0; di < numColDims; di++) {
    const tr = document.createElement('tr');
    for (let ri = 0; ri < numRowDims; ri++) {
      const th = document.createElement('th');
      if (di === 0 && ri === numRowDims - 1) {
        th.textContent = 'features →';
        th.style.cssText = 'text-align:right;opacity:0.7;';
      }
      tr.appendChild(th);
    }
    const dimKey = state.colDims[di];
    const spans = [];
    let curLabel = null, curCount = 0;
    for (const cc of colCombos) {
      const label = _dimLabel(desc, dimKey, cc[dimKey]);
      if (label !== curLabel) {
        if (curCount) spans.push({ label: curLabel, count: curCount });
        curLabel = label;
        curCount = 0;
      }
      curCount++;
    }
    if (curCount) spans.push({ label: curLabel, count: curCount });
    for (const { label, count } of spans) {
      const th = document.createElement('th');
      th.colSpan = count;
      th.textContent = `${dimKey}=${label}`;
      th.style.cssText = 'text-align:center;border-bottom:1px solid currentColor;opacity:0.8;';
      tr.appendChild(th);
    }
    table.appendChild(tr);
  }

  // Data rows.
  for (const rowCombo of rowCombos) {
    const tr = document.createElement('tr');
    for (const rk of state.rowDims) {
      const th = document.createElement('th');
      th.textContent = `${rk}=${_dimLabel(desc, rk, rowCombo[rk])}`;
      th.style.cssText = 'text-align:right;white-space:nowrap;padding-right:6px;';
      tr.appendChild(th);
    }

    for (const colCombo of colCombos) {
      const td = document.createElement('td');
      td.style.cssText = 'padding:0;width:5rem;height:2.4rem;text-align:center;vertical-align:middle;cursor:pointer;';

      const cell = cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === colCombo[k])
      );

      if (cell) {
        const maxT = _maxTflops(cell.index, desc);
        if (maxT > 0) {
          const frac = Math.min(1, maxT / anchor);
          td.style.background = perfColor(frac);
          td.style.color      = cellTextColor(frac);
          const pct = Math.round(frac * 100);
          td.innerHTML = `<div style="line-height:1.2">${maxT.toFixed(1)}<br><span style="opacity:0.8">${pct}%</span></div>`;
          td.title = `Max TFLOPS: ${maxT.toFixed(2)}\nClick to view seqlen matrix`;
          td.addEventListener('click', () => {
            state.autozoom.drilldown = { rowCombo, colCombo };
            renderGrid();
          });
        } else {
          td.style.background = 'rgba(128,128,128,0.15)';
        }
      } else {
        td.style.background = 'rgba(128,128,128,0.15)';
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  grid.appendChild(table);
}

// ---------------------------------------------------------------------------
// Grid rendering
// ---------------------------------------------------------------------------

function renderGrid() {
  const grid = document.getElementById('perf-grid');
  if (!grid) return;
  grid.innerHTML = '';

  if (!state.data || !state.descriptor) {
    grid.textContent = 'No data.';
    return;
  }

  const anchor = computeAnchor(state.data, state);
  const layout = buildLayout(state);
  if (!layout.cells || !layout.cells.length) {
    grid.textContent = 'No matching data for current filters.';
    return;
  }

  // Autozoom: overview or drilldown.
  if (state.displayMode === 'autozoom') {
    const dd = state.autozoom.drilldown;
    if (dd) {
      // Drilldown: show full heatmap for one cell + return button.
      const returnBtn = document.createElement('button');
      returnBtn.textContent = '← Return to overview';
      returnBtn.style.cssText = 'margin-bottom:0.6rem;';
      returnBtn.addEventListener('click', () => {
        state.autozoom.drilldown = null;
        renderGrid();
      });
      grid.appendChild(returnBtn);

      // Title.
      const title = document.createElement('div');
      const label = [
        ...state.rowDims.map(k => `${k}=${_dimLabel(state.descriptor, k, dd.rowCombo[k])}`),
        ...state.colDims.map(k => `${k}=${_dimLabel(state.descriptor, k, dd.colCombo[k])}`),
      ].join(', ');
      title.style.cssText = 'margin-bottom:0.4rem;font-weight:bold;';
      title.textContent = label;
      grid.appendChild(title);

      const cell = layout.cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === dd.rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === dd.colCombo[k])
      );
      if (cell) {
        const container = document.createElement('div');
        renderHeatmap(container, cell.seqQ, cell.seqK, cell.index, state.descriptor, anchor);
        grid.appendChild(container);
      }
    } else {
      renderAutozoom(grid, layout, anchor);
    }

    const legend = document.createElement('div');
    legend.style.cssText = 'margin-top:8px;opacity:0.75;';
    legend.textContent = `Scale anchor: ${anchor.toFixed(1)} TFLOPS (${state.scale.mode})`;
    grid.appendChild(legend);
    return;
  }

  // Build column-header rows.
  const numRowDims = state.rowDims.length;
  const numColDims = state.colDims.length;
  const table = document.createElement('table');
  table.style.cssText = 'border-collapse:separate;border-spacing:4px;';

  // Header rows — one per colDim.
  for (let di = 0; di < numColDims; di++) {
    const tr = document.createElement('tr');
    // Empty corner cells.
    for (let ri = 0; ri < numRowDims; ri++) {
      const th = document.createElement('th');
      if (di === 0 && ri === numRowDims - 1) {
        th.textContent = 'features →';
        th.style.cssText = 'text-align:right;opacity:0.7;';
      }
      tr.appendChild(th);
    }
    // Compute spans for this dim level.
    const dimKey = state.colDims[di];
    // Group consecutive colCombos by the shared prefix up to di.
    const spans = [];
    let curLabel = null, curCount = 0;
    for (const cc of layout.colCombos) {
      const label = _dimLabel(state.descriptor, dimKey, cc[dimKey]);
      if (label !== curLabel) {
        if (curCount) spans.push({ label: curLabel, count: curCount });
        curLabel = label;
        curCount = 0;
      }
      curCount++;
    }
    if (curCount) spans.push({ label: curLabel, count: curCount });

    for (const { label, count } of spans) {
      const th = document.createElement('th');
      th.colSpan = count;
      th.textContent = `${state.colDims[di]}=${label}`;
      th.style.cssText = 'text-align:center;border-bottom:1px solid currentColor;opacity:0.8;';
      tr.appendChild(th);
    }
    table.appendChild(tr);
  }

  // Data rows — one per rowCombo, with a cell per colCombo.
  for (const rowCombo of layout.rowCombos) {
    const tr = document.createElement('tr');

    // Row header cells.
    for (const rk of state.rowDims) {
      const th = document.createElement('th');
      th.textContent = `${rk}=${_dimLabel(state.descriptor, rk, rowCombo[rk])}`;
      th.style.cssText = 'text-align:right;white-space:nowrap;padding-right:6px;';
      tr.appendChild(th);
    }

    // Matrix cells.
    for (const colCombo of layout.colCombos) {
      const td = document.createElement('td');
      td.style.cssText = 'vertical-align:top;padding:2px;';

      const matchingCell = layout.cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === colCombo[k])
      );
      if (matchingCell) {
        const { seqQ, seqK, index } = matchingCell;
        if (state.displayMode === '3d') {
          render3D(td, seqQ, seqK, index, state.descriptor, anchor);
        } else {
          renderHeatmap(td, seqQ, seqK, index, state.descriptor, anchor);
        }
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  grid.appendChild(table);

  // Scale legend.
  const legend = document.createElement('div');
  legend.style.cssText = 'margin-top:8px;opacity:0.75;';
  legend.textContent = `Scale anchor: ${anchor.toFixed(1)} TFLOPS (${state.scale.mode})`;
  grid.appendChild(legend);
}

// ---------------------------------------------------------------------------
// Dimension label helper
// ---------------------------------------------------------------------------

function _dimLabel(desc, key, value) {
  if (!desc) return String(value);
  const dim = desc.dims.find(d => d.key === key);
  if (!dim) return String(value);
  if (dim.valueLabels) {
    const idx = dim.values.indexOf(value);
    if (idx >= 0 && dim.valueLabels[idx] !== undefined) return dim.valueLabels[idx];
  }
  return String(value);
}

// ---------------------------------------------------------------------------
// Initialization and event wiring
// ---------------------------------------------------------------------------

function initPerf() {
  const archSel    = document.getElementById('perf-arch');
  const kernelSel  = document.getElementById('perf-kernel');
  const scaleSel   = document.getElementById('perf-scale');
  const scaleInput = document.getElementById('perf-scale-value');
  const dispSel    = document.getElementById('perf-display');
  const refreshBtn = document.getElementById('perf-refresh');
  const status     = document.getElementById('perf-status');

  async function load() {
    if (!archSel || !kernelSel) return;
    const arch   = archSel.value;
    const kernel = kernelSel.value;
    if (!arch || !kernel) return;
    // Infer mode from the active descriptor's ops set; fall back to kernel.
    const activeDesc = Object.values(DESCRIPTORS)[0];
    const mode = (activeDesc && activeDesc.ops && activeDesc.ops.has(kernel)) ? 'op' : 'kernel';

    state.arch = arch;
    state.kernel = kernel;
    if (status) status.textContent = 'Loading…';

    try {
      state.data = await fetchData(arch, kernel, mode);
      // Pick descriptor by kernel family — currently always flash.
      state.descriptor = DESCRIPTORS['flash'];
      if (state.descriptor) {
        state.colDims = [...state.descriptor.defaultColDims];
        state.rowDims = [...state.descriptor.defaultRowDims];
        state.fixed   = { ...state.descriptor.defaultFixed };
        state.filter  = {};   // clear all value filters on fresh load
        state.autozoom.drilldown = null;
      }
      updateDimPanel();
      renderGrid();
      if (status) status.textContent = `${state.data.rows.length} rows loaded.`;
      // Open the layout panel on first load so users discover it.
      const panel = document.getElementById('perf-layout-panel');
      if (panel && !panel.open) panel.open = true;
    } catch (err) {
      if (status) status.textContent = `Error: ${err.message}`;
    }
  }

  // ---- Dimension layout panel ------------------------------------------------

  // Move dim to a new role (and optionally a position within that role's list).
  function moveDim(key, toRole, beforeKey) {
    // Remove from current role list.
    state.colDims = state.colDims.filter(k => k !== key);
    state.rowDims = state.rowDims.filter(k => k !== key);
    // Also clear fixed value when moving into col/row.
    if (toRole === 'fixed') {
      if (state.fixed[key] === undefined) {
        const vals = state.data && state.data.axes[key];
        state.fixed[key] = vals && vals.length ? vals[0] : null;
      }
      delete state.filter[key];   // fixed select takes over; no grid filter needed
    } else {
      delete state.fixed[key];
      // Preserve any existing filter[key] when moving back to col/row.
    }
    // Insert into target list.
    const targetList = toRole === 'col' ? state.colDims : toRole === 'row' ? state.rowDims : null;
    if (targetList !== null) {
      const idx = beforeKey ? targetList.indexOf(beforeKey) : -1;
      if (idx >= 0) targetList.splice(idx, 0, key);
      else targetList.push(key);
    }
  }

  function buildDimChip(key) {
    const vals = (state.data && state.data.axes[key]) || [];
    const isChecked = !state.filter[key];   // null = all shown = checked

    const chip = document.createElement('span');
    chip.className = 'dim-chip';
    chip.dataset.dim = key;

    // Dropdown for value filtering — always visible, disabled when checkbox is checked.
    // Built before the checkbox so the checkbox handler can reference it.
    const sel = document.createElement('select');
    sel.style.cssText = 'margin-left:0.3rem;';
    sel.disabled = isChecked;
    // Pick the currently filtered value, or the first value as default.
    const currentVal = state.filter[key]
      ? [...state.filter[key]][0]
      : (vals.length ? String(vals[0]) : '');
    vals.forEach(v => {
      const opt = document.createElement('option');
      opt.value = String(v);
      opt.textContent = _dimLabel(state.descriptor, key, v);
      opt.selected = String(v) === currentVal;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', e => {
      e.stopPropagation();
      state.filter[key] = new Set([sel.value]);
      renderGrid();
    });

    // Checkbox toggles between "all values" (dropdown disabled) and "single value" (dropdown enabled).
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = isChecked;
    cb.title = 'Uncheck to filter to a single value';
    cb.addEventListener('change', e => {
      e.stopPropagation();
      if (cb.checked) {
        state.filter[key] = null;
        sel.disabled = true;
      } else {
        state.filter[key] = new Set([sel.value || currentVal]);
        sel.disabled = false;
      }
      renderGrid();
    });

    // Label (not draggable-sensitive).
    const lbl = document.createElement('span');
    lbl.textContent = key;

    // Make chip draggable only from the label, not from controls.
    lbl.draggable = true;
    lbl.style.cursor = 'grab';
    lbl.addEventListener('dragstart', e => {
      e.dataTransfer.setData('text/plain', key);
      e.dataTransfer.effectAllowed = 'move';
      chip.classList.add('dragging');
    });
    chip.addEventListener('dragend', () => chip.classList.remove('dragging'));

    chip.appendChild(cb);
    chip.appendChild(lbl);
    chip.appendChild(sel);
    return chip;
  }

  function buildFixedSelect(key) {
    const vals = (state.data && state.data.axes[key]) || [];
    const wrap = document.createElement('span');
    wrap.style.cssText = 'display:inline-flex;align-items:center;gap:0.3rem;margin:0.15rem;';
    const lbl = document.createElement('span');
    lbl.textContent = key + ':';
    const sel = document.createElement('select');
    vals.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = _dimLabel(state.descriptor, key, v);
      if (state.fixed[key] !== undefined && String(state.fixed[key]) === String(v)) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => {
      const raw = sel.value;
      state.fixed[key] = isNaN(Number(raw)) ? raw : Number(raw);
      renderGrid();
    });
    wrap.appendChild(lbl);
    wrap.appendChild(sel);
    return wrap;
  }

  function wireDropTarget(zone) {
    zone.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const key = e.dataTransfer.getData('text/plain');
      if (!key) return;
      const role = zone.dataset.role;
      // Detect drop before a sibling chip.
      const siblings = [...zone.querySelectorAll('[data-dim]')];
      let beforeKey = null;
      for (const sib of siblings) {
        const r = sib.getBoundingClientRect();
        if (e.clientX < r.left + r.width / 2) { beforeKey = sib.dataset.dim; break; }
      }
      moveDim(key, role, beforeKey);
      updateDimPanel();
      renderGrid();
    });
  }

  function updateDimPanel() {
    if (!state.descriptor || !state.data) return;
    const allDims = state.descriptor.dims.map(d => d.key)
                       .filter(k => (state.data.axes[k] || []).length > 1 ||
                                    state.colDims.includes(k) || state.rowDims.includes(k));

    const colZone   = document.getElementById('perf-col-dims');
    const rowZone   = document.getElementById('perf-row-dims');
    const fixedZone = document.getElementById('perf-fixed-filters');
    if (!colZone || !rowZone || !fixedZone) return;

    colZone.innerHTML   = '';
    rowZone.innerHTML   = '';
    fixedZone.innerHTML = '';

    // Render chips in order for col/row zones; fixed zone gets selects.
    for (const key of state.colDims) {
      if (allDims.includes(key)) colZone.appendChild(buildDimChip(key));
    }
    for (const key of state.rowDims) {
      if (allDims.includes(key)) rowZone.appendChild(buildDimChip(key));
    }
    for (const key of allDims) {
      if (!state.colDims.includes(key) && !state.rowDims.includes(key)) {
        fixedZone.appendChild(buildFixedSelect(key));
      }
    }

    // Wire drop targets once per rebuild (they are fresh DOM nodes).
    [colZone, rowZone, fixedZone].forEach(wireDropTarget);
  }

  if (refreshBtn) refreshBtn.addEventListener('click', load);

  if (scaleSel) {
    scaleSel.addEventListener('change', () => {
      state.scale.mode = scaleSel.value;
      if (scaleInput) scaleInput.style.display = scaleSel.value === 'user' ? 'inline' : 'none';
      renderGrid();
    });
  }
  if (scaleInput) {
    scaleInput.addEventListener('change', () => {
      state.scale.userValue = parseFloat(scaleInput.value) || null;
      renderGrid();
    });
  }
  if (dispSel) {
    dispSel.addEventListener('change', () => {
      state.displayMode = dispSel.value;
      state.autozoom.drilldown = null;
      renderGrid();
    });
  }

  // Auto-load on first visit if arch+kernel already selected.
  if (archSel && archSel.value && kernelSel && kernelSel.value) {
    load();
  }
}

document.addEventListener('DOMContentLoaded', initPerf);
