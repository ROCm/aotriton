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


// Map a raw linear fraction [0,1] to a perceptual fraction using log scaling.
// log(1 + k*x)/log(1+k) with k=9 maps 10% linear → ~53% perceptual,
// spreading the color range across the lower-performance majority.
const _LOG_K = 9;
function _logFrac(linearFrac) {
  const x = Math.max(0, Math.min(1, linearFrac));
  return Math.log1p(_LOG_K * x) / Math.log1p(_LOG_K);
}

// Compute the display fraction for tflops/anchor, log-scaled.
function perfFrac(tflops, anchor) {
  return _logFrac(tflops / anchor);
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
  descriptorId: 'flash',
  data: null,           // raw API response {arch, kernel, axes, rows}
  colDims: [],
  rowDims: [],
  fixed: {},
  // filter: per-dim allowed value sets for grid dims (col/row).
  // null = all values shown. A Set means only those values are shown.
  filter: {},
  displayMode: 'autozoom',  // 'heatmap' | '3d' | 'autozoom'
  autozoom: {
    drilldown: null,   // {rowCombo, colCombo} or null for overview
    seqMode: 'max_load',  // 'max_load' = arch-specific sq=sk target; 'max_tflops' = max across all seqlens
  },
  // Level-2 (psel/copt) drilldown — null when not active.
  cellDetail: null,    // {rowCombo, colCombo, seqQ, seqK, row, data, loading}
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

async function fetchCellDetail(row) {
  const mode = state.descriptor && state.descriptor.ops &&
               state.descriptor.ops.has(row._kernel) ? 'op' : 'kernel';
  const params = new URLSearchParams({
    task_id: row.task_id,
    kernel:  row._kernel,
    mode:    mode,
  });
  const resp = await fetch(`/api/perf/cell_detail?${params}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
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

// Enumerate distinct dim-value tuples present in rows, ordered by the
// descriptor's declared values[] order for each key (falls back to natural order).
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
  // Sort by descriptor values[] order for each dim key.
  const desc = state.descriptor;
  result.sort((a, b) => {
    for (const k of dimKeys) {
      const dim = desc && desc.dims.find(d => d.key === k);
      const order = dim ? dim.values : null;
      const ai = order ? order.indexOf(a[k]) : -1;
      const bi = order ? order.indexOf(b[k]) : -1;
      // Known values sort by index; unknowns go last, then by natural order.
      const av = ai >= 0 ? ai : Infinity;
      const bv = bi >= 0 ? bi : Infinity;
      if (av !== bv) return av - bv;
      if (a[k] < b[k]) return -1;
      if (a[k] > b[k]) return 1;
    }
    return 0;
  });
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

// visibleRows: the rows actually rendered in the current view (caller's responsibility).
function computeAnchor(visibleRows, state) {
  const desc = state.descriptor;
  if (!desc) return 1;

  if (state.scale.mode === 'user' && state.scale.userValue > 0) {
    return state.scale.userValue;
  }
  if (state.scale.mode === 'theoretical') {
    const peak = THEORETICAL_PEAK_TFLOPS[state.arch];
    if (peak) {
      // Return a per-dtype Map so each dtype column is normalized to its own peak.
      const dtypes = state.data && state.data.axes.dtype;
      const map = new Map();
      (dtypes || Object.keys(peak)).forEach(d => { if (peak[d]) map.set(d, peak[d]); });
      if (map.size) return map;
    }
    // Fall through to max_observed if arch not found.
  }
  // max_observed: per-dtype max TFLOPS across the rows currently visible.
  // Returns a Map<dtype, number> so each dtype is normalized to its own peak.
  const byDtype = new Map();
  for (const r of visibleRows) {
    if (r.median_ms > 0) {
      const t = desc.tflops(r);
      const d = r.dtype || '';
      if (!byDtype.has(d) || t > byDtype.get(d)) byDtype.set(d, t);
    }
  }
  return byDtype.size ? byDtype : new Map([['', 1]]);
}

// Pick the anchor value for a given dtype from computeAnchor's return value.
// Like computeAnchor but uses _cellTflops() per cell instead of raw row values.
// Used by autozoom overview so the anchor matches what is actually displayed.
function computeAnchorFromCells(cells, state) {
  const desc = state.descriptor;
  if (!desc) return new Map([['', 1]]);

  if (state.scale.mode === 'user' && state.scale.userValue > 0) {
    return state.scale.userValue;
  }
  if (state.scale.mode === 'theoretical') {
    const peak = THEORETICAL_PEAK_TFLOPS[state.arch];
    if (peak) {
      const dtypes = state.data && state.data.axes.dtype;
      const map = new Map();
      (dtypes || Object.keys(peak)).forEach(d => { if (peak[d]) map.set(d, peak[d]); });
      if (map.size) return map;
    }
  }
  // max_observed per dtype, using the same per-cell tflops value as displayed.
  const byDtype = new Map();
  for (const cell of cells) {
    const t = _cellTflops(cell.index, desc);
    if (t > 0) {
      // Determine dtype from combo dims.
      const dtype = (cell.colCombo && cell.colCombo.dtype) ||
                    (cell.rowCombo && cell.rowCombo.dtype) || '';
      if (!byDtype.has(dtype) || t > byDtype.get(dtype)) byDtype.set(dtype, t);
    }
  }
  return byDtype.size ? byDtype : new Map([['', 1]]);
}

// When anchor is a Map (max_observed), look up by dtype; otherwise use directly.
function anchorFor(anchor, dtype) {
  if (!(anchor instanceof Map)) return anchor;
  return anchor.get(dtype) || anchor.get('') || [...anchor.values()][0] || 1;
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
      const a      = anchorFor(anchor, row.dtype);
      const frac   = perfFrac(tflops, a);
      td.style.background = perfColor(frac);
      td.style.color      = cellTextColor(frac);

      const pct = Math.round(tflops / a * 100);
      td.innerHTML = `<div style="line-height:1.2">${tflops.toFixed(1)}<br><span style="opacity:0.8">${pct}%</span></div>`;

      // Level-2 click-through: psel × copt matrix for this (task_id, kernel).
      const cd = desc.cellDetail;
      const supportsL2 = cd && row.task_id != null
                         && cd.kernels && cd.kernels[row._kernel];
      if (supportsL2) {
        td.style.cursor = 'pointer';
        td.addEventListener('click', () => {
          state.cellDetail = { row, loading: true, data: null };
          renderGrid();
          fetchCellDetail(row).then(data => {
            // Bail if user navigated away in the meantime.
            if (state.cellDetail && state.cellDetail.row === row) {
              state.cellDetail = { row, loading: false, data };
              renderGrid();
            }
          }).catch(err => {
            if (state.cellDetail && state.cellDetail.row === row) {
              state.cellDetail = { row, loading: false, data: null, error: String(err) };
              renderGrid();
            }
          });
        });
      } else {
        td.style.cursor = 'default';
      }

      // Tooltip.
      if (desc.tooltip) {
        const lines = desc.tooltip(row);
        lines.unshift(`TFLOPS (matrix): ${tflops.toFixed(2)}`);
        if (supportsL2) lines.push('Click to view psel × copt matrix');
        td.title = lines.join('\n');
      }
    });
  });

  container.appendChild(tbl);
}


// ---------------------------------------------------------------------------
// Rendering: 3-D bars via Plotly (one bar per autozoom cell)
// ---------------------------------------------------------------------------

// Build a single Plotly mesh3d figure where each autozoom cell becomes one
// colored rectangular bar. X axis = colCombo index, Y axis = rowCombo index,
// Z axis = TFLOPS. Each bar is a unit-width box.
function render3D(container, layout, anchor) {
  if (typeof Plotly === 'undefined') {
    container.textContent = 'Plotly.js not loaded — 3D view unavailable.';
    return;
  }

  const desc = state.descriptor;
  const { rowCombos, colCombos, cells } = layout;
  const nCol = colCombos.length;
  const nRow = rowCombos.length;

  // Collect (xi, yi, tflops, color) per cell.
  const xs = [], ys = [], zs = [], colors = [];
  for (let ri = 0; ri < nRow; ri++) {
    for (let ci = 0; ci < nCol; ci++) {
      const rowCombo = rowCombos[ri];
      const colCombo = colCombos[ci];
      const cell = cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === colCombo[k])
      );
      const cellT = cell ? _cellTflops(cell.index, desc) : 0;
      const dtype = colCombo.dtype || rowCombo.dtype;
      const a = anchorFor(anchor, dtype);
      const frac = cellT > 0 ? perfFrac(cellT, a) : 0;
      xs.push(ci);
      ys.push(ri);
      zs.push(cellT > 0 ? cellT : 0);
      colors.push(frac);
    }
  }

  // Build one mesh3d bar per cell using 8 vertices + 12 triangles.
  const BAR_W = 0.8;  // fraction of unit grid spacing
  const allVx = [], allVy = [], allVz = [], allI = [], allJ = [], allK = [];
  const allColor = [], allCustom = [];
  let vOffset = 0;

  // Pre-build per-cell hover label: all feature dims + TFLOPS.
  function _cellHoverLabel(bi) {
    const ri = Math.floor(bi / nCol), ci = bi % nCol;
    const rowCombo = rowCombos[ri], colCombo = colCombos[ci];
    const parts = [];
    for (const k of state.rowDims) parts.push(`${k}=${_dimLabel(desc, k, rowCombo[k])}`);
    for (const k of state.colDims) parts.push(`${k}=${_dimLabel(desc, k, colCombo[k])}`);
    parts.push(`TFLOPS=${zs[bi].toFixed(1)}`);
    return parts.join('<br>');
  }

  for (let bi = 0; bi < xs.length; bi++) {
    const cx = xs[bi], cy = ys[bi], h = zs[bi], c = colors[bi];
    const x0 = cx - BAR_W / 2, x1 = cx + BAR_W / 2;
    const y0 = cy - BAR_W / 2, y1 = cy + BAR_W / 2;
    // 8 corners: bottom face then top face
    allVx.push(x0, x1, x1, x0,  x0, x1, x1, x0);
    allVy.push(y0, y0, y1, y1,  y0, y0, y1, y1);
    allVz.push( 0,  0,  0,  0,   h,  h,  h,  h);
    const label = _cellHoverLabel(bi);
    for (let v = 0; v < 8; v++) { allColor.push(c); allCustom.push(label); }

    // 12 triangles (6 faces × 2 triangles each), using local indices 0–7.
    const tris = [
      [0,1,2],[0,2,3],  // bottom
      [4,6,5],[4,7,6],  // top
      [0,4,5],[0,5,1],  // front
      [2,6,7],[2,7,3],  // back
      [1,5,6],[1,6,2],  // right
      [0,3,7],[0,7,4],  // left
    ];
    for (const [a, b, c2] of tris) {
      allI.push(vOffset + a);
      allJ.push(vOffset + b);
      allK.push(vOffset + c2);
    }
    vOffset += 8;
  }

  // Axis tick labels from combo labels.
  const xTickText  = colCombos.map((cc, ci) =>
    state.colDims.map(k => _dimLabel(desc, k, cc[k])).join('/'));
  const yTickText  = rowCombos.map((rc, ri) =>
    state.rowDims.map(k => _dimLabel(desc, k, rc[k])).join('/'));

  // HSL colorscale matching the hue-wheel.
  const plotlyScale = [
    [0,    'hsl(0,90%,45%)'],
    [0.25, 'hsl(30,90%,50%)'],
    [0.5,  'hsl(60,90%,50%)'],
    [0.75, 'hsl(120,70%,40%)'],
    [1,    'hsl(220,80%,45%)'],
  ];

  const plotDiv = document.createElement('div');
  plotDiv.style.cssText = 'width:100%;height:520px;';
  container.appendChild(plotDiv);

  Plotly.newPlot(plotDiv, [{
    type: 'mesh3d',
    x: allVx, y: allVy, z: allVz,
    i: allI,  j: allJ,  k: allK,
    intensity: allColor,
    colorscale: plotlyScale,
    cmin: 0, cmax: 1,
    showscale: false,
    flatshading: true,
    lighting: { ambient: 0.9, diffuse: 0.3 },
    customdata: allCustom,
    hovertemplate: '%{customdata}<extra></extra>',
  }], {
    margin: { t: 20, b: 0, l: 0, r: 0 },
    scene: {
      xaxis: {
        title: state.colDims.join('/'),
        tickvals: colCombos.map((_, i) => i),
        ticktext: xTickText,
        tickfont: { size: 9 },
      },
      yaxis: {
        title: state.rowDims.join('/'),
        tickvals: rowCombos.map((_, i) => i),
        ticktext: yTickText,
        tickfont: { size: 9 },
      },
      zaxis: { title: 'TFLOPS' },
      aspectmode: 'manual',
      aspectratio: { x: Math.max(1, nCol * 0.4), y: Math.max(1, nRow * 0.4), z: 1 },
    },
  }, { responsive: true, displayModeBar: false });
}

// ---------------------------------------------------------------------------
// Autozoom rendering
// ---------------------------------------------------------------------------

// Default "representative" seqlen for autozoom fixed mode, keyed by arch.
// gfx1100 has limited HBM so 2k×2k is the canonical point; everything else uses 8k×8k.
const _AZ_TARGET_SEQLEN = {
  gfx1100: 2048,
};
const _AZ_DEFAULT_SEQLEN = 8192;

function _azTargetSeqlen() {
  return _AZ_TARGET_SEQLEN[state.arch] || _AZ_DEFAULT_SEQLEN;
}

// Returns the TFLOPS at (sq, sk). If that exact entry is missing, finds the
// closest available sq and sk independently (by absolute distance) and falls
// back to that entry, or 0 if still not found.
function _tflopsAtSeq(index, desc, sq, sk) {
  const row = index.get(`${sq}|${sk}`);
  if (row && row.median_ms > 0) return desc.tflops(row);

  // Collect available seqlens from the index keys.
  const seqQs = new Set(), seqKs = new Set();
  for (const key of index.keys()) {
    const [q, k] = key.split('|').map(Number);
    seqQs.add(q); seqKs.add(k);
  }
  const nearQ = [...seqQs].reduce((best, v) => Math.abs(v - sq) < Math.abs(best - sq) ? v : best, [...seqQs][0]);
  const nearK = [...seqKs].reduce((best, v) => Math.abs(v - sk) < Math.abs(best - sk) ? v : best, [...seqKs][0]);
  const fallback = index.get(`${nearQ}|${nearK}`);
  return (fallback && fallback.median_ms > 0) ? desc.tflops(fallback) : 0;
}

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

// Returns the representative TFLOPS for an autozoom cell based on seqMode.
function _cellTflops(index, desc) {
  if (state.autozoom.seqMode === 'max_tflops') return _maxTflops(index, desc);
  const sq = _azTargetSeqlen(), sk = _azTargetSeqlen();
  return _tflopsAtSeq(index, desc, sq, sk);
}

// Overview table: rows = rowDims combos, cols = colDims combos.
// Each cell shows the representative TFLOPS value; click to drill in.
function renderAutozoom(grid, layout, anchor) {
  const desc = state.descriptor;
  const { rowCombos, colCombos, seqQ, seqK, cells } = layout;
  const numRowDims = state.rowDims.length;
  const numColDims = state.colDims.length;

  const table = document.createElement('table');
  table.style.cssText = 'border-collapse:separate;border-spacing:4px;';

  // Per-column dim-value tuples in colDims order; used both for separators and
  // for the merged-header spans below.
  const colVecs = colCombos.map(cc => state.colDims.map(k => cc[k]));
  const colNeedsSep = colCombos.map((_, ci) => {
    if (ci === 0) return false;
    for (let li = 0; li < state.colDims.length; li++) {
      if (colVecs[ci][li] !== colVecs[ci - 1][li]) return true;
    }
    return false;
  });

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
    const spans = computeHeaderSpans(colVecs, di,
      (vec, l) => _dimLabel(desc, state.colDims[l], vec[l]));
    for (const { label, count, start } of spans) {
      const th = document.createElement('th');
      th.colSpan = count;
      th.textContent = `${dimKey}=${label}`;
      const sep = start > 0 ? 'border-left:2px solid currentColor;' : '';
      th.style.cssText = `text-align:center;border-bottom:1px solid currentColor;opacity:0.8;${sep}`;
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

    for (const [ci, colCombo] of colCombos.entries()) {
      const td = document.createElement('td');
      const sep = colNeedsSep[ci] ? 'border-left:2px solid currentColor;' : '';
      td.style.cssText = `padding:0;width:5rem;height:2.4rem;text-align:center;vertical-align:middle;cursor:pointer;${sep}`;

      const cell = cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === colCombo[k])
      );

      if (cell) {
        const cellT = _cellTflops(cell.index, desc);
        if (cellT > 0) {
          const dtype = colCombo.dtype || rowCombo.dtype;
          const a    = anchorFor(anchor, dtype);
          const frac = perfFrac(cellT, a);
          td.style.background = perfColor(frac);
          td.style.color      = cellTextColor(frac);
          const pct = Math.round(cellT / a * 100);
          td.innerHTML = `<div style="line-height:1.2">${cellT.toFixed(1)}<br><span style="opacity:0.8">${pct}%</span></div>`;
          const sq = _azTargetSeqlen();
          const tooltipLabel = state.autozoom.seqMode === 'max_tflops'
            ? `Max TFLOPS: ${cellT.toFixed(2)}`
            : `TFLOPS @ ${sq}×${sq}: ${cellT.toFixed(2)}`;
          td.title = `${tooltipLabel}\nClick to view seqlen matrix`;
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

// ---------------------------------------------------------------------------
// Generic header-merging helper
// ---------------------------------------------------------------------------

// Compute colSpan groups for a hierarchical-header row at `level`.
//   vectors: array of per-column tuples (each tuple is an array of length L).
//   level:   0..L-1 — which entry of the tuple this header row displays.
//   labelFn: (vec, level) → display label string (defaults to String(vec[level])).
// A new span starts when the label at `level` changes OR any earlier (ancestor)
// level changes — so a child header never spans across a parent's break.
// Returns [{label, count, start}].
function computeHeaderSpans(vectors, level, labelFn) {
  labelFn = labelFn || ((v, l) => String(v[l]));
  const spans = [];
  let curLabel = null, curCount = 0, curStart = 0;
  vectors.forEach((vec, i) => {
    const label = labelFn(vec, level);
    let ancestorBreak = false;
    if (i > 0) {
      const prev = vectors[i - 1];
      for (let li = 0; li < level; li++) {
        if (vec[li] !== prev[li]) { ancestorBreak = true; break; }
      }
    }
    if (i === 0 || label !== curLabel || ancestorBreak) {
      if (curCount) spans.push({ label: curLabel, count: curCount, start: curStart });
      curLabel = label; curCount = 0; curStart = i;
    }
    curCount++;
  });
  if (curCount) spans.push({ label: curLabel, count: curCount, start: curStart });
  return spans;
}

// ---------------------------------------------------------------------------
// Rendering: level-2 (psel × copt) cell detail
// ---------------------------------------------------------------------------

// Build a stable column-vector key from a list of (field, value) pairs.
function _vecKey(fields, src) {
  return fields.map(f => `${f}=${src[f]}`).join('|');
}

function renderCellDetail(container, cd) {
  const desc   = state.descriptor;
  const schema = desc.cellDetail.kernels[cd.row._kernel];

  const back = document.createElement('button');
  back.textContent = '← Return to seqlen matrix';
  back.style.cssText = 'margin-bottom:0.6rem;';
  back.addEventListener('click', () => { state.cellDetail = null; renderGrid(); });
  container.appendChild(back);

  const title = document.createElement('div');
  title.style.cssText = 'margin-bottom:0.4rem;font-weight:bold;';
  title.textContent =
    `${cd.row._kernel} @ sq=${cd.row.seqlen_q} sk=${cd.row.seqlen_k}, ` +
    `hdim=${cd.row.hdim}, ${cd.row.dtype}, causal=${cd.row.causal ? 'T' : 'F'}, ` +
    `task_id=${cd.row.task_id}`;
  container.appendChild(title);

  if (cd.loading) {
    const p = document.createElement('p');
    p.textContent = 'Loading psel × copt detail…';
    container.appendChild(p);
    return;
  }
  if (cd.error) {
    const p = document.createElement('p');
    p.style.color = 'red';
    p.textContent = `Error: ${cd.error}`;
    container.appendChild(p);
    return;
  }
  if (!cd.data) return;

  const cands = cd.data.candidates || [];
  if (!cands.length) {
    const p = document.createElement('p');
    p.textContent = 'No candidate tuning_results rows for this cell.';
    container.appendChild(p);
    return;
  }

  // Build threshold lookup {tc: {tensor: abs_err}}.
  const threshold = {};
  for (const t of (cd.data.thresholds || [])) {
    (threshold[t.test_case] = threshold[t.test_case] || {})[t.tensor] = t.absolute_error;
  }

  // Index by (psel-vector, copt-vector) → candidate.
  const cellByKey = new Map();
  const pselSet = new Map();   // key → ordered field values
  const coptSet = new Map();
  for (const cand of cands) {
    const pkey = _vecKey(schema.psels, cand.psels);
    const ckey = _vecKey(schema.copts, cand.copts);
    if (!pselSet.has(pkey)) pselSet.set(pkey, schema.psels.map(f => cand.psels[f]));
    if (!coptSet.has(ckey)) coptSet.set(ckey, schema.copts.map(f => cand.copts[f]));
    cellByKey.set(`${pkey}||${ckey}`, cand);
  }

  // Lexicographic sort by field values.
  const cmpVec = (a, b) => {
    for (let i = 0; i < a.length; i++) {
      const av = a[i], bv = b[i];
      if (av === bv) continue;
      if (av === undefined || av === null) return 1;
      if (bv === undefined || bv === null) return -1;
      if (av < bv) return -1;
      if (av > bv) return 1;
    }
    return 0;
  };
  const psels = [...pselSet.entries()].sort((a, b) => cmpVec(a[1], b[1]));
  const copts = [...coptSet.entries()].sort((a, b) => cmpVec(a[1], b[1]));

  // Compute TFLOPS for each cell + max for anchor.
  const cellInfo = new Map();
  let maxT = 0;
  for (const cand of cands) {
    const t = desc.cellDetail.candidateTflops(cand, cd.row);
    const pkey = _vecKey(schema.psels, cand.psels);
    const ckey = _vecKey(schema.copts, cand.copts);
    const passed = desc.passesAccuracy(cand, threshold);
    cellInfo.set(`${pkey}||${ckey}`, { tflops: t, passed, cand });
    if (t > maxT) maxT = t;
  }
  const anchor = maxT > 0 ? maxT : 1;

  // Build table: rows = copts (Y), cols = psels (X).
  const tbl = document.createElement('table');
  tbl.style.cssText = 'border-collapse:collapse;font-size:0.8em;font-family:monospace;';

  const thead = tbl.createTHead();
  const pselVecs = psels.map(([, v]) => v);
  // One header row per psel field, with adjacent identical values merged.
  for (let li = 0; li < schema.psels.length; li++) {
    const tr = thead.insertRow();
    const lbl = document.createElement('th');
    lbl.textContent = li === 0 ? `psel \\ copt` : '';
    lbl.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.7;font-weight:normal;';
    tr.appendChild(lbl);
    // Pad the copt-label columns at the start (we use one column per copt field below).
    for (let cj = 0; cj < schema.copts.length - 1; cj++) {
      tr.appendChild(document.createElement('th'));
    }
    const fieldName = document.createElement('th');
    fieldName.textContent = schema.psels[li];
    fieldName.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.8;font-weight:normal;';
    tr.appendChild(fieldName);
    const spans = computeHeaderSpans(pselVecs, li);
    for (const { label, count, start } of spans) {
      const th = document.createElement('th');
      th.colSpan = count;
      th.textContent = label;
      const sep = start > 0 ? 'border-left:2px solid currentColor;' : '';
      th.style.cssText = `padding:2px 4px;text-align:center;border-bottom:1px solid currentColor;opacity:0.8;font-weight:normal;white-space:nowrap;${sep}`;
      tr.appendChild(th);
    }
  }
  // Header row with copt field names.
  const tr0 = thead.insertRow();
  const lbl0 = document.createElement('th');
  lbl0.textContent = 'copt:';
  lbl0.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.7;font-weight:normal;';
  tr0.appendChild(lbl0);
  for (const cf of schema.copts) {
    const th = document.createElement('th');
    th.textContent = cf;
    th.style.cssText = 'padding:2px 4px;text-align:center;opacity:0.8;font-weight:normal;';
    tr0.appendChild(th);
  }
  for (let i = 0; i < psels.length; i++) {
    tr0.appendChild(document.createElement('th'));
  }

  const tbody = tbl.createTBody();
  for (const [ckey, cvec] of copts) {
    const tr = tbody.insertRow();
    const lbl = document.createElement('th');
    lbl.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.5;font-weight:normal;';
    tr.appendChild(lbl);
    for (const v of cvec) {
      const th = document.createElement('th');
      th.textContent = String(v);
      th.style.cssText = 'padding:2px 6px;text-align:right;opacity:0.8;font-weight:normal;white-space:nowrap;';
      tr.appendChild(th);
    }
    for (const [pkey, pvec] of psels) {
      const td = tr.insertCell();
      td.style.cssText = 'padding:0;width:5rem;height:2.2rem;text-align:center;vertical-align:middle;position:relative;';
      const info = cellInfo.get(`${pkey}||${ckey}`);
      if (!info || !(info.tflops > 0)) {
        td.style.background = 'rgba(128,128,128,0.15)';
        td.title = info ? `index=${info.cand.index}, result=${info.cand.result}` : 'missing';
        continue;
      }
      const frac = perfFrac(info.tflops, anchor);
      td.style.background = perfColor(frac);
      td.style.color      = cellTextColor(frac);
      const pct = Math.round(info.tflops / anchor * 100);
      let html = `<div style="line-height:1.2">${info.tflops.toFixed(1)}<br><span style="opacity:0.8">${pct}%</span></div>`;
      if (!info.passed) {
        // Black ✕ with a white halo via paint-order:stroke — readable on any
        // background (the perfColor() palette spans hues 0..220, so avoiding
        // pure chromatic markers entirely is the safest).
        html += `<svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"
                     viewBox="0 0 10 10" preserveAspectRatio="none">
                  <g stroke="#111" stroke-width="1.1" fill="none"
                     style="paint-order:stroke;stroke-linecap:round">
                    <line x1="1.5" y1="1.5" x2="8.5" y2="8.5"
                          stroke="#fff" stroke-width="2.4"/>
                    <line x1="8.5" y1="1.5" x2="1.5" y2="8.5"
                          stroke="#fff" stroke-width="2.4"/>
                    <line x1="1.5" y1="1.5" x2="8.5" y2="8.5"/>
                    <line x1="8.5" y1="1.5" x2="1.5" y2="8.5"/>
                  </g>
                </svg>`;
      }
      td.innerHTML = html;
      const psStr = schema.psels.map((f, i) => `${f}=${pvec[i]}`).join(', ');
      const coStr = schema.copts.map((f, i) => `${f}=${cvec[i]}`).join(', ');
      td.title = [
        `TFLOPS: ${info.tflops.toFixed(2)} (${pct}% of best)`,
        `psels: ${psStr}`,
        `copts: ${coStr}`,
        `median_ms: ${info.cand.median_ms != null ? info.cand.median_ms.toFixed(4) : 'n/a'}`,
        `index: ${info.cand.index}, result: ${info.cand.result}`,
        info.passed ? 'accuracy: PASS' : 'accuracy: FAIL (✕)',
      ].join('\n');
    }
  }

  container.appendChild(tbl);

  const legend = document.createElement('div');
  legend.style.cssText = 'margin-top:8px;opacity:0.75;';
  const nFail = [...cellInfo.values()].filter(c => !c.passed).length;
  legend.textContent =
    `${cellInfo.size} candidates, ${nFail} failed accuracy gate (✕). ` +
    `Anchor: best in this cell = ${anchor.toFixed(1)} TFLOPS.`;
  container.appendChild(legend);
}

function renderGrid() {
  const grid = document.getElementById('perf-grid');
  if (!grid) return;
  grid.innerHTML = '';

  if (!state.data || !state.descriptor) {
    grid.textContent = 'No data.';
    return;
  }

  // Level-2 view supersedes everything else when active.
  if (state.cellDetail) {
    renderCellDetail(grid, state.cellDetail);
    return;
  }

  const layout = buildLayout(state);
  if (!layout.cells || !layout.cells.length) {
    grid.textContent = 'No matching data for current filters.';
    return;
  }

  // Autozoom and 3D both use the autozoom data path (one value per cell).
  if (state.displayMode === 'autozoom' || state.displayMode === '3d') {
    const dd = state.autozoom.drilldown;
    if (dd) {
      // Anchor from drilldown cell rows only.
      const cell = layout.cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === dd.rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === dd.colCombo[k])
      );
      const visibleRows = cell ? [...cell.index.values()].filter(Boolean) : [];
      const anchor = computeAnchor(visibleRows, state);

      // Drilldown: show full heatmap for one cell + return button.
      const returnBtn = document.createElement('button');
      returnBtn.textContent = '← Return to overview';
      returnBtn.style.cssText = 'margin-bottom:0.6rem;';
      returnBtn.addEventListener('click', () => {
        state.autozoom.drilldown = null;
        renderGrid();
      });
      grid.appendChild(returnBtn);

      const title = document.createElement('div');
      const label = [
        ...state.rowDims.map(k => `${k}=${_dimLabel(state.descriptor, k, dd.rowCombo[k])}`),
        ...state.colDims.map(k => `${k}=${_dimLabel(state.descriptor, k, dd.colCombo[k])}`),
      ].join(', ');
      title.style.cssText = 'margin-bottom:0.4rem;font-weight:bold;';
      title.textContent = label;
      grid.appendChild(title);

      if (cell) {
        const container = document.createElement('div');
        renderHeatmap(container, cell.seqQ, cell.seqK, cell.index, state.descriptor, anchor);
        grid.appendChild(container);
      }

      const legend = document.createElement('div');
      legend.style.cssText = 'margin-top:8px;opacity:0.75;';
      legend.textContent = `Scale anchor: ${anchor instanceof Map ? [...anchor.entries()].map(([d,v])=>`${d}:${v.toFixed(1)}`).join(", ") : anchor.toFixed(1)} TFLOPS (${state.scale.mode})`;
      grid.appendChild(legend);
    } else {
      // Overview: anchor based on the same per-cell tflops values that will be displayed.
      const anchor = computeAnchorFromCells(layout.cells, state);
      if (state.displayMode === '3d') {
        render3D(grid, layout, anchor);
      } else {
        renderAutozoom(grid, layout, anchor);
      }

      const legend = document.createElement('div');
      legend.style.cssText = 'margin-top:8px;opacity:0.75;';
      const azLabel = state.autozoom.seqMode === 'max_tflops'
        ? 'max TFLOPS'
        : `${_azTargetSeqlen()}×${_azTargetSeqlen()} TFLOPS`;
      legend.textContent = `Showing: ${azLabel} — Scale anchor: ${anchor instanceof Map ? [...anchor.entries()].map(([d,v])=>`${d}:${v.toFixed(1)}`).join(", ") : anchor.toFixed(1)} TFLOPS (${state.scale.mode})`;
      grid.appendChild(legend);
    }
    return;
  }

  // Heatmap / 3D: anchor from all visible rows.
  const visibleRows = layout.cells.flatMap(c => [...c.index.values()].filter(Boolean));
  const anchor = computeAnchor(visibleRows, state);

  // Build column-header rows.
  const numRowDims = state.rowDims.length;
  const numColDims = state.colDims.length;
  const table = document.createElement('table');
  table.style.cssText = 'border-collapse:separate;border-spacing:4px;';

  // Per-column dim-value tuples in colDims order; powers separators + spans.
  const colVecs = layout.colCombos.map(cc => state.colDims.map(k => cc[k]));
  const colNeedsSep = layout.colCombos.map((_, ci) => {
    if (ci === 0) return false;
    for (let li = 0; li < state.colDims.length; li++) {
      if (colVecs[ci][li] !== colVecs[ci - 1][li]) return true;
    }
    return false;
  });

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
    const dimKey = state.colDims[di];
    const spans = computeHeaderSpans(colVecs, di,
      (vec, l) => _dimLabel(state.descriptor, state.colDims[l], vec[l]));
    for (const { label, count, start } of spans) {
      const th = document.createElement('th');
      th.colSpan = count;
      th.textContent = `${state.colDims[di]}=${label}`;
      const sep = start > 0 ? 'border-left:2px solid currentColor;' : '';
      th.style.cssText = `text-align:center;border-bottom:1px solid currentColor;opacity:0.8;${sep}`;
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
    for (const [ci, colCombo] of layout.colCombos.entries()) {
      const td = document.createElement('td');
      const sep = colNeedsSep[ci] ? 'border-left:2px solid currentColor;' : '';
      td.style.cssText = `vertical-align:top;padding:2px;${sep}`;

      const matchingCell = layout.cells.find(c =>
        state.rowDims.every(k => c.rowCombo[k] === rowCombo[k]) &&
        state.colDims.every(k => c.colCombo[k] === colCombo[k])
      );
      if (matchingCell) {
        const { seqQ, seqK, index } = matchingCell;
        renderHeatmap(td, seqQ, seqK, index, state.descriptor, anchor);
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  grid.appendChild(table);

  // Scale legend.
  const legend = document.createElement('div');
  legend.style.cssText = 'margin-top:8px;opacity:0.75;';
  legend.textContent = `Scale anchor: ${anchor instanceof Map ? [...anchor.entries()].map(([d,v])=>`${d}:${v.toFixed(1)}`).join(", ") : anchor.toFixed(1)} TFLOPS (${state.scale.mode})`;
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
// URL state persistence
// ---------------------------------------------------------------------------

function pushURLState() {
  const p = new URLSearchParams();
  if (state.arch)         p.set('arch',    state.arch);
  if (state.descriptorId) p.set('module',  state.descriptorId);
  if (state.kernel)       p.set('kernel',  state.kernel);
  p.set('display',  state.displayMode);
  p.set('az_mode',  state.autozoom.seqMode);
  p.set('scale',    state.scale.mode);
  if (state.scale.mode === 'user' && state.scale.userValue)
    p.set('scale_value', state.scale.userValue);
  if (state.colDims.length) p.set('col_dims', state.colDims.join(','));
  if (state.rowDims.length) p.set('row_dims', state.rowDims.join(','));
  // Encode active single-value filters as f_<key>=<value>.
  for (const [k, v] of Object.entries(state.filter)) {
    if (v instanceof Set) p.set(`f_${k}`, [...v][0]);
  }
  history.replaceState(null, '', `?${p}`);
}

function restoreFromURL() {
  const p = new URLSearchParams(location.search);
  return {
    arch:        p.get('arch')        || '',
    kernel:      p.get('kernel')      || '',
    module:      p.get('module')      || '',
    display:     p.get('display')     || '',
    scale:       p.get('scale')       || '',
    scale_value: p.get('scale_value') || '',
    az_mode:     p.get('az_mode')     || '',
    col_dims:    p.get('col_dims')    || '',
    row_dims:    p.get('row_dims')    || '',
    filters:     [...p.entries()]
                   .filter(([k]) => k.startsWith('f_'))
                   .map(([k, v]) => [k.slice(2), v]),
  };
}

// ---------------------------------------------------------------------------
// Single-page HTML export (autozoom only)
// ---------------------------------------------------------------------------

// Collect the col-dim filter: for each col dim, if it has an active single-
// value filter (Set in state.filter), include that value; otherwise include
// all values present in the data axes (empty list = "all allowed" on server).
function _colDimFilters() {
  const filters = {};
  for (const k of state.colDims) {
    const f = state.filter[k];
    if (f instanceof Set) {
      filters[k] = [...f].map(String);
    } else {
      filters[k] = [];  // empty = all values
    }
  }
  return filters;
}

// Build the URL params object that the exported HTML should start with.
function _exportURLParams() {
  const p = {};
  if (state.arch)    p.arch    = state.arch;
  if (state.kernel)  p.kernel  = state.kernel;
  if (state.descriptorId) p.module = state.descriptorId;
  p.display  = 'autozoom';  // export only supports autozoom
  p.az_mode  = state.autozoom.seqMode;
  p.scale    = state.scale.mode;
  if (state.scale.mode === 'user' && state.scale.userValue)
    p.scale_value = String(state.scale.userValue);
  if (state.colDims.length) p.col_dims = state.colDims.join(',');
  if (state.rowDims.length) p.row_dims = state.rowDims.join(',');
  for (const [k, v] of Object.entries(state.filter)) {
    if (v instanceof Set) p[`f_${k}`] = [...v][0];
  }
  return p;
}

async function exportZip(statusEl) {
  if (!state.data) return;
  if (statusEl) statusEl.textContent = 'Building export (all arches/kernels)…';

  const body = {
    col_dim_filters: _colDimFilters(),
    url_params:      _exportURLParams(),
  };

  try {
    const resp = await fetch('/api/perf/export_zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      if (statusEl) statusEl.textContent = `Export failed: ${err.error || resp.statusText}`;
      return;
    }
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = 'aotriton_perf.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (statusEl) statusEl.textContent = 'Export downloaded.';
  } catch (err) {
    if (statusEl) statusEl.textContent = `Export error: ${err.message}`;
  }
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
  const azModeSel  = document.getElementById('perf-az-mode');
  const azModeLabel = document.getElementById('perf-az-mode-label');
  const refreshBtn = document.getElementById('perf-refresh');
  const exportBtn  = document.getElementById('perf-export');
  const status     = document.getElementById('perf-status');

  function _syncAzModeVisibility() {
    if (azModeLabel) azModeLabel.style.display =
      (state.displayMode === 'autozoom') ? '' : 'none';
  }

  // Apply URL params to selectors before load().
  const saved = restoreFromURL();
  if (saved.module) state.descriptorId = saved.module;
  if (saved.arch   && archSel)   archSel.value   = saved.arch;
  if (saved.kernel && kernelSel) kernelSel.value = saved.kernel;
  if (saved.display && dispSel) {
    dispSel.value = saved.display;
    state.displayMode = saved.display || 'autozoom';
  }
  if (saved.az_mode && azModeSel) {
    azModeSel.value = saved.az_mode;
    state.autozoom.seqMode = saved.az_mode;
  }
  _syncAzModeVisibility();
  if (saved.scale  && scaleSel) {
    scaleSel.value = saved.scale;
    state.scale.mode = saved.scale;
    if (scaleInput) scaleInput.style.display = saved.scale === 'user' ? 'inline' : 'none';
  }
  if (saved.scale_value && scaleInput) {
    scaleInput.value = saved.scale_value;
    state.scale.userValue = parseFloat(saved.scale_value) || null;
  }

  async function load() {
    if (!archSel || !kernelSel) return;
    const arch   = archSel.value;
    const kernel = kernelSel.value;
    if (!arch || !kernel) return;
    // Infer mode from the active descriptor's ops set; fall back to kernel.
    const activeDesc = DESCRIPTORS[state.descriptorId] || DESCRIPTORS[Object.keys(DESCRIPTORS)[0]];
    const mode = (activeDesc && activeDesc.ops && activeDesc.ops.has(kernel)) ? 'op' : 'kernel';

    state.arch = arch;
    state.kernel = kernel;
    if (dispSel) state.displayMode = dispSel.value;
    if (status) status.textContent = 'Loading…';

    try {
      state.data = await fetchData(arch, kernel, mode);
      // Pick descriptor by id (from URL state or default).
      state.descriptor = DESCRIPTORS[state.descriptorId] || DESCRIPTORS[Object.keys(DESCRIPTORS)[0]];
      if (state.descriptor) {
        // Restore col/row dims from URL if present, else use descriptor defaults.
        state.colDims = saved.col_dims
          ? saved.col_dims.split(',').filter(Boolean)
          : [...state.descriptor.defaultColDims];
        state.rowDims = saved.row_dims
          ? saved.row_dims.split(',').filter(Boolean)
          : [...state.descriptor.defaultRowDims];
        state.fixed   = { ...state.descriptor.defaultFixed };
        state.filter  = {};
        // Restore per-dim filters from URL.
        for (const [k, v] of saved.filters) {
          state.filter[k] = new Set([v]);
        }
        state.autozoom.drilldown = null;
      }
      updateDimPanel();
      renderGrid();
      pushURLState();
      if (status) status.textContent = `${state.data.rows.length} rows loaded.`;
      if (exportBtn) { exportBtn.disabled = false; exportBtn.title = ''; }
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
    state.colDims = state.colDims.filter(k => k !== key);
    state.rowDims = state.rowDims.filter(k => k !== key);
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
      pushURLState();
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
      pushURLState();
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
      pushURLState();
    });
  }

  function updateDimPanel() {
    if (!state.descriptor || !state.data) return;
    const allDims = state.descriptor.dims.map(d => d.key)
                       .filter(k => (state.data.axes[k] || []).length > 1 ||
                                    state.colDims.includes(k) || state.rowDims.includes(k));

    const colZone = document.getElementById('perf-col-dims');
    const rowZone = document.getElementById('perf-row-dims');
    if (!colZone || !rowZone) return;

    colZone.innerHTML = '';
    rowZone.innerHTML = '';

    for (const key of state.colDims) {
      if (allDims.includes(key)) colZone.appendChild(buildDimChip(key));
    }
    for (const key of state.rowDims) {
      if (allDims.includes(key)) rowZone.appendChild(buildDimChip(key));
    }

    [colZone, rowZone].forEach(wireDropTarget);
  }

  if (refreshBtn) refreshBtn.addEventListener('click', load);
  if (exportBtn)  exportBtn.addEventListener('click', () => exportZip(status));

  if (scaleSel) {
    scaleSel.addEventListener('change', () => {
      state.scale.mode = scaleSel.value;
      if (scaleInput) scaleInput.style.display = scaleSel.value === 'user' ? 'inline' : 'none';
      renderGrid();
      pushURLState();
    });
  }
  if (scaleInput) {
    scaleInput.addEventListener('change', () => {
      state.scale.userValue = parseFloat(scaleInput.value) || null;
      renderGrid();
      pushURLState();
    });
  }
  if (dispSel) {
    dispSel.addEventListener('change', () => {
      state.displayMode = dispSel.value;
      state.autozoom.drilldown = null;
      _syncAzModeVisibility();
      renderGrid();
      pushURLState();
    });
  }
  if (azModeSel) {
    azModeSel.addEventListener('change', () => {
      state.autozoom.seqMode = azModeSel.value;
      renderGrid();
      pushURLState();
    });
  }

  // Auto-load if arch+kernel are set (either from URL or server-rendered default).
  if (archSel && archSel.value && kernelSel && kernelSel.value) {
    load();
  }
}

document.addEventListener('DOMContentLoaded', initPerf);
