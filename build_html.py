"""
build_html.py — bake a data directory into a standalone interactive page.

    python build_html.py --data data_stub --out heat-cities-shape.html

The grids are quantised to one base-36 character per cell and embedded, so the
page is a single file with no fetches. Run it again after the real Earth Engine
pull to refresh the page with real footprints.
"""

import argparse
import json
import os

import numpy as np

import render_shape as rs

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def encode(grid):
    q = np.clip((grid / max(grid.max(), 1e-9) * 35).round().astype(int), 0, 35)
    return "".join(DIGITS[v] for v in q.ravel())


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — proof of concept</title>
<style>
  :root {
    --ground: #151C26; --ground-up: #1D2633; --rule: #2E3A4A;
    --ink: #E8EDF3; --ink-quiet: #8494A8; --warn: #DE985C;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--ground); color: var(--ink);
    font-family: "Helvetica Neue", Helvetica, Inter, Arial, sans-serif;
    -webkit-font-smoothing: antialiased; }
  .stage { max-width: 1180px; margin: 0 auto; padding: 40px 36px 32px;
    min-height: 100vh; display: flex; flex-direction: column; }
  header { display: flex; align-items: baseline; justify-content: space-between;
    gap: 24px; border-bottom: 1px solid var(--rule); padding-bottom: 18px; }
  h1 { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; margin: 0; }
  .deck { margin: 6px 0 0; font-size: 14px; line-height: 1.5;
    color: var(--ink-quiet); max-width: 60ch; }
  .year { font-size: 76px; font-weight: 200; line-height: 0.9;
    letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
  .cities { flex: 1; display: grid; gap: 24px; justify-items: center;
    align-items: start; padding: 36px 0 28px; }
  .city { text-align: center; width: 100%; }
  canvas { width: 100%; height: auto; display: block; }
  .label { margin-top: 14px; font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
  .readout { margin-top: 7px; font-size: 13.5px; color: var(--ink-quiet);
    font-variant-numeric: tabular-nums; }
  .readout b { color: var(--ink); font-weight: 500; }
  .overflow { min-height: 1.5em; font-size: 12px; }
  .compact { gap: 20px 24px; padding: 24px 0 20px; }
  .compact .label { margin-top: 6px; font-size: 14px; }
  .compact .readout { margin-top: 2px; font-size: 11.5px; }
  .compact .overflow { font-size: 10.5px; }
  footer { border-top: 1px solid var(--rule); padding-top: 20px; display: grid;
    grid-template-columns: 240px 1fr; gap: 36px; align-items: center; }
  .legend-bar { height: 12px; border-radius: 2px; background: linear-gradient(90deg,
    #FFE066 0%, #FBB02D 30%, #EC7026 60%, #CD3420 82%, #960E18 100%); }
  .legend-scale { display: flex; justify-content: space-between; font-size: 11.5px;
    color: var(--ink-quiet); margin-top: 7px; font-variant-numeric: tabular-nums; }
  .transport { display: flex; align-items: center; gap: 18px; }
  button { background: var(--ground-up); color: var(--ink); border: 1px solid var(--rule);
    border-radius: 3px; padding: 9px 20px; font: inherit; font-size: 14px;
    cursor: pointer; min-width: 88px; }
  button:hover { background: #253044; }
  button:focus-visible { outline: 2px solid #FBC02D; outline-offset: 2px; }
  input[type=range] { flex: 1; accent-color: #E8632A; cursor: pointer; }
  .note { grid-column: 1 / -1; font-size: 12px; color: var(--ink-quiet);
    line-height: 1.6; border-top: 1px solid var(--rule); padding-top: 16px; max-width: 88ch; }
  .note b { color: var(--warn); font-weight: 600; }
  .trend { border-top: 1px solid var(--rule); padding-top: 22px; margin-top: 6px; }
  .trend-head { display: flex; align-items: baseline; justify-content: space-between;
    gap: 20px; flex-wrap: wrap; }
  .trend h2 { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .key { display: flex; gap: 18px; font-size: 12px; color: var(--ink-quiet); }
  .key span { display: inline-flex; align-items: center; gap: 7px; }
  .key i { width: 18px; border-top: 2px solid; display: inline-block; }
  .trend-wrap { position: relative; margin-top: 12px; }
  .trend svg { width: 100%; height: auto; display: block; }
  .tip { position: absolute; pointer-events: none; background: var(--ground-up);
    border: 1px solid var(--rule); border-radius: 3px; padding: 7px 10px; font-size: 12px;
    color: var(--ink-quiet); white-space: nowrap; line-height: 1.5;
    transform: translate(-50%, -100%); opacity: 0; transition: opacity .12s; }
  .tip b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
  .table-view { margin-top: 14px; font-size: 12px; color: var(--ink-quiet); }
  .table-view summary { cursor: pointer; }
  .table-view table { border-collapse: collapse; margin-top: 10px;
    font-variant-numeric: tabular-nums; }
  .table-view th, .table-view td { text-align: right; padding: 3px 14px 3px 0; }
  .table-view th:first-child, .table-view td:first-child { text-align: left; }
  .table-view th { color: var(--ink); font-weight: 600; border-bottom: 1px solid var(--rule); }
  @media (max-width: 860px) {
    .stage { padding: 24px 20px; }
    header { flex-direction: column; align-items: flex-start; gap: 10px; }
    .year { font-size: 54px; }
    footer { grid-template-columns: 1fr; gap: 20px; }
  }
</style>
</head>
<body>
<div class="stage">
  <header>
    <div>
      <h1>__TITLE__</h1>
      <p class="deck">One square is __UNIT__ people, placed on the cells where those people
        live, so each city takes the shape of its own built-up area. Each square is coloured
        by __HEATLABEL__ at that spot, so the hot core and the cooler edges show separately.</p>
    </div>
    <div class="year" id="year">__Y0__</div>
  </header>

  <div class="cities" id="cities"></div>

  <section class="trend">
    <div class="trend-head">
      <div>
        <h2>Year by year, 1975–2025</h2>
        <p class="deck" id="trend-deck"></p>
      </div>
      <div class="key">
        <span><i style="border-color:#8494A8"></i>each of the 20 cities</span>
        <span><i style="border-color:#d95926"></i>20-city mean</span>
      </div>
    </div>
    <div class="trend-wrap">
      <svg id="trend" viewBox="0 0 1180 300" role="img"
           aria-label="Mean daily maximum temperature per year, 1975 to 2025, for 20 cities and their mean"></svg>
      <div class="tip" id="tip"></div>
    </div>
    <details class="table-view">
      <summary>Table view — decade means per city</summary>
      <div id="trend-table"></div>
    </details>
  </section>

  <footer>
    <div>
      <div class="legend-bar"></div>
      <div class="legend-scale"><span id="lo"></span><span id="hi"></span></div>
    </div>
    <div class="transport">
      <button id="play">Pause</button>
      <input type="range" id="scrub" min="__Y0__" max="__Y1__" step="1" value="__Y0__" aria-label="Year">
    </div>
    <p class="note" id="note"></p>
  </footer>
</div>

<script>
const DATA = __DATA__;
const GRID = __GRID__;
const PER_SQUARE = __PERSQ__;
const LO = __LO__, HI = __HI__;
const DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz";
const SECONDS_PER_LOOP = 18;
const CELL = 22, PAD = 0.80;
const SPREAD = __SPREAD__;

const EPOCHS = DATA.epochs;
const HEAT_YEARS = DATA.heat_years;
const Y0 = EPOCHS[0], Y1 = EPOCHS[EPOCHS.length - 1];

const RAMP = [[0,[255,224,102]],[0.30,[251,176,45]],[0.60,[236,112,38]],
              [0.82,[205,52,32]],[1,[150,14,24]]];

function rampColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [p0, c0] = RAMP[i-1], [p1, c1] = RAMP[i];
      const f = (t - p0) / (p1 - p0);
      const m = c0.map((c, j) => Math.round(c + (c1[j] - c) * f));
      return `rgb(${m[0]},${m[1]},${m[2]})`;
    }
  }
  return "rgb(150,14,24)";
}

function atYear(series, year, axis = EPOCHS) {
  if (year <= axis[0]) return series[0];
  if (year >= axis[axis.length - 1]) return series[series.length - 1];
  for (let i = 1; i < axis.length; i++) {
    if (year <= axis[i]) {
      const f = (year - axis[i-1]) / (axis[i] - axis[i-1]);
      return series[i-1] + (series[i] - series[i-1]) * f;
    }
  }
}

DATA.cities.forEach(c => {
  c.fields = c.grids.map(s => Array.from(s, ch => DIGITS.indexOf(ch)));
});

const host = document.getElementById("cities");
host.style.gridTemplateColumns = `repeat(${Math.min(DATA.cities.length, 5)}, 1fr)`;
if (DATA.cities.length > 5) host.classList.add("compact");

DATA.cities.forEach(c => {
  const el = document.createElement("div");
  el.className = "city";
  el.innerHTML = `<canvas width="${GRID*CELL}" height="${GRID*CELL}"></canvas>
    <div class="label">${c.name}</div>
    <div class="readout"><b class="pop"></b> people &middot; <b class="heat"></b> across the city</div>
    <div class="readout overflow"></div>`;
  host.appendChild(el);
  c.ctx = el.querySelector("canvas").getContext("2d");
  c.popEl = el.querySelector(".pop");
  c.heatEl = el.querySelector(".heat");
  c.overflowEl = el.querySelector(".overflow");
});

document.getElementById("lo").textContent = LO + DATA.unit;
document.getElementById("hi").textContent = HI + DATA.unit;
document.getElementById("note").innerHTML =
  (DATA.synthetic ? "<b>" + DATA.source_note + "</b> " : DATA.source_note + " ") +
  "Reanalysis temperature does not resolve the urban heat island, so fill shows " +
  "regional warming rather than the city\\u2019s own heat.";

// interpolate the density field, rank cells, draw the top N
function draw(year) {
  document.getElementById("year").textContent = Math.round(year);
  markYear(year);                      // the chart shows where the animation is
  DATA.cities.forEach(c => {
    const pop = atYear(c.pop, year);
    const heat = atYear(c.heat, year, HEAT_YEARS);
    let field;
    if (year <= Y0) field = c.fields[0];
    else if (year >= Y1) field = c.fields[c.fields.length - 1];
    else {
      let i = 1;
      while (year > EPOCHS[i]) i++;
      const f = (year - EPOCHS[i-1]) / (EPOCHS[i] - EPOCHS[i-1]);
      const a = c.fields[i-1], b = c.fields[i];
      field = a.map((v, k) => v * (1 - f) + b[k] * f);
    }

    // dividing by a fixed per-cell weight scatters squares toward the fringe,
    // so the block reaches the real edge of the built-up area
    const fmax = Math.max(...field);
    const score = field.map((v, k) =>
      v > 0.05 * fmax ? v / Math.pow(c.dither[k], SPREAD) : -1);
    const nValid = score.reduce((a, v) => a + (v > 0 ? 1 : 0), 0);
    const n = Math.round(pop / PER_SQUARE);
    let idx = score.map((v, k) => [v, k]).sort((p, q) => q[0] - p[0]).slice(0, Math.min(n, nValid));
    const overflow = n > nValid;
    if (overflow) {
      // more squares than built-up cells: the extras take the empty cells nearest
      // the footprint, ring by ring (same rule as render_shape.py)
      const valid = idx.map(([, k]) => k);
      const rest = [];
      score.forEach((v, k) => {
        if (v > 0) return;
        const r = Math.floor(k / GRID), col = k % GRID;
        let d2 = valid.length ? Infinity : 0;
        valid.forEach(j => {
          const dr = r - Math.floor(j / GRID), dc = col - j % GRID;
          d2 = Math.min(d2, dr * dr + dc * dc);
        });
        rest.push([d2, -field[k], c.dither[k], k]);
      });
      rest.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
      idx = idx.concat(rest.slice(0, n - nValid).map(p => [0, p[3]]));
    }

    const ctx = c.ctx;
    ctx.clearRect(0, 0, GRID * CELL, GRID * CELL);
    const s = CELL * PAD;
    let tLo = 99, tHi = -99;
    idx.forEach(([, k]) => {
      const r = Math.floor(k / GRID), col = k % GRID;
      const t = heat + c.offset[k];          // city mean + this cell's heat-island offset
      if (t < tLo) tLo = t;
      if (t > tHi) tHi = t;
      ctx.fillStyle = rampColor((t - LO) / (HI - LO));
      ctx.fillRect(col * CELL, r * CELL, s, s);
    });

    c.popEl.textContent = (pop / 1e6).toFixed(1) + "M";
    c.heatEl.textContent = idx.length
      ? tLo.toFixed(1) + "\u2013" + tHi.toFixed(1) + DATA.unit : "\u2014";
    c.overflowEl.textContent = overflow
      ? "squares extend past the built-up area (too dense for the grid)" : "";
  });
}

// ---- trend chart -------------------------------------------------------
// 20 cities as context, the 20-city mean as the story. Not 20 categorical
// hues (past the 8-slot ceiling) and not coloured by temperature either:
// the y position already encodes that, so hue would re-encode it.
const NS = "http://www.w3.org/2000/svg";
const TW = 1180, TH = 300, TM = {t: 14, r: 128, b: 34, l: 46};
const CONTEXT = "#8494A8", ACCENT = "#d95926";
const tYears = HEAT_YEARS;
const tSeries = DATA.cities.map(c => ({name: c.name, v: c.heat}));
const tMean = tYears.map((_, i) => tSeries.reduce((s, c) => s + c.v[i], 0) / tSeries.length);
const tAll = [].concat(...tSeries.map(s => s.v));
const tLo = Math.floor(Math.min(...tAll)) - 0.5, tHi = Math.ceil(Math.max(...tAll)) + 0.5;
const tx = y => TM.l + (y - tYears[0]) / (tYears[tYears.length-1] - tYears[0]) * (TW - TM.l - TM.r);
const ty = v => TM.t + (tHi - v) / (tHi - tLo) * (TH - TM.t - TM.b);
const svg = document.getElementById("trend");
const tip = document.getElementById("tip");

function svgEl(tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;   // names are data, never markup
  svg.appendChild(n);
  return n;
}

function linePath(vals, stroke, width, opacity) {
  const d = vals.map((v, i) => (i ? "L" : "M") + tx(tYears[i]).toFixed(1) + " " + ty(v).toFixed(1)).join(" ");
  return svgEl("path", {d: d, fill: "none", stroke: stroke, "stroke-width": width,
                        "stroke-opacity": opacity, "stroke-linejoin": "round", "stroke-linecap": "round"});
}

// recessive grid: solid hairlines, clean tick values
for (let v = Math.ceil(tLo / 2) * 2; v <= tHi; v += 2) {
  svgEl("line", {x1: TM.l, x2: TW - TM.r, y1: ty(v), y2: ty(v), stroke: "#2E3A4A", "stroke-width": 1});
  svgEl("text", {x: TM.l - 10, y: ty(v) + 4, fill: "#8494A8", "font-size": 11, "text-anchor": "end"}, v + "°");
}
for (let y = tYears[0]; y <= tYears[tYears.length-1]; y += 10)
  svgEl("text", {x: tx(y), y: TH - 12, fill: "#8494A8", "font-size": 11, "text-anchor": "middle"}, y);

const yearMark = svgEl("line", {x1: tx(tYears[0]), x2: tx(tYears[0]), y1: TM.t, y2: TH - TM.b,
                                stroke: "#E8EDF3", "stroke-width": 1, "stroke-opacity": 0.28});
tSeries.forEach(s => linePath(s.v, CONTEXT, 1.4, 0.45));
linePath(tMean, ACCENT, 2, 1);

// direct labels: the extremes and the mean. Text wears ink; the dot carries colour.
const lastI = tYears.length - 1;
const hottest = tSeries.reduce((a, b) => b.v[lastI] > a.v[lastI] ? b : a);
const coolest = tSeries.reduce((a, b) => b.v[lastI] < a.v[lastI] ? b : a);
[[hottest.name, hottest.v[lastI], CONTEXT, "#8494A8"],
 ["20-city mean", tMean[lastI], ACCENT, "#E8EDF3"],
 [coolest.name, coolest.v[lastI], CONTEXT, "#8494A8"]].forEach(([nm, v, dot, ink]) => {
  svgEl("circle", {cx: tx(tYears[lastI]), cy: ty(v), r: 4, fill: dot, stroke: "#151C26", "stroke-width": 2});
  svgEl("text", {x: tx(tYears[lastI]) + 11, y: ty(v) + 4, "font-size": 12, fill: ink},
        nm + " " + v.toFixed(1) + "°");
});

const cross = svgEl("line", {x1: 0, x2: 0, y1: TM.t, y2: TH - TM.b, stroke: "#E8EDF3",
                             "stroke-width": 1, "stroke-opacity": 0, "pointer-events": "none"});
const crossDot = svgEl("circle", {r: 4, fill: ACCENT, stroke: "#151C26", "stroke-width": 2,
                                  opacity: 0, "pointer-events": "none"});
const hit = svgEl("rect", {x: TM.l, y: TM.t, width: TW - TM.l - TM.r, height: TH - TM.t - TM.b,
                           fill: "transparent", tabindex: 0,
                           "aria-label": "Crosshair: arrow keys step through the years"});

function markYear(y) {
  const px = tx(Math.max(tYears[0], Math.min(tYears[lastI], y)));
  yearMark.setAttribute("x1", px);
  yearMark.setAttribute("x2", px);
}

// crosshair snaps to the nearest year; the nearest city comes to the pointer,
// so no line has to be landed on to be read
function showAt(i, svgY) {
  const px = tx(tYears[i]);
  cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("stroke-opacity", 0.45);
  crossDot.setAttribute("cx", px); crossDot.setAttribute("cy", ty(tMean[i])); crossDot.setAttribute("opacity", 1);
  let near = null;
  if (svgY !== undefined) {
    let best = Infinity;
    tSeries.forEach(s => {
      const dy = Math.abs(ty(s.v[i]) - svgY);
      if (dy < best) { best = dy; near = s; }
    });
  }
  tip.textContent = "";
  const mv = document.createElement("b");
  mv.textContent = tMean[i].toFixed(1) + "°C";
  tip.append(tYears[i] + "  ", mv, " 20-city mean");
  if (near) {
    const cv = document.createElement("b");
    cv.textContent = near.v[i].toFixed(1) + "°C";
    tip.append(document.createElement("br"), cv, " " + near.name);
  }
  const box = svg.getBoundingClientRect(), k = box.width / TW;
  tip.style.left = (px * k) + "px";
  tip.style.top = (ty(tMean[i]) * k - 14) + "px";
  tip.style.opacity = 1;
}

function hideCross() {
  cross.setAttribute("stroke-opacity", 0);
  crossDot.setAttribute("opacity", 0);
  tip.style.opacity = 0;
}

let focusI = Math.round(lastI / 2);
hit.addEventListener("pointermove", ev => {
  const box = svg.getBoundingClientRect(), k = TW / box.width;
  const sx = (ev.clientX - box.left) * k, sy = (ev.clientY - box.top) * k;
  const span = (tYears[lastI] - tYears[0]) * (sx - TM.l) / (TW - TM.l - TM.r);
  focusI = Math.max(0, Math.min(lastI, Math.round(span)));
  showAt(focusI, sy);
});
hit.addEventListener("pointerleave", hideCross);
hit.addEventListener("focus", () => showAt(focusI));
hit.addEventListener("blur", hideCross);
hit.addEventListener("keydown", ev => {
  if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
  focusI = Math.max(0, Math.min(lastI, focusI + (ev.key === "ArrowRight" ? 1 : -1)));
  showAt(focusI);
  ev.preventDefault();
});

// the claim in the deck is measured, not assumed: some of these cities cooled
const decades = s => [s.v.slice(0, 10).reduce((a, b) => a + b, 0) / 10,
                      s.v.slice(-10).reduce((a, b) => a + b, 0) / 10];
const warmed = tSeries.filter(s => { const d = decades(s); return d[1] > d[0]; }).length;
const meanEarly = tMean.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
const meanLate = tMean.slice(-10).reduce((a, b) => a + b, 0) / 10;
document.getElementById("trend-deck").textContent =
  "Each thin line is one city's mean daily maximum for that year; the orange line is the 20-city mean, "
  + meanEarly.toFixed(1) + "° across 1975–1984 and " + meanLate.toFixed(1) + "° across 2016–2025. "
  + warmed + " of the " + tSeries.length + " cities are hotter in the recent decade than the first; "
  + (tSeries.length - warmed) + " are not. Single years swing with El Niño, so the trend lives in the "
  + "decade means rather than any one year.";

(function buildTable() {
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["City", "1975–1984", "2016–2025", "Change"].forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  });
  table.appendChild(head);
  tSeries.concat([{name: "20-city mean", v: tMean}]).forEach(s => {
    const d = decades(s), tr = document.createElement("tr");
    [s.name, d[0].toFixed(1) + "°", d[1].toFixed(1) + "°",
     (d[1] - d[0] >= 0 ? "+" : "") + (d[1] - d[0]).toFixed(2) + "°"].forEach(val => {
      const td = document.createElement("td");
      td.textContent = val;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  document.getElementById("trend-table").appendChild(table);
})();

let year = Y0;
let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let last = null;
const playBtn = document.getElementById("play");
const scrub = document.getElementById("scrub");
playBtn.textContent = playing ? "Pause" : "Play";

function tick(now) {
  if (last === null) last = now;
  const dt = (now - last) / 1000;
  last = now;
  if (playing) {
    year += dt * (Y1 - Y0) / SECONDS_PER_LOOP;
    if (year > Y1) year = Y0;
    scrub.value = Math.round(year);
    draw(year);
  }
  requestAnimationFrame(tick);
}

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
scrub.addEventListener("input", () => {
  playing = false;
  playBtn.textContent = "Play";
  year = Number(scrub.value);
  draw(year);
});

draw(year);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main(data_dir, out_path, lo, hi, per_square):
    meta = rs.load(data_dir)
    per_square = rs.nice_unit(meta, per_square)
    cell_vals = []
    for c in meta["cities"]:
        occ = c["grids"][-1] > 0.02
        off = c["offset"][occ] if occ.any() else c["offset"].ravel()
        cell_vals += [min(c["heat"]) + float(off.min()), max(c["heat"]) + float(off.max())]
    span = max(cell_vals) - min(cell_vals)
    lo = lo if lo is not None else round(min(cell_vals) - 0.03 * span, 1)
    hi = hi if hi is not None else round(max(cell_vals) + 0.03 * span, 1)

    payload = dict(
        epochs=meta["epochs"], heat_years=meta.get("heat_years", meta["epochs"]),
        unit=meta.get("unit", ""),
        source_note=meta["source_note"], synthetic=meta.get("synthetic", False),
        cities=[dict(name=c["name"], pop=c["pop"], heat=c["heat"],
                     offset=[round(float(v), 2) for v in c["offset"].ravel()],
                     dither=[round(float(v), 3) for v in c["dither"].ravel()],
                     grids=[encode(g) for g in c["grids"]]) for c in meta["cities"]],
    )

    page = (PAGE
            .replace("__TITLE__", meta.get("title", "Growing into the heat"))
            .replace("__HEATLABEL__", meta["heat_label"])
            .replace("__UNIT__", f"{per_square/1000:.0f},000")
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__GRID__", str(rs.GRID))
            .replace("__PERSQ__", str(int(per_square)))
            .replace("__SPREAD__", str(rs.SPREAD))
            .replace("__LO__", str(lo)).replace("__HI__", str(hi))
            .replace("__Y0__", str(meta["epochs"][0]))
            .replace("__Y1__", str(meta["epochs"][-1])))

    open(out_path, "w").write(page)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_dir", default="data_stub")
    p.add_argument("--out", dest="out_path", default="heat-cities-shape.html")
    p.add_argument("--lo", type=float, default=None)
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--per-square", dest="per_square", type=float, default=None)
    main(**vars(p.parse_args()))
