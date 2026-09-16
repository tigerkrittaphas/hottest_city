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
    #FFFFFF 0%, #FFE9A8 32%, #FBC02D 55%, #E8632A 80%, #B3161B 100%); }
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

const RAMP = [[0,[255,255,255]],[0.32,[255,233,168]],[0.55,[251,192,45]],
              [0.80,[232,99,42]],[1,[179,22,27]]];

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
  return "rgb(179,22,27)";
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
