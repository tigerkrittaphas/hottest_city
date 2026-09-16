# Heat sim teaching slide — city growth + warming

Twenty of the world's hottest large cities, 1975–2025, one frame per year. Each
city is drawn as squares laid on its own built-up area, so the block of squares
takes the shape of the city map. One square is a fixed slice of population, placed
on the densest cells first, so the count is the population and the arrangement is
where people live. Every square takes its colour from a white–yellow–red ramp
driven by that year's temperature plus the cell's own heat-island offset.

Outputs: `out/heat-cities.gif`, `out/heat-cities.mp4`, `out/heat-cities-final.png`
(1920×1080, four rows of five) and `page.html`, a standalone interactive page.

## 1. Data sources

Everything comes from Earth Engine. Nothing is drawn by hand.

| Layer | Dataset | Native resolution | Years |
|---|---|---|---|
| Built-up area | `JRC/GHSL/P2023A/GHS_BUILT_S` (R2023A) | 100 m | 1975–2030, every 5 years |
| Population | `JRC/GHSL/P2023A/GHS_POP` (R2023A) | 100 m | 1975–2030, every 5 years |
| City temperature | `ECMWF/ERA5_LAND/DAILY_AGGR` | 11 km (0.1°) | 1950–present, daily |
| Within-city pattern | `MODIS/061/MOD11A2` `LST_Day_1km` | 927 m | 2000–present, 8-day |

Each city is a 75 km box around its centre (`extent_km` in `pull_ghsl_era5.py`),
the same window for every city so tiles stay comparable.

**Which 20 cities.** The 20 cities with 5M+ people (GHS_POP 2025, summed over the
75 km window) with the highest 2021–2025 mean daily maximum temperature, out of 89
large candidates worldwide. `rank_cities.py` computes the ranking and writes
`city_heat_rank.csv`. Cross-checked against Tuholske et al. 2021, PNAS 118(41),
doi:10.1073/pnas.2024792118; the megacity heat-stress study in The Geographical
Journal, doi:10.1111/geoj.12568 (its top five — Chennai, Bangkok, Kolkata,
Karachi, Delhi — are all in the list); and IIED's September 2025 briefing on days
over 35 °C.

Two selection judgements worth knowing:

- **The ranking uses mean daily max, not counts of hot days.** ERA5-Land badly
  undercounts 35 °C days in humid tropical cities: for 2015–2024 IIED's airport
  stations give Manila 53 days a year and Kuala Lumpur 49, where ERA5-Land gives
  9 and 0. Mean daily max is far less sensitive to that smoothing.
- **Luanda ranked 16th but was replaced by Manila**, the next city down. GHSL's
  early population for Luanda is implausible: 0.08M in 1975 spread over 73,552
  populated cells, then a jump from 1.8M (2005) to 4.0M (2010), while its
  built-up fraction barely moves (3.1% → 5.8%).

## 2. Data preprocessing

`pull_ghsl_era5.py` turns those four layers into one `.npz` per city plus a shared
`data/cities.json`. Per city:

- **Built-up density**, one raster per 5-year epoch (11 epochs, 1975–2025).
  `built_surface` is m² built per 100 m cell, so it is divided by 10,000 and
  clamped to 0–1, then sampled onto a 240×240 grid over the box (312.5 m per
  pixel at 75 km) via `sampleRectangle`. Epochs can differ by a pixel, so all are
  cropped to the smallest shared shape.
- **Population**, one total per epoch, summed at 100 m over the box. GHSL stores
  sea and other no-data cells as **−200 rather than masking them**, which made
  early totals negative (Karachi 2020 came out as −14.3M), so cells below zero are
  masked before summing.
- **Temperature**, one value per calendar year, 1975–2025 (51 values): the mean of
  `temperature_2m_max` over that year, averaged across the box at 9 km sampling.
  All 51 years are computed server-side in one request. Single-year values, so
  El Niño years show as real swings rather than a smoothed trend.
- **Within-city offset**, one static raster: the 2015–2024 mean of MODIS daytime
  land surface temperature, minus its own spatial mean, with its spread rescaled
  to at most about 4 °C so it reads as an air-temperature-like spread. Sea and
  never-observed pixels are filled with a sentinel and set to **0 offset**; left
  as raw fill they became −20 °C artefacts that dragged coastal squares cold.

`--resume` reuses rasters and series already written, so adding one city costs one
city's queries rather than a full re-pull. Add `--metric tmax|apparent --thresh N`
to count hot days instead; those runs carry no within-city pattern, because a °C
offset cannot be added to a day count.

`make_stub_masks.py` writes the same file layout from procedurally generated
shapes, for working on the renderer without Earth Engine access. They are **not**
real cities, and every frame built from them is captioned as synthetic, in amber.

## 3. Visualization

`render_shape.py` (GIF, MP4, PNG) and `build_html.py` (interactive page) read the
same data and follow the same rules, so the two agree.

1. **Coarsen** each 240×240 raster to a **16×16 grid** of block means — one square
   per grid cell, sized to read from the back of a room.
2. **Interpolate** between epochs: population, built-up density and the
   silhouette ease from one 5-year snapshot to the next, so the city grows
   smoothly instead of snapping every five years. Temperature is already yearly
   and is read straight off its own timeline.
3. **Count the squares**: `population ÷ per_square`, where the unit is chosen to
   fill about 70% of the grid at the largest city (190,000 people per square for
   these 20; override with `--per-square`).
4. **Place them** on the highest-ranked cells, where the rank is density divided
   by a fixed per-cell weight (`--spread`, default 0.6) that scatters squares
   outward to the real edge of the built-up area. The weight is fixed per cell, so
   squares never flicker between frames.
5. **Overflow**: a city denser than one square per cell (Karachi, Mumbai,
   Kinshasa, Manila by 2025) puts its extra squares in the empty cells nearest the
   footprint, ring by ring, and is marked `*` with a footnote. Without this the
   leftovers filled the grid in row order and drew a rectangle.
6. **Colour** each square by that year's city mean plus the cell's own offset, on
   a white–yellow–red ramp anchored by `--lo`/`--hi`, which the legend prints.
7. **Animate** 51 frames, one per year. `--fps` sets the rate (the slide render
   uses 3, a 17 s loop); the last frame holds about a second before looping.

The page mirrors this in the browser: grids are quantised to one base-36 character
per cell and embedded, so it is a single file with no fetches, with play/pause and
a year scrubber for jumping to a year mid-discussion.

## Running it

```bash
pip install earthengine-api numpy pillow
earthengine authenticate
export EE_PROJECT=your-cloud-project

python rank_cities.py city_heat_rank.csv                              # optional, re-rank
python pull_ghsl_era5.py --out data                                   # full pull
python pull_ghsl_era5.py --out data --resume                          # only what's missing
python render_shape.py --data data --out out --lo 28 --hi 38 --fps 3  # gif + mp4 + png
python build_html.py --data data --out page.html --lo 28 --hi 38
```

Without Earth Engine access:

```bash
python make_stub_masks.py --out data_stub
python render_shape.py --data data_stub --out out
```

Files: `rank_cities.py` (city selection) → `pull_ghsl_era5.py` (data) →
`render_shape.py` / `build_html.py` (output); `make_stub_masks.py` for synthetic
stand-ins.

## Publishing

`site/index.html` is what GitHub Pages serves: a copy of the built `page.html`,
which stays ignored as a build product. Refresh it after a rebuild:

```bash
python build_html.py --data data --out page.html --lo 28 --hi 38
cp page.html site/index.html
git commit -am "Update published page" && git push
```

Pushing to `main` runs `.github/workflows/deploy.yml`, which uploads `site/` to
Pages: <https://tigerkrittaphas.github.io/hottest_city/>

## Encoding choices worth defending

- **One square = 190,000 people**, chosen automatically to fill about 70% of the
  grid at the largest city. Bigger squares read better on a projector and blur the
  silhouette; smaller ones sharpen the shape and stop being countable. `GRID` and
  the unit move together.
- **Cells are ranked by density, not by built-up extent**, so squares land where
  people are. A city that densifies without sprawling gains squares inward.
- **`--spread` (default 0.6)** reaches the real edge of the built-up area instead
  of packing into the core. The cost is that thin features — a river gap, a narrow
  coastal strip — get partly filled in by stray squares.
- **Colour anchors 28–38 °C.** The automatic range (25.8–39.2) left everything
  outside the Middle East and the Sahel pale yellow. The tighter range clamps a
  few extremes — the coolest edges of Hyderabad and Delhi go pure white — and the
  legend prints the anchors so the slide stays honest.
- **16×16 grid at 20 cities**, four rows of five on a 1920×1080 frame. Five or
  fewer cities render on one row at 1600×900.

## Caveats and open decisions

1. **Within-city temperature is half-real, and the caption has to say so.**
   ERA5-Land is 11 km per cell — it cannot vary meaningfully inside a city. The
   spatial pattern therefore comes from MODIS land surface temperature as a fixed
   2015–2024 climatology added to the ERA5 city-mean trend. So the year-to-year
   movement is air temperature and real; the hot core and cool edges are surface
   temperature, rescaled, and do not change across epochs. The alternative —
   Landsat LST per epoch — gives a pattern that genuinely moves, but only from
   1985, and still is not air temperature.
2. **Population is everyone in the 75 km box**, not a city-proper figure, so
   Bangkok reads 20.0M and Delhi 34.4M. The largest metros run past the box: a
   110 km window would add 40% to Kolkata, 23% to Delhi, 18% to Jakarta and 17%
   to Ho Chi Minh City. Kept at 75 km so every tile covers the same area.
3. **Population source.** GHSL figures differ substantially from UN World
   Urbanization Prospects for some cities. GHSL is used throughout; if the slide
   quotes UN numbers anywhere, they will not match.
4. **The GIF and the page can place squares differently.** The scatter seed comes
   from Python's `hash()`, which is randomised per process, so tie-breaks differ
   between runs. Square counts always match. A fixed seed is a one-line fix.
5. **Single-year temperatures move both ways.** With yearly values some cities
   read cooler in 2025 than in 2020 (Delhi, Hyderabad, Kolkata). That is weather,
   not a bug; a 5-year mean would smooth it at the cost of honesty about variance.
