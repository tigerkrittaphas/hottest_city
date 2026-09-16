"""
pull_ghsl_era5.py — real data for render_shape.py and build_html.py, via Earth Engine.

Emits, per city:
    data/<slug>.npz    density (T,H,W) built-up fraction 0-1
                       heat_offset (H,W) within-city degC offset
                       years (T,), pop (T,), heat (Y,), heat_years (Y,), metric
                       (the series are stored so --resume can skip the queries)
    data/cities.json   epochs[T], heat_years[Y], unit, labels; per city name, country,
                       extent_km, pop[T] (5-yearly GHSL), heat[Y] (yearly ERA5-Land)

Sources
    GHS-BUILT-S R2023A   JRC/GHSL/P2023A/GHS_BUILT_S/<year>   m2 per 100m cell
    GHS-POP     R2023A   JRC/GHSL/P2023A/GHS_POP/<year>       persons per cell
    ERA5-Land            ECMWF/ERA5_LAND/DAILY_AGGR           1950-present
    MODIS LST            MODIS/061/MOD11A2                    2000-present

Setup (once):
    pip install earthengine-api numpy
    earthengine authenticate
    # needs a Cloud project with the Earth Engine API enabled
    export EE_PROJECT=your-project-id

Run:
    python pull_ghsl_era5.py --out data                    # mean daily max, degC
    python pull_ghsl_era5.py --out data --metric apparent --thresh 40

--metric tmax / apparent count days above --thresh instead. A degC within-city
offset cannot be added to a day count, so those runs write a zero offset and
every square in a city takes the same colour.

NOTE: written against the EE Python API but not executed in this sandbox (no
network to EE). Expect to shake out one or two API details on first run.
"""

import argparse
import json
import os

import numpy as np
import ee

# lon, lat of the urban core. extent_km must be wide enough for the 2025
# footprint or the tile clips — check the last frame and widen if needed.
# The 20 cities of 5M+ people (GHSL 2025, 75 km window) with the highest
# 2021-2025 mean daily maximum temperature, in rank order; see rank_cities.py.
CITIES = [
    dict(slug="khartoum", name="Khartoum", country="Sudan",
         lon=32.56, lat=15.501, extent_km=75),
    dict(slug="dubai", name="Dubai", country="United Arab Emirates",
         lon=55.271, lat=25.205, extent_km=75),
    dict(slug="kano", name="Kano", country="Nigeria",
         lon=8.592, lat=12.002, extent_km=75),
    dict(slug="riyadh", name="Riyadh", country="Saudi Arabia",
         lon=46.675, lat=24.713, extent_km=75),
    dict(slug="jeddah", name="Jeddah", country="Saudi Arabia",
         lon=39.192, lat=21.485, extent_km=75),
    dict(slug="baghdad", name="Baghdad", country="Iraq",
         lon=44.366, lat=33.315, extent_km=75),
    dict(slug="ahmedabad", name="Ahmedabad", country="India",
         lon=72.571, lat=23.023, extent_km=75),
    dict(slug="bangkok", name="Bangkok", country="Thailand",
         lon=100.501, lat=13.756, extent_km=75),
    dict(slug="chennai", name="Chennai", country="India",
         lon=80.271, lat=13.083, extent_km=75),
    dict(slug="yangon", name="Yangon", country="Myanmar",
         lon=96.195, lat=16.866, extent_km=75),
    dict(slug="hcmc", name="Ho Chi Minh City", country="Vietnam",
         lon=106.66, lat=10.762, extent_km=75),
    dict(slug="surat", name="Surat", country="India",
         lon=72.831, lat=21.17, extent_km=75),
    dict(slug="kolkata", name="Kolkata", country="India",
         lon=88.364, lat=22.573, extent_km=75),
    dict(slug="karachi", name="Karachi", country="Pakistan",
         lon=67.01, lat=24.861, extent_km=75),
    dict(slug="mumbai", name="Mumbai", country="India",
         lon=72.878, lat=19.076, extent_km=75),
    dict(slug="kinshasa", name="Kinshasa", country="DR Congo",
         lon=15.266, lat=-4.442, extent_km=75),
    dict(slug="delhi", name="Delhi", country="India",
         lon=77.209, lat=28.614, extent_km=75),
    dict(slug="hyderabad", name="Hyderabad", country="India",
         lon=78.487, lat=17.385, extent_km=75),
    dict(slug="jakarta", name="Jakarta", country="Indonesia",
         lon=106.845, lat=-6.208, extent_km=75),
    # Luanda ranked 16th but GHSL's early population there is implausible
    # (0.08M in 1975, 1.8M -> 4.0M between 2005 and 2010), so Manila takes its place.
    dict(slug="manila", name="Manila", country="Philippines",
         lon=120.984, lat=14.6, extent_km=75),
]

EPOCHS = list(range(1975, 2026, 5))   # GHSL runs to 2030; 2025/2030 are modelled
HEAT_YEARS = list(range(1975, 2026))   # ERA5-Land is daily, so heat can be yearly
N = 240                                # grid cells per tile side
CELL_M2 = 100.0 * 100.0                # GHS-BUILT-S cell area
NODATA = -9999.0                       # sampleRectangle fill for masked pixels


def window(city):
    """Square region around the core, in metres, on a local equal-area-ish CRS."""
    half = city["extent_km"] * 500.0
    pt = ee.Geometry.Point([city["lon"], city["lat"]])
    return pt.buffer(half).bounds(), half * 2 / N   # geom, scale in m/px


def built_mask(year, geom, scale):
    img = (ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}")
           .select("built_surface")
           .divide(CELL_M2)             # -> fraction of cell built
           .clamp(0, 1)
           .reproject(crs="EPSG:3857", scale=scale))
    arr = img.sampleRectangle(region=geom, defaultValue=0).get("built_surface")
    return np.asarray(arr.getInfo(), dtype=np.float32)


def population(year, geom):
    img = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count")
    # sea and other no-data cells hold -200 rather than a mask; summed, they go negative
    img = img.updateMask(img.gte(0))
    return float(img.reduceRegion(ee.Reducer.sum(), geom, 100,
                                  maxPixels=1e10).get("population_count").getInfo())


def hot_day(img, metric, thresh):
    """1 where the day's metric exceeds `thresh` (degC), else 0."""
    t = img.select("temperature_2m_max").subtract(273.15)
    if metric == "tmax":
        val = t
    else:
        # BoM apparent temperature: AT = T + 0.33e - 0.70ws - 4.00
        td = img.select("dewpoint_temperature_2m").subtract(273.15)
        e = td.expression(
            "6.105 * exp(17.27 * td / (237.7 + td))", {"td": td})
        ws = img.select("u_component_of_wind_10m").pow(2).add(
            img.select("v_component_of_wind_10m").pow(2)).sqrt()
        val = t.add(e.multiply(0.33)).subtract(ws.multiply(0.70)).subtract(4.0)
    return val.gt(thresh).rename("hot")


def heat_series(geom, metric="tmax_mean", thresh=35.0, years=HEAT_YEARS):
    """One heat value per calendar year, computed server-side in one request.

    tmax_mean: that year's mean daily maximum 2 m air temperature (degC).
    tmax / apparent: days in that year where the metric exceeds `thresh`.
    Single-year values, so El Nino and La Nina years show as real swings.
    """
    col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    def one(y):
        start = ee.Date.fromYMD(y, 1, 1)
        year_days = col.filterDate(start, start.advance(1, "year"))
        if metric == "tmax_mean":
            img = year_days.select("temperature_2m_max").mean().subtract(273.15)
        else:
            img = year_days.map(lambda d: hot_day(d, metric, thresh)).sum()
        return img.reduceRegion(ee.Reducer.mean(), geom, 9000,
                                maxPixels=1e10).values().get(0)

    return [float(v) for v in ee.List(years).map(one).getInfo()]


def heat_offset(geom, scale, years=(2015, 2024)):
    """Within-city heat pattern, in degC, relative to the city's own mean.

    ERA5 is ~28 km, far too coarse to vary inside a city, so the spatial pattern
    comes from MODIS land surface temperature instead: a daytime
    climatology, minus its own spatial mean. Two consequences to caption:
    it is land SURFACE temperature, which runs hotter and more contrasty than
    air temperature, and it is a modern pattern held fixed across all epochs
    (MODIS starts in 2000), so only the city-mean trend moves with time.
    """
    col = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate(f"{years[0]}-01-01", f"{years[1]}-12-31")
           .select("LST_Day_1km"))
    # toFloat widens the band's value range so NODATA is an accepted fill
    lst = (col.mean().multiply(0.02).subtract(273.15).toFloat()
           .reproject(crs="EPSG:3857", scale=scale))
    arr = np.asarray(lst.sampleRectangle(region=geom, defaultValue=NODATA)
                     .get("LST_Day_1km").getInfo(), dtype=np.float32)
    ok = np.isfinite(arr) & (arr > NODATA / 2)
    if not ok.any():
        return np.zeros_like(arr)
    valid = arr[ok]
    # rescale the LST spread to a plausible air-temperature spread
    centred = arr - valid.mean()
    spread = np.percentile(valid, 95) - np.percentile(valid, 5)
    if spread > 0:
        centred *= min(1.0, 4.0 / spread)
    # sea and never-observed pixels carry no heat-island signal, not -30 degC
    centred[~ok] = 0.0
    return centred.astype(np.float32)


def main(out_dir, metric, thresh, resume):
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    os.makedirs(out_dir, exist_ok=True)
    days = metric != "tmax_mean"
    if days:
        heat_label = (f"days per year above {thresh:g}\u00b0C "
                      + ("daily max" if metric == "tmax" else "apparent temperature"))
        pattern_note = "Within-city: not shown for day counts."
    else:
        heat_label = "mean daily maximum temperature"
        pattern_note = "Within-city: MODIS LST 2015\u20132024, held fixed."
    meta = dict(
        title="Built-up extent and heat, 1975\u20132025",
        heat_label=heat_label,
        unit=" days" if days else "\u00b0C",
        epochs=EPOCHS,
        heat_years=HEAT_YEARS,
        source_note=("Population: GHSL R2023A, 5-yearly, interpolated. "
                     "Heat: ERA5-Land, yearly. " + pattern_note),
        cities=[],
    )
    for c in CITIES:
        geom, scale = window(c)
        path = os.path.join(out_dir, c["slug"] + ".npz")
        # --resume reuses whatever a previous run already wrote for this city
        cached = np.load(path) if resume and os.path.exists(path) else None
        if cached is not None and list(cached["years"]) != EPOCHS:
            cached = None
        if cached is not None:
            dens, off = cached["density"], cached["heat_offset"]
            print(f"  {c['slug']} rasters reused")
        else:
            masks = [built_mask(y, geom, scale) for y in EPOCHS]
            h = min(m.shape[0] for m in masks)
            w = min(m.shape[1] for m in masks)
            dens = np.stack([m[:h, :w] for m in masks])
            off = (np.zeros((h, w), dtype=np.float32) if days
                   else heat_offset(geom, scale)[:h, :w])

        same_metric = (cached is not None and "metric" in cached.files
                       and str(cached["metric"]) == metric)
        if cached is not None and "pop" in cached.files:
            pop = [float(v) for v in cached["pop"]]
        else:
            pop = [population(y, geom) for y in EPOCHS]
        print(f"  {c['slug']} pop {EPOCHS[0]}-{EPOCHS[-1]}: "
              + " ".join(f"{v/1e6:.1f}M" for v in pop))
        if same_metric and len(cached["heat"]) == len(HEAT_YEARS):
            heat = [float(v) for v in cached["heat"]]
        else:
            # ERA5-Land starts 1950 and lags about ten days, so 2025 is a full year
            heat = heat_series(geom, metric, thresh)
        print(f"  {c['slug']} heat {HEAT_YEARS[0]}-{HEAT_YEARS[-1]} ({meta['unit'].strip()}): "
              + " ".join(f"{v:.1f}" for v in heat))
        np.savez_compressed(path, density=dens, heat_offset=off,
                            years=np.array(EPOCHS), pop=np.array(pop),
                            heat=np.array(heat), heat_years=np.array(HEAT_YEARS),
                            metric=np.array(metric))
        meta["cities"].append(dict(slug=c["slug"], name=c["name"],
                                   country=c.get("country", ""),
                                   extent_km=c["extent_km"], pop=pop, heat=heat))
    json.dump(meta, open(os.path.join(out_dir, "cities.json"), "w"), indent=1)
    print("wrote ->", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", dest="out_dir", default="data")
    p.add_argument("--metric", default="tmax_mean", choices=["tmax_mean", "tmax", "apparent"])
    p.add_argument("--thresh", type=float, default=35.0)
    p.add_argument("--resume", action="store_true",
                   help="reuse rasters and series already written to <out>")
    main(**vars(p.parse_args()))
