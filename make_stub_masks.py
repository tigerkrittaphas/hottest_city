"""
make_stub_masks.py — SYNTHETIC density rasters for render_shape.py.

Writes the same files pull_ghsl_era5.py writes, so the renderer can be built,
reviewed and demoed before anyone authenticates to Earth Engine:

    <out>/<slug>.npz     density (T,H,W) in [0,1], years (T,)
    <out>/cities.json    name, pop[T], heat[T], labels

The shapes are procedurally generated. They are NOT Bangkok, Karachi or Lagos.
They grow like cities grow — a core, later satellite cores, a water edge that
blocks expansion on one side, irregular fringes — which is enough to build and
judge the design against. Delete this output the moment the real pull runs;
every frame it produces is captioned as synthetic for exactly that reason.
"""

import argparse
import json
import os

import numpy as np

EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
RES = 96                      # raster side; render quantises this down

CITIES = [
    dict(slug="bangkok", name="Bangkok", seed=11,
         pop=[3.8e6, 4.5e6, 5.3e6, 5.9e6, 6.1e6, 6.3e6, 6.9e6, 8.3e6, 9.2e6, 10.5e6, 11.2e6],
         heat=[27.9, 28.0, 28.2, 28.3, 28.5, 28.6, 28.8, 29.0, 29.1, 29.3, 29.4],
         water="river"),
    dict(slug="karachi", name="Karachi", seed=23,
         pop=[4.0e6, 5.0e6, 6.2e6, 7.5e6, 9.0e6, 10.0e6, 12.2e6, 13.1e6, 15.4e6, 16.1e6, 18.0e6],
         heat=[25.9, 26.0, 26.1, 26.3, 26.5, 26.6, 26.8, 27.0, 27.1, 27.3, 27.4],
         water="coast"),
    dict(slug="lagos", name="Lagos", seed=37,
         pop=[1.9e6, 2.6e6, 3.5e6, 4.8e6, 6.0e6, 7.3e6, 8.8e6, 10.4e6, 12.2e6, 14.4e6, 16.5e6],
         heat=[26.6, 26.7, 26.8, 26.9, 27.0, 27.2, 27.3, 27.4, 27.6, 27.8, 27.8],
         water="coast"),
]


def smooth_noise(rng, n, octaves=(3, 6, 12)):
    """Cheap fractal noise: upsample a few coarse random fields and add them."""
    out = np.zeros((n, n))
    amp = 1.0
    for o in octaves:
        coarse = rng.random((o, o))
        yi = np.linspace(0, o - 1, n)
        xi = np.linspace(0, o - 1, n)
        y0 = np.clip(np.floor(yi).astype(int), 0, o - 1)
        x0 = np.clip(np.floor(xi).astype(int), 0, o - 1)
        y1 = np.clip(y0 + 1, 0, o - 1)
        x1 = np.clip(x0 + 1, 0, o - 1)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        top = coarse[np.ix_(y0, x0)] * (1 - fx) + coarse[np.ix_(y0, x1)] * fx
        bot = coarse[np.ix_(y1, x0)] * (1 - fx) + coarse[np.ix_(y1, x1)] * fx
        out += amp * (top * (1 - fy) + bot * fy)
        amp *= 0.55
    out -= out.min()
    return out / out.max()


def water_mask(kind, n, rng):
    """Zero where the city cannot expand, so growth is lopsided like a real one."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    if kind == "coast":
        edge = 0.84 + 0.05 * np.sin(xx * 7 + rng.random() * 6)
        return (yy < edge).astype(float)
    # river: a meandering band removed from the middle
    band = np.abs(xx - (0.46 + 0.10 * np.sin(yy * 5.2 + 1.1)))
    return (band > 0.035).astype(float)


def build_city(city):
    rng = np.random.default_rng(city["seed"])
    n = RES
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)

    texture = smooth_noise(rng, n)
    water = water_mask(city["water"], n, rng)

    # a primary core plus satellites that switch on partway through
    cores = [dict(y=0.52, x=0.46, born=0.0, weight=1.0)]
    for _ in range(rng.integers(2, 4)):
        cores.append(dict(y=float(rng.uniform(0.22, 0.78)),
                          x=float(rng.uniform(0.18, 0.80)),
                          born=float(rng.uniform(0.25, 0.65)),
                          weight=float(rng.uniform(0.35, 0.7))))

    frames = []
    for i, _year in enumerate(EPOCHS):
        t = i / (len(EPOCHS) - 1)
        field = np.zeros((n, n))
        for c in cores:
            if t < c["born"]:
                continue
            age = (t - c["born"]) / max(1e-6, 1 - c["born"])
            sigma = 0.07 + 0.20 * age ** 0.7          # sprawl widens with age
            d2 = (yy - c["y"]) ** 2 + (xx - c["x"]) ** 2
            field += c["weight"] * np.exp(-d2 / (2 * sigma ** 2))
        field *= 0.55 + 0.75 * texture                # ragged fringes
        field *= water
        if field.max() > 0:
            field /= field.max()
        frames.append(field)

    dens = np.stack(frames).astype(np.float32)

    # Within-city heat offset (°C), static across years: built-up cores run
    # hotter, water and fringe run cooler. Stands in for an LST-derived urban
    # heat island pattern.
    built = dens[-1]
    offset = 1.9 * (built - built.mean()) / max(built.std(), 1e-6) * 0.5
    offset += 0.45 * (smooth_noise(rng, n, octaves=(4, 9)) - 0.5)
    offset -= 1.1 * (1 - water)                     # water cools its surroundings
    offset -= offset.mean()
    return dens, offset.astype(np.float32)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(
        title="Growing into the heat",
        heat_label="annual mean temperature",
        heat_note=("City mean varies by year; the within-city pattern is a static "
                   "heat-island offset."),
        unit="\u00b0C",
        source_note="SYNTHETIC shapes and values \u2014 not real footprints. Placeholder only.",
        epochs=EPOCHS,
        synthetic=True,
        cities=[],
    )
    for c in CITIES:
        dens, offset = build_city(c)
        np.savez_compressed(os.path.join(out_dir, c["slug"] + ".npz"),
                            density=dens, heat_offset=offset, years=np.array(EPOCHS))
        meta["cities"].append(dict(slug=c["slug"], name=c["name"],
                                   pop=c["pop"], heat=c["heat"]))
        print(f"  {c['slug']}: {dens.shape}")
    json.dump(meta, open(os.path.join(out_dir, "cities.json"), "w"), indent=1)
    print("wrote ->", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", dest="out_dir", default="data_stub")
    main(**vars(p.parse_args()))
