"""
render_shape.py — population squares laid out in the city's own map shape.

Each square is a fixed slice of population (default 50,000 people). Squares are
placed on the densest cells of the city's population raster, so the block of
squares takes the silhouette of the built-up area and grows outward as the city
grows. Fill colour is the heat metric on a continuous white-yellow-red ramp.

Because count and shape are both driven by the data, the picture stays honest:
the number of squares is the population, the arrangement is where people are.

Real data:
    python pull_ghsl_era5.py --out data
    python render_shape.py --data data --out out

Synthetic shapes, for building and reviewing the design without Earth Engine:
    python make_stub_masks.py --out data_stub
    python render_shape.py --data data_stub --out out

Outputs <out>/heat-cities.gif, heat-cities.mp4, heat-cities-final.png
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
GROUND = (21, 28, 38)
RULE = (46, 58, 74)
INK = (232, 237, 243)
QUIET = (132, 148, 168)
WARN = (222, 152, 92)

GRID = 16              # cells across each city's tile (fewer = larger squares)
SPREAD = 0.6           # 0 = squares pack into the core, 1 = spread to the edges
FIELD_TOP = 200        # top of the square field
FIELD_H = 520
YEAR_STEP = 1
FPS = 10               # gif and mp4 frame rate; --fps overrides
MAX_COLS = 5           # cities per row; more cities wrap onto a 1920x1080 frame

RAMP = [
    (0.00, (255, 255, 255)),
    (0.32, (255, 233, 168)),
    (0.55, (251, 192, 45)),
    (0.80, (232, 99, 42)),
    (1.00, (179, 22, 27)),
]

# DejaVu on Linux, Helvetica Neue on macOS (a .ttc, so faces are picked by index)
FD = "/usr/share/fonts/truetype/dejavu/"
HN = "/System/Library/Fonts/HelveticaNeue.ttc"
FACES = {
    "light": (FD + "DejaVuSans-ExtraLight.ttf", 5),
    "bold": (FD + "DejaVuSans-Bold.ttf", 1),
    "regular": (FD + "DejaVuSans.ttf", 0),
}


@lru_cache(maxsize=None)
def font(face, size):
    path, hn_index = FACES[face]
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.truetype(HN, size, index=hn_index)


f_year = font("light", 92)
f_title = font("bold", 26)
f_deck = font("regular", 17)
f_read = font("regular", 18)
f_small = font("regular", 14)
f_read_s = font("regular", 13)    # compact readout when cities wrap onto rows


def ramp_color(t):
    t = max(0.0, min(1.0, t))
    for i in range(1, len(RAMP)):
        if t <= RAMP[i][0]:
            p0, c0 = RAMP[i - 1]
            p1, c1 = RAMP[i]
            f = (t - p0) / (p1 - p0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return RAMP[-1][1]


def at_year(series, epochs, year):
    if year <= epochs[0]:
        return series[0]
    if year >= epochs[-1]:
        return series[-1]
    for i in range(1, len(epochs)):
        if year <= epochs[i]:
            f = (year - epochs[i - 1]) / (epochs[i] - epochs[i - 1])
            return series[i - 1] + (series[i] - series[i - 1]) * f


def coarsen(field, g):
    """Block-average a raster down to a g x g grid."""
    h, w = field.shape
    ys = np.linspace(0, h, g + 1).astype(int)
    xs = np.linspace(0, w, g + 1).astype(int)
    out = np.zeros((g, g))
    for r in range(g):
        for c in range(g):
            out[r, c] = field[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].mean()
    return out


_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def text_size(text, f):
    box = _MEASURE.textbbox((0, 0), text, font=f)
    return box[2], box[3]


def sub_font(size):
    """Country type: smaller than the city name, never below readable."""
    return font("regular", max(11, round(size * 0.62)))


def name_font_size(cities, max_w, sizes):
    """Largest size at which every "City Country" label still fits the column."""
    def fits(s):
        return all(text_size(c["name"] + " *", font("bold", s))[0] + 8
                   + text_size(c.get("country", ""), sub_font(s))[0] <= max_w
                   for c in cities)
    return next((s for s in sizes if fits(s)), sizes[-1])


def left_label(d, name, country, x, y, size):
    """City name with the country beside it, left-aligned at x, sharing a baseline."""
    f_name = font("bold", size)
    nw, nh = text_size(name, f_name)
    d.text((x, y), name, font=f_name, fill=INK)
    if country:
        f_sub = sub_font(size)
        d.text((x + nw + 8, y + nh - text_size(country, f_sub)[1]), country,
               font=f_sub, fill=QUIET)


def load(data_dir):
    meta = json.load(open(os.path.join(data_dir, "cities.json")))
    for c in meta["cities"]:
        z = np.load(os.path.join(data_dir, c["slug"] + ".npz"))
        raw = z["density"] if "density" in z else z["masks"]
        c["grids"] = np.stack([coarsen(f, GRID) for f in raw])
        # within-city heat offset in °C; zero-filled if the pull didn't supply one
        c["offset"] = (coarsen(z["heat_offset"], GRID) if "heat_offset" in z
                       else np.zeros((GRID, GRID)))
        # fixed dither weights: same every frame, so squares never flicker
        rng = np.random.default_rng(abs(hash(c["slug"])) % (2**32))
        c["dither"] = np.clip(rng.random((GRID, GRID)), 0.18, 1.0)
    return meta


def build(meta, lo, hi, unit_label, per_square, spread=SPREAD):
    epochs = meta["epochs"]
    heat_years = meta.get("heat_years", epochs)   # yearly heat; older data is per epoch
    cities = meta["cities"]
    n = len(cities)
    rows = math.ceil(n / MAX_COLS)
    cols = math.ceil(n / rows)
    compact = rows > 1                  # wrapped rows: bigger frame, one-line labels
    width, height = (1920, 1080) if compact else (W, H)
    # name and country sit above each tile, the readout under it
    head_h = 26 if compact else 52     # label hugs its own tile
    label_h = 34 if compact else 60    # readout, then a gutter before the next row
    row_h = (height - 90 - FIELD_TOP) / rows
    col_w = (width - 110) / cols
    step = min((col_w - 40) / GRID, (row_h - head_h - label_h) / GRID, FIELD_H / GRID)
    cell = step * 0.80
    name_size = name_font_size(cities, col_w - 30,
                               (22, 20, 18, 16, 15) if compact else (34, 30, 26, 22))
    synthetic = meta.get("synthetic", False)

    def cells_for(city, year):
        """Top-N densest cells, N set by population; returns (cells, overflow).

        The density field is interpolated between epochs before ranking, so the
        silhouette eases outward instead of snapping every five years.
        """
        g = city["grids"]
        if year <= epochs[0]:
            field = g[0]
        elif year >= epochs[-1]:
            field = g[-1]
        else:
            i = next(k for k in range(1, len(epochs)) if year <= epochs[k])
            f = (year - epochs[i - 1]) / (epochs[i] - epochs[i - 1])
            field = g[i - 1] * (1 - f) + g[i] * f
        # Dividing density by a fixed per-cell weight scatters squares out toward
        # the fringe instead of packing them all into the core, so the block
        # reaches the real edge of the built-up area. SPREAD sets how far.
        score = field / np.power(city["dither"], spread)
        score = np.where(field > 0.05 * max(field.max(), 1e-9), score, -1.0)

        pop = at_year(city["pop"], epochs, year)
        n_units = int(round(pop / per_square))
        order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
        valid = score > 0
        n_valid = int(valid.sum())
        if n_units <= n_valid:
            return order[:n_units], False
        # More squares than built-up cells (a city denser than one square per
        # cell): the extras take the empty cells nearest the footprint, ring by
        # ring, with denser cells and then the fixed dither breaking ties.
        vy, vx = np.nonzero(valid)
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        d2 = (((yy[..., None] - vy) ** 2 + (xx[..., None] - vx) ** 2).min(axis=-1)
              if n_valid else np.zeros(field.shape))
        rest = np.lexsort((city["dither"].ravel(), -field.ravel(), d2.ravel()))
        rest = rest[~valid.ravel()[rest]]
        extra = np.dstack(np.unravel_index(rest, field.shape))[0]
        return np.concatenate([order[:n_valid], extra])[:n_units], True

    def frame(year):
        img = Image.new("RGB", (width, height), GROUND)
        d = ImageDraw.Draw(img)

        d.text((70, 62), meta.get("title", "Growing into the heat"), font=f_title, fill=INK)
        d.text((70, 104),
               f"One square is {per_square/1000:.0f},000 people, placed where those people live. "
               f"Each square is coloured by {meta['heat_label']} at that spot.",
               font=f_deck, fill=QUIET)

        yt = str(int(round(year)))
        w = d.textbbox((0, 0), yt, font=f_year)[2]
        d.text((width - 70 - w, 60), yt, font=f_year, fill=INK)
        d.line([(70, 178), (width - 70, 178)], fill=RULE, width=1)

        any_overflow = False
        for ci, c in enumerate(cities):
            row, slot = divmod(ci, cols)
            cx = 55 + col_w * (slot + 0.5)
            left = cx - GRID * step / 2
            top = FIELD_TOP + row * row_h + head_h   # squares start below the label
            base = top + GRID * step                 # bottom edge of this city's squares
            pop = at_year(c["pop"], epochs, year)
            heat = at_year(c["heat"], heat_years, year)
            t_lo, t_hi = 99.0, -99.0

            cells, overflow = cells_for(c, year)
            any_overflow = any_overflow or overflow
            # every cell carries the city mean plus its own heat-island offset
            for (r, col) in cells:
                x = left + col * step
                y = top + r * step
                t = heat + c["offset"][r, col]
                t_lo, t_hi = min(t_lo, t), max(t_hi, t)
                d.rectangle([x, y, x + cell, y + cell],
                            fill=ramp_color((t - lo) / (hi - lo)))

            # a city smaller than one square draws nothing, so it has no range to show
            span = f"{t_lo:.1f}\u2013{t_hi:.1f}{unit_label}" if len(cells) else "\u2014"
            left_label(d, c["name"] + (" *" if overflow else ""), c.get("country", ""),
                       left, top - head_h + 2, name_size)
            if compact:
                d.text((left, base + 8), f"{pop/1e6:.1f}M   {span}", font=f_read_s, fill=QUIET)
            else:
                d.text((left, base + 26), f"{pop/1e6:.1f}M people   {span} across the city",
                       font=f_read, fill=QUIET)
                if overflow:
                    d.text((left, base + 54),
                           "squares extend past the built-up area (too dense for the grid)",
                           font=f_small, fill=QUIET)

        lx, ly, lw = 70, height - 48, 300
        for px in range(lw):
            d.line([(lx + px, ly), (lx + px, ly + 12)], fill=ramp_color(px / lw))
        d.text((lx, ly + 18), f"{lo:.1f}{unit_label}", font=f_small, fill=QUIET)
        d.text((lx + lw - 52, ly + 18), f"{hi:.1f}{unit_label}", font=f_small, fill=QUIET)
        d.text((lx + lw + 40, ly + 2), meta["source_note"], font=f_small,
               fill=WARN if synthetic else QUIET)
        if compact and any_overflow:
            d.text((lx + lw + 40, ly + 21),
                   "* squares extend past the built-up area (too dense for the grid)",
                   font=f_small, fill=QUIET)
        return img

    frames, y = [], epochs[0]
    while y <= epochs[-1]:
        frames.append(frame(y))
        y += YEAR_STEP
    return frames


def nice_unit(meta, per_square):
    """Pick a unit that fills roughly 70% of the grid at the largest city."""
    if per_square:
        return per_square
    biggest = max(max(c["pop"]) for c in meta["cities"])
    raw = biggest / (0.70 * GRID * GRID)
    return max(10_000, round(raw / 10_000) * 10_000)


def main(data_dir, out_dir, lo, hi, per_square, spread, fps):
    meta = load(data_dir)
    per_square = nice_unit(meta, per_square)
    print(f"one square = {per_square:,.0f} people on a {GRID}x{GRID} grid")
    unit_label = meta.get("unit", "")
    cell_vals = []
    for c in meta["cities"]:
        occupied = c["grids"][-1] > 0.02
        off = c["offset"][occupied] if occupied.any() else c["offset"].ravel()
        cell_vals += [min(c["heat"]) + float(off.min()), max(c["heat"]) + float(off.max())]
    span = max(cell_vals) - min(cell_vals)
    lo = lo if lo is not None else min(cell_vals) - 0.03 * span
    hi = hi if hi is not None else max(cell_vals) + 0.03 * span
    print(f"colour ramp anchored {lo:.2f} -> {hi:.2f} {unit_label}")
    if meta.get("synthetic"):
        print("NOTE: synthetic shapes - these are not real city footprints")

    os.makedirs(out_dir, exist_ok=True)
    frames = build(meta, lo, hi, unit_label, per_square, spread)

    hold = max(1, round(fps))          # hold the last frame for about a second
    frames[0].save(os.path.join(out_dir, "heat-cities.gif"), save_all=True,
                   append_images=frames[1:] + [frames[-1]] * hold,
                   duration=round(1000 / fps), loop=0, optimize=True)
    frames[-1].save(os.path.join(out_dir, "heat-cities-final.png"))

    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(os.path.join(td, f"{i:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", os.path.join(td, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", os.path.join(out_dir, "heat-cities.mp4")], check=True)

    print(f"wrote {len(frames)} frames -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_dir", default="data_stub")
    p.add_argument("--out", dest="out_dir", default="out")
    p.add_argument("--lo", type=float, default=None)
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--per-square", dest="per_square", type=float, default=None,
                   help="people per square; default fits the grid")
    p.add_argument("--spread", type=float, default=SPREAD,
                   help="0 packs squares into the core, 1 spreads them to the edges")
    p.add_argument("--fps", type=float, default=FPS,
                   help="frame rate of the gif and mp4 (51 frames, 1975-2025)")
    main(**vars(p.parse_args()))
