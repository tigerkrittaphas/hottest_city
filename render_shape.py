"""
render_shape.py — population squares laid out in the city's own map shape.

Each square is a fixed slice of population (default 50,000 people). Squares are
placed on the densest cells of the city's population raster, so the block of
squares takes the silhouette of the built-up area and grows outward as the city
grows. Fill colour is the heat metric on a continuous white-yellow-red ramp,
stretched over each city's own range (--scale shared puts all on one range).

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
HEADER_GAP = 22        # clear space under the header rule, above the first tile
FIELD_H = 520          # tallest a single row of tiles may get
YEAR_STEP = 1
FPS = 2 # gif and mp4 frame rate; --fps overrides
MAX_COLS = 5           # cities per row; more cities wrap onto a 1920x1080 frame
FOOT_RESERVE = 68      # space kept below the tiles for the legend
COL_PAD = 24           # horizontal breathing room per column

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
        # a heat-index pull stores the offset in degC of temperature plus the
        # city's heat-index degrees per degree, so the scaling is applied here
        c["offset"] = c["offset"] * c.get("offset_gain", 1.0)
        # fixed dither weights: same every frame, so squares never flicker
        rng = np.random.default_rng(abs(hash(c["slug"])) % (2**32))
        c["dither"] = np.clip(rng.random((GRID, GRID)), 0.18, 1.0)
    return meta


def color_ranges(meta, lo=None, hi=None, scale="city"):
    """(lo, hi) of the colour ramp for each city.

    "city" stretches the ramp over each city's own coolest-to-hottest cell across
    all years, so the pattern inside every city shows; colours then can't be
    compared between tiles. "shared" puts every city on one range, which --lo
    and --hi can pin.
    """
    spans = []
    for c in meta["cities"]:
        occupied = c["grids"][-1] > 0.02
        off = c["offset"][occupied] if occupied.any() else c["offset"].ravel()
        spans.append((min(c["heat"]) + float(off.min()), max(c["heat"]) + float(off.max())))
    if scale == "shared":
        a, b = min(s[0] for s in spans), max(s[1] for s in spans)
        pad = 0.03 * (b - a)
        a = lo if lo is not None else round(a - pad, 1)
        b = hi if hi is not None else round(b + pad, 1)
        return [(a, b)] * len(spans)
    out = []
    for a, b in spans:
        pad = 0.03 * max(b - a, 0.5)
        out.append((round(a - pad, 1), round(b + pad, 1)))
    return out


def build(meta, ranges, unit_label, per_square, spread=SPREAD):
    epochs = meta["epochs"]
    heat_years = meta.get("heat_years", epochs)   # yearly heat; older data is per epoch
    cities = meta["cities"]
    n = len(cities)
    rows = math.ceil(n / MAX_COLS)
    cols = math.ceil(n / rows)
    compact = rows > 1                  # wrapped rows: bigger frame, one-line labels
    width, height = (1920, 1080) if compact else (W, H)
    # the rule sits below whatever the header actually draws, so the big year
    # number can never collide with it and no space is wasted guessing
    rule_y = round(max(60 + text_size("2025", f_year)[1],
                       104 + text_size("Ay", f_deck)[1])) + 18
    field_top = rule_y + HEADER_GAP
    # name and country sit above each tile, the readout under it
    head_h = 32 if compact else 52     # name band: the text, then a gap above the tile
    label_h = 38 if compact else 60    # readout under the tile, then the row gutter
    row_h = (height - FOOT_RESERVE - field_top) / rows
    col_w = (width - 110) / cols
    step = min((col_w - COL_PAD) / GRID, (row_h - head_h - label_h) / GRID, FIELD_H / GRID)
    cell = step * 0.80
    name_size = name_font_size(cities, col_w - 30,
                               (22, 20, 18, 16, 15) if compact else (34, 30, 26, 22))
    synthetic = meta.get("synthetic", False)
    per_city = len(set(ranges)) > 1

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
               f"Each square is coloured by {meta['heat_label']} at that spot"
               + (", on each city's own scale." if per_city else "."),
               font=f_deck, fill=QUIET)

        yt = str(int(round(year)))
        w = d.textbbox((0, 0), yt, font=f_year)[2]
        d.text((width - 70 - w, 60), yt, font=f_year, fill=INK)
        d.line([(70, rule_y), (width - 70, rule_y)], fill=RULE, width=1)

        any_overflow = False
        for ci, c in enumerate(cities):
            row, slot = divmod(ci, cols)
            cx = 55 + col_w * (slot + 0.5)
            left = cx - GRID * step / 2
            top = field_top + row * row_h + head_h   # squares start below the label
            base = top + GRID * step                 # bottom edge of this city's squares
            pop = at_year(c["pop"], epochs, year)
            heat = at_year(c["heat"], heat_years, year)
            lo, hi = ranges[ci]
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
                d.text((left, base + 6), f"{pop/1e6:.1f}M   {span}", font=f_read_s, fill=QUIET)
            else:
                d.text((left, base + 26), f"{pop/1e6:.1f}M people   {span} across the city",
                       font=f_read, fill=QUIET)
            if per_city:
                # this tile's own ramp, at the right edge of the readout line
                f_key = f_read_s if compact else f_small
                key_y = base + (6 if compact else 58)
                lo_t, hi_t = f"{lo:.1f}", f"{hi:.1f}\u00b0"
                bar_w = 56 if compact else 90
                right = left + GRID * step
                if compact:
                    right = max(right, left + text_size(f"{pop/1e6:.1f}M   {span}", f_key)[0]
                                + 18 + text_size(lo_t, f_key)[0] + 6 + bar_w + 6
                                + text_size(hi_t, f_key)[0])
                hx = right - text_size(hi_t, f_key)[0]
                bx = hx - 6 - bar_w
                lx0 = bx - 6 - text_size(lo_t, f_key)[0]
                th = text_size("0", f_key)[1]
                by = key_y + th // 2 - 3
                for px in range(bar_w):
                    d.line([(bx + px, by), (bx + px, by + 7)], fill=ramp_color(px / bar_w))
                d.text((lx0, key_y), lo_t, font=f_key, fill=QUIET)
                d.text((hx, key_y), hi_t, font=f_key, fill=QUIET)
            if not compact:
                if overflow:
                    d.text((left, base + 80 if per_city else base + 54),
                           "squares extend past the built-up area (too dense for the grid)",
                           font=f_small, fill=QUIET)

        lx, ly, lw = 70, height - 48, 300
        for px in range(lw):
            d.line([(lx + px, ly), (lx + px, ly + 12)], fill=ramp_color(px / lw))
        if per_city:
            d.text((lx, ly + 18), "coolest", font=f_small, fill=QUIET)
            d.text((lx + lw - text_size("hottest", f_small)[0], ly + 18), "hottest",
                   font=f_small, fill=QUIET)
            d.text((lx + 76, ly + 18), "each city on its own scale",
                   font=f_small, fill=QUIET)
        else:
            lo, hi = ranges[0]
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


def main(data_dir, out_dir, lo, hi, per_square, spread, fps, scale):
    meta = load(data_dir)
    per_square = nice_unit(meta, per_square)
    print(f"one square = {per_square:,.0f} people on a {GRID}x{GRID} grid")
    unit_label = meta.get("unit", "")
    ranges = color_ranges(meta, lo, hi, scale)
    for c, (a, b) in zip(meta["cities"], ranges):
        print(f"  {c['name']:<18} colour ramp {a:.1f} -> {b:.1f} {unit_label}")
    if meta.get("synthetic"):
        print("NOTE: synthetic shapes - these are not real city footprints")

    os.makedirs(out_dir, exist_ok=True)
    frames = build(meta, ranges, unit_label, per_square, spread)

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
    p.add_argument("--scale", choices=["city", "shared"], default="city",
                   help="city: each city gets its own colour range; shared: one range for all")
    p.add_argument("--lo", type=float, default=None, help="shared scale only")
    p.add_argument("--hi", type=float, default=None, help="shared scale only")
    p.add_argument("--per-square", dest="per_square", type=float, default=None,
                   help="people per square; default fits the grid")
    p.add_argument("--spread", type=float, default=SPREAD,
                   help="0 packs squares into the core, 1 spreads them to the edges")
    p.add_argument("--fps", type=float, default=FPS,
                   help="frame rate of the gif and mp4 (51 frames, 1975-2025)")
    main(**vars(p.parse_args()))
