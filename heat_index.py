"""
heat_index.py — the US National Weather Service heat index, in degC.

Two versions of the same formula: `hi_numpy` for checking values locally and
`hi_ee` for Earth Engine images. Both take air temperature and dewpoint in degC.

The NWS algorithm (Rothfusz regression with Steadman's simple form below about
80 degF, plus the two published humidity adjustments):
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    python heat_index.py        # checks hi_numpy against the NWS heat index table
"""

import numpy as np

# Rothfusz regression, degF and % relative humidity
C = (-42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783,
     -0.05481717, 0.00122874, 0.00085282, -0.00000199)


def rh_from_dewpoint(t, td):
    """Relative humidity (%) from temperature and dewpoint in degC (Magnus form)."""
    return 100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))


def hi_numpy(t, td):
    t, td = np.asarray(t, float), np.asarray(td, float)
    f = t * 1.8 + 32
    rh = np.clip(rh_from_dewpoint(t, td), 0, 100)
    simple = 0.5 * (f + 61 + (f - 68) * 1.2 + rh * 0.094)
    roth = (C[0] + C[1] * f + C[2] * rh + C[3] * f * rh + C[4] * f * f
            + C[5] * rh * rh + C[6] * f * f * rh + C[7] * f * rh * rh
            + C[8] * f * f * rh * rh)
    dry = (rh < 13) & (f >= 80) & (f <= 112)
    roth = np.where(dry, roth - (13 - rh) / 4
                    * np.sqrt(np.clip(17 - np.abs(f - 95), 0, None) / 17), roth)
    wet = (rh > 85) & (f >= 80) & (f <= 87)
    roth = np.where(wet, roth + (rh - 85) / 10 * (87 - f) / 5, roth)
    hi_f = np.where((simple + f) / 2 >= 80, roth, simple)
    return (hi_f - 32) / 1.8


def hi_ee(t, td):
    """Same as hi_numpy, for single-band ee.Images in degC."""
    import ee
    f = t.multiply(1.8).add(32).rename("f")
    rh = td.expression(
        "100 * exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))",
        {"t": t, "td": td}).clamp(0, 100).rename("rh")
    v = {"f": f, "rh": rh}
    simple = f.expression("0.5 * (f + 61 + (f - 68) * 1.2 + rh * 0.094)", v)
    roth = f.expression(
        "c0 + c1*f + c2*rh + c3*f*rh + c4*f*f + c5*rh*rh + c6*f*f*rh"
        " + c7*f*rh*rh + c8*f*f*rh*rh",
        dict(v, **{f"c{i}": c for i, c in enumerate(C)}))
    dry = rh.lt(13).And(f.gte(80)).And(f.lte(112))
    # inside the dry mask 80 <= f <= 112, so 17 - |f - 95| is never negative there
    dry_adj = f.expression("(13 - rh) / 4 * sqrt(abs(17 - abs(f - 95)) / 17)", v)
    roth = roth.where(dry, roth.subtract(dry_adj))
    wet = rh.gt(85).And(f.gte(80)).And(f.lte(87))
    wet_adj = f.expression("(rh - 85) / 10 * (87 - f) / 5", v)
    roth = roth.where(wet, roth.add(wet_adj))
    hi_f = simple.where(simple.add(f).divide(2).gte(80), roth)
    return ee.Image(hi_f.subtract(32).divide(1.8)).rename("heat_index")


def _check():
    """Spot values from the NWS heat index chart (degF in, degF out, +-1)."""
    def dewpoint(t_c, rh):
        a = np.log(rh / 100) + 17.625 * t_c / (243.04 + t_c)
        return 243.04 * a / (17.625 - a)
    table = [(80, 40, 80), (90, 40, 91), (90, 70, 105), (96, 50, 108),
             (100, 40, 109), (104, 55, 137), (86, 90, 105)]
    for tf, rh, want in table:
        t_c = (tf - 32) / 1.8
        got = float(hi_numpy(t_c, dewpoint(t_c, rh))) * 1.8 + 32
        ok = abs(got - want) <= 1.5
        print(f"  {tf} degF {rh:>3}% RH -> {got:6.1f} degF (chart {want})  {'ok' if ok else 'OFF'}")
        assert ok
    print("heat index matches the NWS chart")


if __name__ == "__main__":
    _check()
