"""rank_cities.py — rank large cities by heat, same sources and window as the viz.

    EE_PROJECT=your-project python rank_cities.py city_heat_rank.csv

Heat (ERA5-Land daily, 2021-2025, mean over a 15 km radius around the centre):
  tmax_mean    mean daily maximum 2 m temperature, degC
  days35/40    days per year with daily max above 35 / 40 degC
  days_wbgt30  days per year with simplified WBGT above 30 degC
               (BoM: 0.567*Tmax + 0.393*e + 3.94, e from daily-mean dewpoint;
               approximates the WBGTmax > 30 threshold of Tuholske et al. 2021)
Population: GHSL GHS_POP 2025, summed over the viz's 75 km window.
"""
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import ee

ee.Initialize(project=os.environ.get("EE_PROJECT"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "city_heat_rank.csv"

# name, country, lon, lat, contrast (True = famously hot but under 5M, for comparison)
CITIES = [
    ("Delhi", "India", 77.209, 28.614, False), ("Mumbai", "India", 72.878, 19.076, False),
    ("Kolkata", "India", 88.364, 22.573, False), ("Chennai", "India", 80.271, 13.083, False),
    ("Bangalore", "India", 77.595, 12.972, False), ("Hyderabad", "India", 78.487, 17.385, False),
    ("Ahmedabad", "India", 72.571, 23.023, False), ("Pune", "India", 73.857, 18.520, False),
    ("Surat", "India", 72.831, 21.170, False), ("Dhaka", "Bangladesh", 90.413, 23.811, False),
    ("Chittagong", "Bangladesh", 91.783, 22.357, False), ("Karachi", "Pakistan", 67.010, 24.861, False),
    ("Lahore", "Pakistan", 74.359, 31.520, False), ("Bangkok", "Thailand", 100.501, 13.756, False),
    ("Jakarta", "Indonesia", 106.845, -6.208, False), ("Manila", "Philippines", 120.984, 14.600, False),
    ("Ho Chi Minh City", "Vietnam", 106.660, 10.762, False), ("Hanoi", "Vietnam", 105.834, 21.028, False),
    ("Yangon", "Myanmar", 96.195, 16.866, False), ("Kuala Lumpur", "Malaysia", 101.687, 3.139, False),
    ("Singapore", "Singapore", 103.820, 1.352, False), ("Guangzhou", "China", 113.264, 23.129, False),
    ("Shenzhen", "China", 114.058, 22.543, False), ("Dongguan", "China", 113.752, 23.021, False),
    ("Hong Kong", "China", 114.169, 22.319, False), ("Shanghai", "China", 121.474, 31.230, False),
    ("Beijing", "China", 116.407, 39.904, False), ("Tianjin", "China", 117.200, 39.084, False),
    ("Chongqing", "China", 106.551, 29.563, False), ("Chengdu", "China", 104.066, 30.573, False),
    ("Wuhan", "China", 114.305, 30.593, False), ("Hangzhou", "China", 120.155, 30.274, False),
    ("Nanjing", "China", 118.797, 32.060, False), ("Xi'an", "China", 108.940, 34.341, False),
    ("Changsha", "China", 112.939, 28.228, False), ("Zhengzhou", "China", 113.625, 34.747, False),
    ("Taipei", "Taiwan", 121.565, 25.033, False), ("Tokyo", "Japan", 139.692, 35.690, False),
    ("Osaka", "Japan", 135.502, 34.694, False), ("Seoul", "South Korea", 126.978, 37.567, False),
    ("Riyadh", "Saudi Arabia", 46.675, 24.713, False), ("Jeddah", "Saudi Arabia", 39.192, 21.485, False),
    ("Baghdad", "Iraq", 44.366, 33.315, False), ("Tehran", "Iran", 51.389, 35.689, False),
    ("Dubai", "UAE", 55.271, 25.205, False), ("Istanbul", "Turkey", 28.978, 41.008, False),
    ("Cairo", "Egypt", 31.236, 30.044, False), ("Alexandria", "Egypt", 29.919, 31.200, False),
    ("Khartoum", "Sudan", 32.560, 15.501, False), ("Lagos", "Nigeria", 3.379, 6.524, False),
    ("Kano", "Nigeria", 8.592, 12.002, False), ("Kinshasa", "DR Congo", 15.266, -4.442, False),
    ("Luanda", "Angola", 13.234, -8.839, False), ("Dar es Salaam", "Tanzania", 39.208, -6.792, False),
    ("Abidjan", "Ivory Coast", -4.008, 5.360, False), ("Nairobi", "Kenya", 36.822, -1.292, False),
    ("Addis Ababa", "Ethiopia", 38.747, 9.030, False), ("Johannesburg", "South Africa", 28.047, -26.204, False),
    ("Mexico City", "Mexico", -99.133, 19.433, False), ("Guadalajara", "Mexico", -103.350, 20.659, False),
    ("Monterrey", "Mexico", -100.316, 25.686, False), ("Sao Paulo", "Brazil", -46.633, -23.550, False),
    ("Rio de Janeiro", "Brazil", -43.173, -22.907, False), ("Belo Horizonte", "Brazil", -43.938, -19.920, False),
    ("Buenos Aires", "Argentina", -58.382, -34.604, False), ("Lima", "Peru", -77.043, -12.046, False),
    ("Bogota", "Colombia", -74.072, 4.711, False), ("Santiago", "Chile", -70.669, -33.449, False),
    ("New York", "USA", -74.006, 40.713, False), ("Los Angeles", "USA", -118.244, 34.052, False),
    ("Chicago", "USA", -87.630, 41.878, False), ("Houston", "USA", -95.370, 29.760, False),
    ("Dallas", "USA", -96.797, 32.777, False), ("Miami", "USA", -80.192, 25.762, False),
    ("Atlanta", "USA", -84.388, 33.749, False), ("Phoenix", "USA", -112.074, 33.448, False),
    ("London", "UK", -0.128, 51.507, False), ("Paris", "France", 2.352, 48.857, False),
    ("Madrid", "Spain", -3.704, 40.417, False), ("Barcelona", "Spain", 2.173, 41.385, False),
    ("Moscow", "Russia", 37.618, 55.756, False), ("Sydney", "Australia", 151.209, -33.869, False),
    ("Melbourne", "Australia", 144.963, -37.814, False),
    ("Kuwait City", "Kuwait", 47.978, 29.376, True), ("Ahvaz", "Iran", 48.669, 31.318, True),
    ("Basra", "Iraq", 47.783, 30.508, True), ("Doha", "Qatar", 51.531, 25.286, True),
    ("Jacobabad", "Pakistan", 68.438, 28.282, True), ("Niamey", "Niger", 2.117, 13.512, True),
]

Y0, Y1 = 2021, 2025
NYEARS = Y1 - Y0 + 1


def day(img):
    t = img.select("temperature_2m_max").subtract(273.15)
    td = img.select("dewpoint_temperature_2m").subtract(273.15)
    e = td.expression("6.105 * exp(17.27 * td / (237.7 + td))", {"td": td})
    wbgt = t.multiply(0.567).add(e.multiply(0.393)).add(3.94)
    return ee.Image.cat([t.rename("tmax"), t.gt(35).rename("d35"), t.gt(40).rename("d40"),
                         wbgt.gt(30).rename("w30")])


def measure(c):
    name, _, lon, lat, _ = c
    core = ee.Geometry.Point([lon, lat]).buffer(15000)
    daily = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
             .filterBounds(core).filterDate(f"{Y0}-01-01", f"{Y1 + 1}-01-01").map(day))
    heat_img = ee.Image.cat([
        daily.select("tmax").mean().rename("tmax_mean"),
        daily.select("d35").sum().divide(NYEARS).rename("days35"),
        daily.select("d40").sum().divide(NYEARS).rename("days40"),
        daily.select("w30").sum().divide(NYEARS).rename("days_wbgt30"),
    ])
    pop_img = ee.Image("JRC/GHSL/P2023A/GHS_POP/2025").select("population_count")
    pop_img = pop_img.updateMask(pop_img.gte(0))
    window = ee.Geometry.Point([lon, lat]).buffer(37500).bounds()
    both = ee.Dictionary({
        "heat": heat_img.reduceRegion(ee.Reducer.mean(), core, 9000, maxPixels=1e10),
        "pop": pop_img.reduceRegion(ee.Reducer.sum(), window, 100, maxPixels=1e10).get("population_count"),
    })
    for attempt in range(3):
        try:
            return both.getInfo()
        except Exception as e:
            err = e
    print(f"FAILED {name}: {err}", file=sys.stderr)
    return {"heat": {}, "pop": 0}


with ThreadPoolExecutor(12) as ex:
    results = list(ex.map(measure, CITIES))

rows = []
for c, res in zip(CITIES, results):
    h = res["heat"]
    rows.append(dict(name=c[0], country=c[1], lon=c[2], lat=c[3], contrast=c[4],
                     pop_m=round((res["pop"] or 0) / 1e6, 2), tmax_mean=h.get("tmax_mean"),
                     days35=h.get("days35"), days40=h.get("days40"), days_wbgt30=h.get("days_wbgt30")))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

missing = [r["name"] for r in rows if r["tmax_mean"] is None]
print("no ERA5-Land value:", missing or "none")
big = [r for r in rows if not r["contrast"] and r["tmax_mean"] is not None and r["pop_m"] >= 5]
print(f"{len(big)} candidates with >= 5M people in the 75 km window\n")


def table(title, key, n=25):
    print(title)
    print(f"  {'#':>2} {'city':18} {'pop M':>6} {'Tmax':>5} {'d>35':>5} {'d>40':>5} {'WBGT>30':>7}")
    for i, r in enumerate(sorted(big, key=lambda r: -r[key])[:n], 1):
        print(f"  {i:>2} {r['name']:18} {r['pop_m']:6.1f} {r['tmax_mean']:5.1f} {r['days35']:5.0f} "
              f"{r['days40']:5.0f} {r['days_wbgt30']:7.0f}")
    print()


table("BY MEAN DAILY MAX (the viz metric)", "tmax_mean")
table("BY DAYS ABOVE 35C", "days35")
table("BY HUMID HEAT (days simplified WBGT > 30C)", "days_wbgt30")
print("CONTRAST: famously hot, under 5M")
for r in rows:
    if r["contrast"] and r["tmax_mean"] is not None:
        print(f"     {r['name']:18} {r['pop_m']:6.1f} {r['tmax_mean']:5.1f} {r['days35']:5.0f} "
              f"{r['days40']:5.0f} {r['days_wbgt30']:7.0f}")
