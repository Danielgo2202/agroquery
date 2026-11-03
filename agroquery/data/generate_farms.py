"""Generate synthetic farms, fields, sensors, readings, alerts and harvests.

Everything is driven by the real hourly weather downloaded by fetch_weather.py:
  * soil moisture follows an hourly water-balance (bucket) model fed by real
    rainfall and reference evapotranspiration, plus irrigation and sensor noise;
  * soil temperature is a smoothed, damped version of real air temperature;
  * harvest yields depend on the water stress the field actually went through.
So the data has real structure for the agent to find, but the exact answers
cannot have been memorized from any LLM's training data.

Deterministic given config.SEED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agroquery.config import END_DATE, RAW_DIR, REGIONS, SEED, START_DATE
from agroquery.data.fetch_weather import raw_path

READING_EVERY_H = 2  # soil moisture / soil temperature sampling interval


@dataclass(frozen=True)
class CropSpec:
    name_es: str
    cycle: str  # annual | perennial
    moisture_min: float  # optimal volumetric soil moisture range (%)
    moisture_max: float
    kc: float  # crop coefficient for evapotranspiration
    area_ha: tuple[float, float]
    p_irrigated: float
    yield_per_ha: float  # kg/ha per harvest event under ideal conditions
    t_opt: float  # optimal mean air temperature (C)


CROPS: dict[str, CropSpec] = {
    "coffee": CropSpec("Café", "perennial", 30, 45, 0.95, (1, 8), 0.10, 600, 19),
    "avocado": CropSpec("Aguacate", "perennial", 25, 40, 0.90, (2, 10), 0.40, 5000, 18),
    "plantain": CropSpec("Plátano", "perennial", 30, 45, 1.05, (1, 6), 0.10, 2500, 22),
    "oil_palm": CropSpec("Palma de aceite", "perennial", 25, 40, 1.00, (20, 80), 0.10, 1500, 27),
    "rice": CropSpec("Arroz", "annual", 35, 55, 1.15, (10, 60), 0.85, 6500, 27),
    "cacao": CropSpec("Cacao", "perennial", 28, 42, 1.00, (2, 12), 0.15, 300, 26),
    "maize": CropSpec("Maíz", "annual", 22, 38, 1.10, (5, 30), 0.30, 5500, 25),
    "sugarcane": CropSpec("Caña de azúcar", "annual", 25, 40, 1.20, (20, 100), 0.70, 110000, 26),
}

# Relative frequency of each crop within the regions that grow it.
CROP_WEIGHTS = {
    "coffee": 5, "avocado": 2, "plantain": 2, "oil_palm": 3, "rice": 3,
    "cacao": 2, "maize": 2, "sugarcane": 4,
}

# Soil pH range by region (the Llanos have notoriously acidic soils).
REGION_PH = {
    "Eje Cafetero": (5.0, 5.8),
    "Llanos del Meta": (4.7, 5.5),
    "Centro del Tolima": (5.8, 6.8),
    "Norte del Huila": (6.0, 7.0),
    "Valle del Cauca": (6.6, 7.7),
}
FARMS_PER_REGION = (10, 9, 7, 7, 7)

FARM_NAMES = [
    "La Esperanza", "El Paraíso", "Santa Helena", "Buenavista", "La Primavera",
    "El Recreo", "San José", "La Fortuna", "Villa María", "El Porvenir",
    "La Pradera", "Los Naranjos", "El Cedral", "La Cabaña", "Monteverde",
    "Las Palmas", "El Diamante", "La Aurora", "San Isidro", "El Edén",
    "La Victoria", "Bellavista", "El Vergel", "La Floresta", "Santa Rosa",
    "El Retiro", "Los Guaduales", "La Ceiba", "El Mirador", "Altamira",
    "La Palmita", "El Jardín", "Las Brisas", "San Antonio", "La Gaitana",
    "El Samán", "La Herradura", "Villa Luz", "El Encanto", "Guayacanes",
]

# Alert thresholds (warning, critical) and hysteresis margins.
SOIL_TEMP_WARN, SOIL_TEMP_CRIT = 32.0, 34.0
PH_LOW_WARN, PH_LOW_CRIT = 5.0, 4.5
PH_HIGH_WARN, PH_HIGH_CRIT = 7.5, 8.0
MOISTURE_CRIT_MARGIN = 5.0
ALERT_COOLDOWN = pd.Timedelta(hours=24)  # same alert is not re-raised within 24 h


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def load_weather() -> tuple[dict[str, pd.DataFrame], dict]:
    hourly = {}
    for region in REGIONS:
        df = pd.read_parquet(raw_path(region.name))
        df = df[(df.time >= START_DATE) & (df.time < pd.Timestamp(END_DATE) + pd.Timedelta(days=1))]
        hourly[region.name] = df.reset_index(drop=True)
    meta = json.loads((RAW_DIR / "regions_meta.json").read_text(encoding="utf-8"))
    return hourly, meta


def daily_weather(hourly: dict[str, pd.DataFrame], region_ids: dict[str, int]) -> pd.DataFrame:
    frames = []
    for name, df in hourly.items():
        d = df.assign(date=df.time.dt.date).groupby("date").agg(
            temp_mean_c=("temperature_2m", "mean"),
            temp_min_c=("temperature_2m", "min"),
            temp_max_c=("temperature_2m", "max"),
            precipitation_mm=("precipitation", "sum"),
            rel_humidity_mean_pct=("relative_humidity_2m", "mean"),
            radiation_mj_m2=("shortwave_radiation", lambda s: s.sum() * 3600 / 1e6),
            et0_mm=("et0_fao_evapotranspiration", "sum"),
        )
        d = d.round(2).reset_index()
        d.insert(0, "region_id", region_ids[name])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def make_entities(rng: np.random.Generator, meta: dict):
    regions = pd.DataFrame(
        {
            "region_id": range(1, len(REGIONS) + 1),
            "name": [r.name for r in REGIONS],
            "department": [r.department for r in REGIONS],
            "latitude": [r.latitude for r in REGIONS],
            "longitude": [r.longitude for r in REGIONS],
            "elevation_m": [int(round(meta[r.name]["elevation_m"])) for r in REGIONS],
        }
    )
    crops = pd.DataFrame(
        [
            {
                "crop": c,
                "common_name_es": s.name_es,
                "cycle": s.cycle,
                "moisture_min_pct": s.moisture_min,
                "moisture_max_pct": s.moisture_max,
            }
            for c, s in CROPS.items()
        ]
    )

    names = list(FARM_NAMES)
    rng.shuffle(names)
    start = pd.Timestamp(START_DATE)
    farms, fields = [], []
    farm_id = field_id = 0
    pairs = zip(REGIONS, FARMS_PER_REGION, strict=True)
    for region_id, (region, n_farms) in enumerate(pairs, start=1):
        weights = np.array([CROP_WEIGHTS[c] for c in region.crops], dtype=float)
        weights /= weights.sum()
        for _ in range(n_farms):
            farm_id += 1
            # ~70% of farms were on the platform before the study period.
            if rng.random() < 0.7:
                joined = start - pd.Timedelta(days=int(rng.integers(30, 400)))
            else:
                joined = start + pd.Timedelta(days=int(rng.integers(60, 560)))
            farms.append(
                {
                    "farm_id": farm_id,
                    "name": names[farm_id - 1],
                    "region_id": region_id,
                    "latitude": round(region.latitude + rng.uniform(-0.15, 0.15), 4),
                    "longitude": round(region.longitude + rng.uniform(-0.15, 0.15), 4),
                    "joined_date": joined.date(),
                }
            )
            for lot in range(1, int(rng.integers(2, 5)) + 1):
                field_id += 1
                crop = str(rng.choice(region.crops, p=weights))
                spec = CROPS[crop]
                if spec.cycle == "perennial":
                    planting = pd.Timestamp(
                        year=int(rng.integers(2010, 2022)),
                        month=int(rng.integers(1, 13)),
                        day=int(rng.integers(1, 29)),
                    )
                else:
                    first = max(start, pd.Timestamp(joined))
                    planting = first + pd.Timedelta(days=int(rng.integers(0, 60)))
                fields.append(
                    {
                        "field_id": field_id,
                        "farm_id": farm_id,
                        "name": f"Lote {lot}",
                        "crop": crop,
                        "area_ha": round(float(rng.uniform(*spec.area_ha)), 1),
                        "planting_date": planting.date(),
                        "irrigated": bool(rng.random() < spec.p_irrigated),
                        "region_id": region_id,  # helper column, dropped before loading
                    }
                )
    return regions, crops, pd.DataFrame(farms), pd.DataFrame(fields)


def make_sensors(rng: np.random.Generator, farms: pd.DataFrame, fields: pd.DataFrame):
    joined = farms.set_index("farm_id")["joined_date"]
    rows = []
    sensor_id = 0
    for f in fields.itertuples():
        base = pd.Timestamp(joined[f.farm_id])
        types = ["soil_moisture", "soil_temp"] + (["ph"] if rng.random() < 0.5 else [])
        for t in types:
            sensor_id += 1
            installed = base + pd.Timedelta(days=int(rng.integers(0, 45)))
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "field_id": f.field_id,
                    "sensor_type": t,
                    "installed_at": installed.date(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Physical simulation
# ---------------------------------------------------------------------------

def simulate_moisture(rng, weather: pd.DataFrame, region_fields: pd.DataFrame) -> np.ndarray:
    """Hourly volumetric soil moisture (%) for every field of one region.

    Bucket model over a 500 mm root zone (1 % of volume = 5 mm of water):
    rain infiltrates, the crop transpires Kc * ET0 scaled by a stress factor,
    water above field capacity drains, irrigated fields are watered by day
    when they approach the lower optimal bound. Returns (hours, fields).
    """
    rain = weather["precipitation"].to_numpy()
    et0 = weather["et0_fao_evapotranspiration"].to_numpy()
    hour = weather["time"].dt.hour.to_numpy()

    specs = [CROPS[c] for c in region_fields["crop"]]
    m_min = np.array([s.moisture_min for s in specs])
    m_max = np.array([s.moisture_max for s in specs])
    kc = np.array([s.kc for s in specs])
    fc = m_max - rng.uniform(1, 4, len(specs))  # field capacity
    wp = m_min - rng.uniform(10, 14, len(specs))  # wilting point
    sat = fc + 10
    infiltration = rng.uniform(0.7, 0.9, len(specs))
    irrigated = region_fields["irrigated"].to_numpy()

    theta = (m_min + m_max) / 2
    out = np.empty((len(rain), len(specs)))
    for t in range(len(rain)):
        stress = np.clip((theta - wp) / ((m_min + m_max) / 2 - wp), 0, 1)
        delta = (rain[t] * infiltration - kc * et0[t] * stress) / 5.0
        if 6 <= hour[t] <= 17:
            delta += np.where(irrigated & (theta < m_min + 2), 1.0, 0.0)
        theta = theta + delta
        theta = theta - 0.3 * np.clip(theta - fc, 0, None)  # drainage
        theta = np.clip(theta, wp, sat)
        out[t] = theta
    return out


def simulate_soil_temp(rng, weather: pd.DataFrame, n: int) -> np.ndarray:
    air = weather["temperature_2m"].to_numpy()
    offset = rng.uniform(-1.0, 2.0, n)  # canopy shade vs. bare soil
    alpha = 0.04  # ~10 cm depth: damped, lagged diurnal cycle
    soil = np.empty((len(air), n))
    s = np.full(n, air[:48].mean())
    for t in range(len(air)):
        s = s + alpha * (air[t] - s)
        soil[t] = s
    return soil + offset


def outage_mask(rng, ts: pd.DatetimeIndex) -> np.ndarray:
    """True where the sensor was online; each sensor gets 0-4 outages of 1-10 days."""
    keep = np.ones(len(ts), dtype=bool)
    for _ in range(int(rng.integers(0, 5))):
        start = ts[int(rng.integers(0, len(ts)))]
        keep &= ~((ts >= start) & (ts < start + pd.Timedelta(days=int(rng.integers(1, 11)))))
    return keep


def episodes(ts: np.ndarray, values: np.ndarray, beyond, back_inside) -> np.ndarray:
    """Indices where an out-of-range episode starts (hysteresis + 24 h cooldown)."""
    starts, inside, last = [], True, None
    for i, v in enumerate(values):
        if inside and beyond(v):
            if last is None or ts[i] - last >= ALERT_COOLDOWN:
                starts.append(i)
                last = ts[i]
            inside = False
        elif not inside and back_inside(v):
            inside = True
    return np.array(starts, dtype=int)


def alerts_for(sensor_type: str, crop: str, ts: np.ndarray, vals: np.ndarray) -> list[dict]:
    rules = []  # (alert_type, level, beyond, back_inside)
    if sensor_type == "soil_moisture":
        lo, hi = CROPS[crop].moisture_min, CROPS[crop].moisture_max
        cm = MOISTURE_CRIT_MARGIN
        rules += [
            ("low_moisture", "warning", lambda v: v < lo, lambda v: v >= lo + 1),
            ("low_moisture", "critical", lambda v: v < lo - cm, lambda v: v >= lo - cm + 1),
            ("high_moisture", "warning", lambda v: v > hi, lambda v: v <= hi - 1),
            ("high_moisture", "critical", lambda v: v > hi + cm, lambda v: v <= hi + cm - 1),
        ]
    elif sensor_type == "soil_temp":
        rules += [
            ("high_soil_temp", "warning", lambda v: v > SOIL_TEMP_WARN,
             lambda v: v <= SOIL_TEMP_WARN - 1.5),
            ("high_soil_temp", "critical", lambda v: v > SOIL_TEMP_CRIT,
             lambda v: v <= SOIL_TEMP_CRIT - 1.5),
        ]
    else:
        rules += [
            ("low_ph", "warning", lambda v: v < PH_LOW_WARN, lambda v: v >= PH_LOW_WARN + 0.1),
            ("low_ph", "critical", lambda v: v < PH_LOW_CRIT, lambda v: v >= PH_LOW_CRIT + 0.1),
            ("high_ph", "warning", lambda v: v > PH_HIGH_WARN, lambda v: v <= PH_HIGH_WARN - 0.1),
            ("high_ph", "critical", lambda v: v > PH_HIGH_CRIT,
             lambda v: v <= PH_HIGH_CRIT - 0.1),
        ]
    out = []
    for alert_type, level, beyond, back in rules:
        for i in episodes(ts, vals, beyond, back):
            out.append({"ts": ts[i], "alert_type": alert_type, "level": level, "value": vals[i]})
    return out


# ---------------------------------------------------------------------------
# Harvests
# ---------------------------------------------------------------------------

def harvest_dates(rng, crop: str, planting: pd.Timestamp, start: pd.Timestamp,
                  end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(harvest_date, window_start) pairs; the window is what drives the yield."""
    out = []
    if crop in ("coffee", "cacao", "avocado"):
        # Two harvests a year: main crop and secondary ("mitaca" for coffee).
        months = {"coffee": (5, 11), "cacao": (5, 11), "avocado": (3, 9)}[crop]
        for year in range(start.year, end.year + 1):
            for m in months:
                d = pd.Timestamp(year=year, month=m, day=1) + pd.Timedelta(
                    days=int(rng.integers(-15, 25)))
                out.append((d, d - pd.Timedelta(days=60)))
    elif crop in ("plantain", "oil_palm"):
        every = {"plantain": 60, "oil_palm": 30}[crop]
        d = start + pd.Timedelta(days=int(rng.integers(0, every)))
        while d <= end:
            out.append((d, d - pd.Timedelta(days=60)))
            d += pd.Timedelta(days=every + int(rng.integers(-5, 6)))
    else:
        cycle, fallow = {"rice": (120, (15, 30)), "maize": (130, (20, 40)),
                         "sugarcane": (365, (10, 20))}[crop]
        sow = planting
        while True:
            d = sow + pd.Timedelta(days=cycle + int(rng.integers(-10, 11)))
            if d > end:
                break
            out.append((d, sow))
            sow = d + pd.Timedelta(days=int(rng.integers(*fallow)))
    return [(d, w) for d, w in out if start <= d <= end]


def make_harvests(rng, fields, farms, hourly, moisture_by_field, region_names) -> pd.DataFrame:
    joined = farms.set_index("farm_id")["joined_date"]
    end = pd.Timestamp(END_DATE)
    rows = []
    for f in fields.itertuples():
        spec = CROPS[f.crop]
        weather = hourly[region_names[f.region_id]]
        times = weather["time"]
        theta = moisture_by_field[f.field_id]
        rain_daily = weather.set_index("time")["precipitation"].resample("D").sum()
        temp_daily = weather.set_index("time")["temperature_2m"].resample("D").mean()
        start = max(pd.Timestamp(START_DATE), pd.Timestamp(joined[f.farm_id]))
        for d, w in harvest_dates(rng, f.crop, pd.Timestamp(f.planting_date), start, end):
            mask = ((times >= max(w, pd.Timestamp(START_DATE))) & (times < d)).to_numpy()
            if mask.sum() < 24 * 20:
                continue
            window = theta[mask]
            stress = np.mean(window < spec.moisture_min)
            waterlog = np.mean(window > spec.moisture_max)
            water_factor = 1 - 0.6 * stress - 0.3 * waterlog
            t_mean = temp_daily[w:d].mean()
            temp_factor = max(0.6, 1 - 0.025 * abs(t_mean - spec.t_opt))
            noise = rng.lognormal(0, 0.08)
            yield_kg = spec.yield_per_ha * f.area_ha * water_factor * temp_factor * noise
            rain7 = rain_daily[d - pd.Timedelta(days=7):d].sum()
            # Quality drops with water stress and with heavy rain right before harvest.
            score = water_factor - min(0.15, 0.0015 * rain7) + rng.normal(0, 0.05)
            grade = "A" if score > 0.85 else ("B" if score > 0.7 else "C")
            rows.append(
                {
                    "field_id": f.field_id,
                    "harvest_date": d.date(),
                    "yield_kg": round(float(yield_kg), 1),
                    "quality_grade": grade,
                }
            )
    df = pd.DataFrame(rows).sort_values(["harvest_date", "field_id"]).reset_index(drop=True)
    df.insert(0, "harvest_id", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    hourly, meta = load_weather()
    regions, crops, farms, fields = make_entities(rng, meta)
    region_ids = dict(zip(regions["name"], regions["region_id"], strict=True))
    region_names = {v: k for k, v in region_ids.items()}
    weather_daily = daily_weather(hourly, region_ids)
    sensors = make_sensors(rng, farms, fields)

    # Physical simulation per region (all fields of a region share the weather).
    moisture_by_field, soil_temp_by_field = {}, {}
    for region_id, rf in fields.groupby("region_id"):
        w = hourly[region_names[region_id]]
        m = simulate_moisture(rng, w, rf)
        s = simulate_soil_temp(rng, w, len(rf))
        for j, fid in enumerate(rf["field_id"]):
            moisture_by_field[fid] = m[:, j]
            soil_temp_by_field[fid] = s[:, j]

    field_info = fields.set_index("field_id")
    readings, alerts = [], []
    for s in sensors.itertuples():
        f = field_info.loc[s.field_id]
        w = hourly[region_names[f.region_id]]
        times = pd.DatetimeIndex(w["time"])
        first = max(pd.Timestamp(START_DATE), pd.Timestamp(s.installed_at))
        if s.sensor_type == "ph":
            ts = pd.date_range(first.normalize() + pd.Timedelta(hours=6), END_DATE + " 06:00",
                               freq="D")
            lo, hi = REGION_PH[region_names[f.region_id]]
            level, target = rng.uniform(lo, hi), None
            vals = np.empty(len(ts))
            for i in range(len(ts)):
                target = level if target is None else target
                level += 0.02 * (target - level) + rng.normal(0, 0.012)
                if level < 5.0 and rng.random() < 0.01:  # liming raises pH
                    level += rng.uniform(0.3, 0.6)
                vals[i] = level
            vals = vals + rng.normal(0, 0.04, len(vals))
        else:
            sel = (times >= first) & (times.hour % READING_EVERY_H == 0)
            ts = times[sel]
            if s.sensor_type == "soil_moisture":
                vals = moisture_by_field[s.field_id][sel] + rng.normal(0, 0.4, sel.sum())
            else:
                vals = soil_temp_by_field[s.field_id][sel] + rng.normal(0, 0.2, sel.sum())
        keep = outage_mask(rng, ts)
        ts, vals = np.asarray(ts[keep]), np.round(vals[keep], 2)
        readings.append(pd.DataFrame({"sensor_id": s.sensor_id, "ts": ts, "value": vals}))
        for a in alerts_for(s.sensor_type, f.crop, ts, vals):
            alerts.append({"sensor_id": s.sensor_id, **a})

    sensor_readings = pd.concat(readings, ignore_index=True)
    alerts_df = pd.DataFrame(alerts).sort_values(["ts", "sensor_id"]).reset_index(drop=True)
    alerts_df.insert(0, "alert_id", range(1, len(alerts_df) + 1))
    harvests = make_harvests(rng, fields, farms, hourly, moisture_by_field, region_names)

    return {
        "regions": regions,
        "crops": crops,
        "weather_daily": weather_daily,
        "farms": farms,
        "fields": fields.drop(columns=["region_id"]),
        "sensors": sensors,
        "sensor_readings": sensor_readings,
        "harvests": harvests,
        "alerts": alerts_df[["alert_id", "sensor_id", "ts", "alert_type", "level", "value"]],
    }


if __name__ == "__main__":
    tables = generate()
    for name, df in tables.items():
        print(f"{name:16s} {len(df):>10,}")
