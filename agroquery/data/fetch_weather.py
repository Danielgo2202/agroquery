"""Download real hourly weather for each region from the Open-Meteo historical archive.

Docs: https://open-meteo.com/en/docs/historical-weather-api (free, no API key).
Raw data is cached in data/raw/ so the database can be rebuilt offline.
"""

from __future__ import annotations

import json
import time

import httpx
import pandas as pd

from agroquery.config import END_DATE, RAW_DIR, REGIONS, START_DATE, TIMEZONE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",  # deg C
    "relative_humidity_2m",  # %
    "precipitation",  # mm in the preceding hour
    "shortwave_radiation",  # W/m2, mean of the preceding hour
    "et0_fao_evapotranspiration",  # mm in the preceding hour
]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("é", "e")


def raw_path(region_name: str):
    return RAW_DIR / f"weather_hourly_{slug(region_name)}.parquet"


def fetch_region(client: httpx.Client, region) -> tuple[pd.DataFrame, float]:
    params = {
        "latitude": region.latitude,
        "longitude": region.longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    payload = None
    for attempt in range(5):
        try:
            resp = client.get(ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            wait = 5 * 2**attempt
            print(f"[weather]   attempt {attempt + 1} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)
    if payload is None:
        raise RuntimeError(f"Open-Meteo download failed for {region.name}")
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    # Rare gaps in the reanalysis: interpolate in time, rain gaps become 0.
    df["precipitation"] = df["precipitation"].fillna(0.0)
    df = df.set_index("time").interpolate(method="time").reset_index()
    return df, float(payload["elevation"])


def main(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = RAW_DIR / "regions_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    with httpx.Client() as client:
        for region in REGIONS:
            path = raw_path(region.name)
            if path.exists() and region.name in meta and not force:
                print(f"[weather] cached: {region.name}")
                continue
            print(f"[weather] downloading {region.name} ({START_DATE}..{END_DATE})")
            df, elevation = fetch_region(client, region)
            df.to_parquet(path, index=False)
            meta[region.name] = {"elevation_m": elevation}
            print(f"[weather]   {len(df):,} hourly rows, elevation {elevation:.0f} m")
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
