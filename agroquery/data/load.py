"""(Re)create the schema and bulk-load all generated tables with COPY."""

from __future__ import annotations

import io
import time

import pandas as pd
import psycopg

from agroquery.config import DATABASE_URL, ROOT, SCHEMA_SQL
from agroquery.data.generate_farms import generate

LOAD_ORDER = [
    "regions", "crops", "weather_daily", "farms", "fields",
    "sensors", "sensor_readings", "harvests", "alerts",
]
CHUNK_ROWS = 500_000


def copy_df(cur: psycopg.Cursor, table: str, df: pd.DataFrame) -> None:
    cols = ", ".join(df.columns)
    with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as copy:
        for start in range(0, len(df), CHUNK_ROWS):
            buf = io.StringIO()
            df.iloc[start:start + CHUNK_ROWS].to_csv(buf, index=False, header=False)
            copy.write(buf.getvalue())


def main() -> None:
    t0 = time.perf_counter()
    tables = generate()
    print(f"[load] generated data in {time.perf_counter() - t0:.1f}s")
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        for name in LOAD_ORDER:
            t1 = time.perf_counter()
            copy_df(cur, name, tables[name])
            print(f"[load] {name:16s} {len(tables[name]):>10,} rows  "
                  f"({time.perf_counter() - t1:.1f}s)")
        cur.execute((ROOT / "db" / "post_load.sql").read_text(encoding="utf-8"))
    print(f"[load] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
