"""Central configuration: paths, connection strings, study period and regions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
SCHEMA_SQL = ROOT / "db" / "schema.sql"
BENCHMARK_FILE = ROOT / "benchmark" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / ".cache"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agro:agro@localhost:5433/agroquery")
READONLY_DATABASE_URL = os.getenv(
    "READONLY_DATABASE_URL", "postgresql://agro_reader:agro_reader@localhost:5433/agroquery"
)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_USD = float(os.getenv("MAX_USD", "5"))

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "America/Bogota"
SEED = 42


@dataclass(frozen=True)
class Region:
    name: str
    department: str
    latitude: float
    longitude: float
    crops: tuple[str, ...]


# Five agricultural zones of Colombia; weather for each comes from Open-Meteo.
REGIONS: tuple[Region, ...] = (
    Region("Eje Cafetero", "Caldas", 5.07, -75.52, ("coffee", "avocado", "plantain")),
    Region("Llanos del Meta", "Meta", 4.14, -73.63, ("oil_palm", "rice", "cacao")),
    Region("Centro del Tolima", "Tolima", 4.15, -74.88, ("rice", "maize")),
    Region("Norte del Huila", "Huila", 2.93, -75.28, ("cacao", "rice", "maize")),
    Region("Valle del Cauca", "Valle del Cauca", 3.53, -76.30, ("sugarcane", "maize")),
)
