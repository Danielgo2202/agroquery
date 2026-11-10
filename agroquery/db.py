"""Database connections that fail fast with a helpful message."""

from __future__ import annotations

import psycopg

CONNECT_TIMEOUT_S = 10


class DatabaseUnavailable(RuntimeError):
    pass


def connect(url: str, **kwargs) -> psycopg.Connection:
    try:
        return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S, **kwargs)
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailable(
            "Cannot reach PostgreSQL. Is Docker running? Start the database with "
            f"`uv run agroquery up`.\n  ({exc})"
        ) from exc
