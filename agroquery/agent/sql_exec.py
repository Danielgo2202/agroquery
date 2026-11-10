"""Safe execution of (possibly model-generated) SQL.

Queries run as the read-only role `agro_reader` (default_transaction_read_only,
statement_timeout = 30 s, see db/post_load.sql) and at most MAX_ROWS rows are
fetched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psycopg

from agroquery.config import READONLY_DATABASE_URL
from agroquery.db import connect

MAX_ROWS = 10_000


@dataclass
class ExecResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def empty(self) -> bool:
        return self.ok and not self.rows


class SQLExecutor:
    def __init__(self, url: str = READONLY_DATABASE_URL, timeout_s: int = 30):
        self.url = url
        self.timeout_s = timeout_s
        self._conn: psycopg.Connection | None = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = connect(self.url, autocommit=True)
            self._conn.execute(f"SET statement_timeout = '{self.timeout_s}s'")
        return self._conn

    def run(self, sql: str) -> ExecResult:
        conn = self._connection()  # raises DatabaseUnavailable; never scored as a model error
        t0 = time.perf_counter()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return ExecResult(error="Statement returned no result set.",
                                      elapsed_s=time.perf_counter() - t0)
                columns = [d.name for d in cur.description]
                rows = cur.fetchmany(MAX_ROWS + 1)
            truncated = len(rows) > MAX_ROWS
            return ExecResult(columns, rows[:MAX_ROWS], None, time.perf_counter() - t0, truncated)
        except psycopg.Error as exc:
            msg = (exc.diag.message_primary if exc.diag and exc.diag.message_primary else str(exc))
            return ExecResult(error=f"{type(exc).__name__}: {msg}",
                              elapsed_s=time.perf_counter() - t0)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
