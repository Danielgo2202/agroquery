"""Schema introspection and the three schema renderings the strategies use.

* full schema     - every table and column with its documentation, the values
                    of categorical columns and 3 sample rows (S1, S3, S4)
* table catalog   - one line per table, column names only (S2, step 1)
* compact schema  - only the tables/columns selected in S2 step 1, plus the
                    keys needed to join them (S2, step 2)

All documentation comes from COMMENT ON statements in db/schema.sql, so the
agent sees exactly what a new analyst would read in the database itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agroquery.config import DATABASE_URL
from agroquery.db import connect

SAMPLE_ROWS = 3
MAX_LISTED_VALUES = 12


@dataclass
class Column:
    name: str
    type: str
    comment: str = ""
    primary_key: bool = False
    references: str | None = None  # "table(column)"
    values: list[str] = field(default_factory=list)  # low-cardinality text columns


@dataclass
class Table:
    name: str
    comment: str = ""
    columns: list[Column] = field(default_factory=list)
    sample_rows: list[tuple] = field(default_factory=list)

    @property
    def pk(self) -> list[str]:
        return [c.name for c in self.columns if c.primary_key]

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class SchemaInfo:
    tables: dict[str, Table]
    glossary: str = ""


_INTROSPECT_COLUMNS = """
SELECT c.relname AS table_name, a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       coalesce(col_description(c.oid, a.attnum), '') AS column_comment,
       coalesce(obj_description(c.oid, 'pg_class'), '') AS table_comment
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY c.oid, a.attnum
"""

_INTROSPECT_KEYS = """
SELECT con.contype, src.relname, sa.attname, dst.relname, da.attname
FROM pg_constraint con
JOIN pg_class src ON src.oid = con.conrelid
JOIN pg_namespace n ON n.oid = src.relnamespace
CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, i)
JOIN pg_attribute sa ON sa.attrelid = src.oid AND sa.attnum = k.attnum
LEFT JOIN pg_class dst ON dst.oid = con.confrelid
LEFT JOIN pg_attribute da ON da.attrelid = con.confrelid AND da.attnum = con.confkey[k.i]
WHERE n.nspname = 'public' AND con.contype IN ('p', 'f')
"""


def introspect(url: str = DATABASE_URL) -> SchemaInfo:
    tables: dict[str, Table] = {}
    with connect(url) as conn, conn.cursor() as cur:
        cur.execute(_INTROSPECT_COLUMNS)
        for tname, cname, ctype, ccomment, tcomment in cur.fetchall():
            t = tables.setdefault(tname, Table(tname, tcomment))
            t.columns.append(Column(cname, ctype, ccomment))
        cur.execute(_INTROSPECT_KEYS)
        for contype, src, scol, dst, dcol in cur.fetchall():
            col = tables[src].column(scol)
            if contype == "p":
                col.primary_key = True
            else:
                col.references = f"{dst}({dcol})"
        for t in tables.values():
            order = ", ".join(t.pk) or "1"
            cur.execute(f"SELECT * FROM {t.name} ORDER BY {order} LIMIT {SAMPLE_ROWS}")
            t.sample_rows = cur.fetchall()
            for c in t.columns:
                if c.type in ("text", "character(1)") and not c.primary_key:
                    cur.execute(f"SELECT DISTINCT {c.name} FROM {t.name} ORDER BY 1 "
                                f"LIMIT {MAX_LISTED_VALUES + 1}")
                    vals = [r[0] for r in cur.fetchall()]
                    if len(vals) <= MAX_LISTED_VALUES:
                        c.values = vals
        cur.execute("SELECT coalesce(obj_description('public'::regnamespace), '')")
        glossary = cur.fetchone()[0]
    return SchemaInfo(tables, glossary)


# -----------------------------------------------------------------------------
# Renderings
# -----------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _render_table(t: Table, columns: list[Column]) -> str:
    composite_pk = len(t.pk) > 1
    body: list[tuple[str, str]] = []
    for c in columns:
        extra = " PRIMARY KEY" if c.primary_key and not composite_pk else ""
        if c.references:
            extra += f" REFERENCES {c.references}"
        notes = [c.comment] if c.comment else []
        if c.values:
            notes.append("Values: " + ", ".join(f"'{v}'" for v in c.values) + ".")
        body.append((f"    {c.name} {c.type}{extra}", " ".join(notes)))
    if composite_pk:
        body.append((f"    PRIMARY KEY ({', '.join(t.pk)})", ""))

    lines = [f"CREATE TABLE {t.name} (" + (f"  -- {t.comment}" if t.comment else "")]
    for i, (code, note) in enumerate(body):
        sep = "," if i < len(body) - 1 else ""
        lines.append(code + sep + (f"  -- {note}" if note else ""))
    lines.append(");")

    if t.sample_rows:
        idx = [i for i, c in enumerate(t.columns) if c in columns]
        lines.append(f"/* {len(t.sample_rows)} sample rows from {t.name}:")
        lines.append("   " + " | ".join(t.columns[i].name for i in idx))
        for row in t.sample_rows:
            lines.append("   " + " | ".join(_fmt(row[i]) for i in idx))
        lines.append("*/")
    return "\n".join(lines)


def _with_glossary(info: SchemaInfo, parts: list[str]) -> str:
    head = [f"-- {info.glossary}"] if info.glossary else []
    return "\n\n".join(head + parts)


def render_full_schema(info: SchemaInfo) -> str:
    return _with_glossary(info, [_render_table(t, t.columns) for t in info.tables.values()])


def render_table_catalog(info: SchemaInfo) -> str:
    """One line per table: name, column names and the table's description."""
    lines = []
    for t in info.tables.values():
        cols = ", ".join(c.name for c in t.columns)
        lines.append(f"- {t.name}({cols}): {t.comment}")
    return "\n".join(lines)


def render_compact_schema(info: SchemaInfo, selection: dict[str, list[str]]) -> str:
    """Selected tables/columns only, always keeping primary and foreign keys so
    the selected tables can still be joined. Unknown names are ignored."""
    parts = []
    for tname, wanted in selection.items():
        if tname not in info.tables:
            continue
        t = info.tables[tname]
        wanted_set = set(wanted)
        cols = [c for c in t.columns if c.name in wanted_set or c.primary_key or c.references]
        parts.append(_render_table(t, cols))
    return _with_glossary(info, parts)
