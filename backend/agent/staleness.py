"""
Our own confidence/staleness layer.

Verified fact (see the project doc's "Verified Against the API" section):
Hindsight's reflect() does NOT return a staleness flag, a confidence score, or
a source-tier label. It tracks freshness internally to decide what to trust
while reasoning, but never hands that state back to the client.

So the CONFIRMED / INFERRED / STALE badges shown in the UI are entirely our
own bookkeeping, stored locally, keyed by the same `entity` tag we pass to
Hindsight's retain(). We track, per code entity (a method or class name):
  - last_validated_at: when a human last confirmed an explanation
  - validated_by: who confirmed it
  - last_code_changed_at: when we last re-ingested a changed version of it

confidence(entity) derives the badge from those two timestamps:
  - never validated                                  -> INFERRED (yellow)
  - validated, and code hasn't changed since          -> CONFIRMED (green)
  - validated, but code changed after validation       -> STALE (red)
"""
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass

# Anchored to this file's location (backend/agent/), resolved to an absolute
# path once at import time -- otherwise running scripts from different
# working directories (e.g. `python scripts\ingest.py` from the project
# root vs `python -m uvicorn main:app` from backend/) resolves this relative
# path differently and creates separate, out-of-sync database files.
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legacy_sme.db"))


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_state (
            entity TEXT PRIMARY KEY,
            last_validated_at REAL,
            validated_by TEXT,
            last_code_changed_at REAL,
            last_explanation TEXT
        )
    """)
    return conn


@dataclass
class EntityStatus:
    entity: str
    badge: str          # "CONFIRMED" | "INFERRED" | "STALE"
    last_validated_at: float | None
    validated_by: str | None
    last_code_changed_at: float | None


def mark_code_seen(entity: str, explanation: str) -> None:
    """Call this every time an entity is (re-)ingested from the codebase."""
    now = time.time()
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT last_explanation FROM entity_state WHERE entity = ?", (entity,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO entity_state (entity, last_code_changed_at, last_explanation) "
                "VALUES (?, ?, ?)",
                (entity, now, explanation),
            )
        elif row[0] != explanation:
            # the generated explanation changed -> the underlying code changed
            conn.execute(
                "UPDATE entity_state SET last_code_changed_at = ?, last_explanation = ? "
                "WHERE entity = ?",
                (now, explanation, entity),
            )
        conn.commit()


def mark_validated(entity: str, validated_by: str) -> None:
    """Call this when a human confirms/corrects an explanation in chat."""
    now = time.time()
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO entity_state (entity, last_validated_at, validated_by) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(entity) DO UPDATE SET last_validated_at = ?, validated_by = ?",
            (entity, now, validated_by, now, validated_by),
        )
        conn.commit()


def status(entity: str) -> EntityStatus:
    with closing(_connect()) as conn:

        row = conn.execute(
            "SELECT last_validated_at, validated_by, last_code_changed_at "
            "FROM entity_state WHERE entity = ?",
            (entity,),
        ).fetchone()

        # ----------------------------------------------------
        # If this entity has never been tracked before,
        # create an INFERRED entry for it.
        # ----------------------------------------------------
        if row is None:

            conn.execute(
                """
                INSERT INTO entity_state (
                    entity,
                    last_validated_at,
                    validated_by,
                    last_code_changed_at,
                    last_explanation
                )
                VALUES (?, NULL, NULL, NULL, NULL)
                """,
                (entity,),
            )

            conn.commit()

            return EntityStatus(
                entity=entity,
                badge="INFERRED",
                last_validated_at=None,
                validated_by=None,
                last_code_changed_at=None,
            )

        last_validated_at, validated_by, last_code_changed_at = row

    # --------------------------------------------------------
    # Never validated by a human
    # --------------------------------------------------------

    if last_validated_at is None:

        return EntityStatus(
            entity=entity,
            badge="INFERRED",
            last_validated_at=None,
            validated_by=None,
            last_code_changed_at=last_code_changed_at,
        )

    # --------------------------------------------------------
    # Code changed after human validation
    # --------------------------------------------------------

    if (
        last_code_changed_at
        and last_code_changed_at > last_validated_at
    ):

        badge = "STALE"

    # --------------------------------------------------------
    # Human validated and code has not changed
    # --------------------------------------------------------

    else:

        badge = "CONFIRMED"

    return EntityStatus(
        entity=entity,
        badge=badge,
        last_validated_at=last_validated_at,
        validated_by=validated_by,
        last_code_changed_at=last_code_changed_at,
    )


def all_entities() -> list[EntityStatus]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT entity FROM entity_state").fetchall()
    return [status(r[0]) for r in rows]
