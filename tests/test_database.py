from __future__ import annotations

from pathlib import Path

import pytest

from arcane.database.database import Database
from arcane.database.migrations import MIGRATIONS, Migration


async def test_connect_creates_file_and_applies_migrations(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "arcane.db"
    async with Database(path) as db:
        assert path.exists()
        assert await db.schema_version() == max(m.version for m in MIGRATIONS)
        tables = {
            row["name"]
            for row in await db.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"messages", "conversations", "user_profiles", "memories", "initiatives"} <= tables
        journal = await db.fetch_one("PRAGMA journal_mode")
        assert journal is not None and journal[0] == "wal"


async def test_migrations_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "arcane.db"
    async with Database(path):
        pass
    async with Database(path) as db:
        assert await db.schema_version() == max(m.version for m in MIGRATIONS)


async def test_new_migrations_are_applied_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "arcane.db"
    async with Database(path):
        pass
    extra = Migration(
        version=max(m.version for m in MIGRATIONS) + 1,
        description="test table",
        statements=("CREATE TABLE extra (id INTEGER PRIMARY KEY)",),
    )
    async with Database(path, migrations=(*MIGRATIONS, extra)) as db:
        assert await db.schema_version() == extra.version
        assert await db.fetch_one("SELECT name FROM sqlite_master WHERE name='extra'")


async def test_transaction_rolls_back_on_error() -> None:
    async with Database(":memory:") as db:
        with pytest.raises(RuntimeError):
            async with db.transaction() as connection:
                await connection.execute(
                    "INSERT INTO initiatives (bot_id, channel_id, topic, created_at) "
                    "VALUES ('mp3', 1, 't', 0)"
                )
                raise RuntimeError("boom")
        row = await db.fetch_one("SELECT COUNT(*) FROM initiatives")
        assert row is not None and row[0] == 0


async def test_execute_helpers() -> None:
    async with Database(":memory:") as db:
        new_id = await db.execute_returning_id(
            "INSERT INTO initiatives (bot_id, channel_id, topic, created_at) VALUES (?, ?, ?, ?)",
            ("mp3", 1, "stoicism", 0.0),
        )
        assert new_id == 1
        await db.execute_many(
            "INSERT INTO initiatives (bot_id, channel_id, topic, created_at) VALUES (?, ?, ?, ?)",
            [("mp3", 1, "a", 1.0), ("mp3", 1, "b", 2.0)],
        )
        changed = await db.execute("DELETE FROM initiatives WHERE channel_id = ?", (1,))
        assert changed == 3


async def test_using_closed_database_fails_clearly() -> None:
    db = Database(":memory:")
    with pytest.raises(RuntimeError, match="not connected"):
        await db.fetch_all("SELECT 1")
