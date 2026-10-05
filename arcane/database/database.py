"""Async SQLite access.

One :class:`Database` instance wraps one ``aiosqlite`` connection shared by
every bot in the process. ``aiosqlite`` runs SQLite on a dedicated thread, so
queries never block the event loop. Writes are serialised with an asyncio lock
so that explicit transactions from different coroutines cannot interleave.

The connection runs in autocommit mode; use :meth:`Database.transaction` to
group statements atomically.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from arcane.database.migrations import MIGRATIONS, Migration

logger = logging.getLogger(__name__)

Params = Sequence[Any] | Mapping[str, Any]
"""Positional (``?``) or named (``:name``) query parameters."""
MEMORY_DATABASE = ":memory:"


class Database:
    """Lifecycle, migrations, and query helpers for the SQLite database."""

    def __init__(self, path: Path | str, *, migrations: Sequence[Migration] = MIGRATIONS) -> None:
        self._path = str(path)
        self._migrations = migrations
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database is not connected; call connect() first")
        return self._connection

    # ---------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        """Open the connection, apply pragmas, and run pending migrations."""
        if self._connection is not None:
            return
        if self._path != MEMORY_DATABASE:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        connection = await aiosqlite.connect(self._path, isolation_level=None)
        connection.row_factory = aiosqlite.Row
        try:
            if self._path != MEMORY_DATABASE:
                await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.execute("PRAGMA busy_timeout=5000")
            await connection.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            await connection.close()
            raise
        self._connection = connection
        await self.migrate()
        logger.info("Database ready at %s (schema v%d)", self._path, await self.schema_version())

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # --------------------------------------------------------------- migrations

    async def schema_version(self) -> int:
        row = await self.fetch_one("PRAGMA user_version")
        return int(row[0]) if row is not None else 0

    async def migrate(self) -> None:
        """Apply every migration newer than the stored ``user_version``."""
        current = await self.schema_version()
        for migration in sorted(self._migrations, key=lambda m: m.version):
            if migration.version <= current:
                continue
            logger.info("Applying migration %d: %s", migration.version, migration.description)
            async with self.transaction() as connection:
                for statement in migration.statements:
                    await connection.execute(statement)
                # PRAGMA does not accept parameters; version is an int from code.
                await connection.execute(f"PRAGMA user_version = {int(migration.version)}")

    # ------------------------------------------------------------------ queries

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run statements atomically; rolls back on any exception."""
        async with self._write_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                await connection.execute("ROLLBACK")
                raise
            else:
                await connection.execute("COMMIT")

    async def execute(self, sql: str, params: Params = ()) -> int:
        """Execute a write statement and return the number of affected rows."""
        async with self._write_lock:
            cursor = await self.connection.execute(sql, params)
            try:
                return cursor.rowcount
            finally:
                await cursor.close()

    async def execute_returning_id(self, sql: str, params: Params = ()) -> int:
        """Execute an INSERT and return the new row id."""
        async with self._write_lock:
            cursor = await self.connection.execute(sql, params)
            try:
                if cursor.lastrowid is None:
                    raise RuntimeError("statement did not insert a row")
                return cursor.lastrowid
            finally:
                await cursor.close()

    async def execute_many(self, sql: str, rows: Iterable[Params]) -> None:
        async with self.transaction() as connection:
            await connection.executemany(sql, rows)

    async def fetch_one(self, sql: str, params: Params = ()) -> aiosqlite.Row | None:
        async with self.connection.execute(sql, params) as cursor:
            row: aiosqlite.Row | None = await cursor.fetchone()
            return row

    async def fetch_all(self, sql: str, params: Params = ()) -> list[aiosqlite.Row]:
        async with self.connection.execute(sql, params) as cursor:
            return list(await cursor.fetchall())
