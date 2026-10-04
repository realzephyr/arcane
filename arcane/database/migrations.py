"""Versioned schema migrations.

Each migration runs once, inside a transaction, and bumps SQLite's
``PRAGMA user_version``. Never edit a released migration: append a new one.

Timestamps are stored as unix seconds (REAL). Every table is scoped by
``bot_id`` (the personality id) so multiple bots can share one database while
keeping their memories separate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="initial schema: messages, conversations, profiles, memories, initiatives",
        statements=(
            # Short-term memory: recent messages in channels a bot can see.
            """
            CREATE TABLE messages (
                bot_id               TEXT    NOT NULL,
                message_id           INTEGER NOT NULL,
                channel_id           INTEGER NOT NULL,
                guild_id             INTEGER,
                author_id            INTEGER NOT NULL,
                author_name          TEXT    NOT NULL,
                author_is_bot        INTEGER NOT NULL DEFAULT 0,
                is_self              INTEGER NOT NULL DEFAULT 0,
                content              TEXT    NOT NULL,
                reply_to_message_id  INTEGER,
                reply_to_author_name TEXT,
                created_at           REAL    NOT NULL,
                PRIMARY KEY (bot_id, message_id)
            )
            """,
            "CREATE INDEX idx_messages_channel_time ON messages (bot_id, channel_id, created_at)",
            "CREATE INDEX idx_messages_created ON messages (created_at)",
            # Short-term memory: state of active conversations, one per channel.
            """
            CREATE TABLE conversations (
                bot_id               TEXT    NOT NULL,
                channel_id           INTEGER NOT NULL,
                guild_id             INTEGER,
                partner_id           INTEGER,
                partner_name         TEXT,
                participants         TEXT    NOT NULL DEFAULT '{}',
                started_at           REAL    NOT NULL,
                last_activity_at     REAL    NOT NULL,
                bot_turns            INTEGER NOT NULL DEFAULT 0,
                user_turns           INTEGER NOT NULL DEFAULT 0,
                awaiting_reply_since REAL,
                PRIMARY KEY (bot_id, channel_id)
            )
            """,
            # Long-term memory: who the bot has talked to.
            """
            CREATE TABLE user_profiles (
                bot_id            TEXT    NOT NULL,
                user_id           INTEGER NOT NULL,
                display_name      TEXT    NOT NULL,
                first_seen_at     REAL    NOT NULL,
                last_seen_at      REAL    NOT NULL,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (bot_id, user_id)
            )
            """,
            # Long-term memory: durable facts. user_id NULL = general memory.
            """
            CREATE TABLE memories (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id           TEXT    NOT NULL,
                user_id          INTEGER,
                kind             TEXT    NOT NULL,
                content          TEXT    NOT NULL,
                importance       REAL    NOT NULL CHECK (importance >= 0 AND importance <= 1),
                created_at       REAL    NOT NULL,
                updated_at       REAL    NOT NULL,
                last_accessed_at REAL,
                access_count     INTEGER NOT NULL DEFAULT 0,
                expires_at       REAL,
                source           TEXT
            )
            """,
            "CREATE INDEX idx_memories_user ON memories (bot_id, user_id)",
            """
            CREATE INDEX idx_memories_expiry ON memories (expires_at)
            WHERE expires_at IS NOT NULL
            """,
            # Conversations the bot started on its own (for daily caps and topic variety).
            """
            CREATE TABLE initiatives (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id     TEXT    NOT NULL,
                channel_id INTEGER NOT NULL,
                topic      TEXT    NOT NULL,
                created_at REAL    NOT NULL
            )
            """,
            "CREATE INDEX idx_initiatives_channel ON initiatives (bot_id, channel_id, created_at)",
        ),
    ),
)
