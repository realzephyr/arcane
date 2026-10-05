# Changelog

All notable changes to Arcane are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-05

First release: the foundation of the framework and the mp3 personality.

### Added

- Layered architecture: configuration, core models, AI providers, prompt
  building, post-processing, conversation core, memory, database, and Discord
  integration, with the conversation core independent of Discord.
- Environment-driven settings with validation, plus per-bot overrides
  (`ARCANE_BOT_<ID>_*`) for running several bot accounts in one process.
- Provider-agnostic AI layer with an Ollama backend (bounded concurrency,
  timeouts, retries with backoff, health checks, JSON mode).
- Personality framework (identity, style, timing, behaviour, initiative,
  memory, and model profiles) and the **mp3** personality.
- Prompt builder that combines personality, history with reply relationships,
  Discord context, user profile, and long-term memories, within a size budget.
- Output post-processing: strips reasoning blocks, labels, markdown,
  assistant clichés, and hallucinated turns; neutralises mass mentions; splits
  replies into Discord-sized messages.
- Decision engine with logged reasons, conversational focus on one partner,
  per-channel/per-user rate limiting, burst debouncing, and stale-trigger
  dropping.
- Human-like timing: reading, complexity-aware thinking, typing indicators
  and durations, and pauses between messages, all with natural variation.
- Conversation initiation in opted-in channels with daily caps, quiet-period
  checks, and topic rotation.
- SQLite storage with migrations; short-term memory with TTL and per-channel
  caps; long-term memory with extraction, de-duplication, sensitive-data
  filtering, eviction, and per-user forgetting.
- CLI: `run`, `check`, and `chat` (terminal conversation).
- Docker image, Docker Compose setup with Ollama, and GitHub Actions CI
  (lint, strict type-check, tests on Python 3.11 to 3.13, image build).
