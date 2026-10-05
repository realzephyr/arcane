# Changelog

All notable changes to Arcane are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-05

Natural pacing, faster local inference, steadier conversations, and a new voice
for mp3.

### Changed

- **Pacing.** The bot now reads first and types second, like a person. It
  notices new messages (0.3-1.2 s), reads them at a human reading speed with no
  typing indicator, and absorbs follow-ups that arrive meanwhile. The typing
  indicator starts the moment the model starts generating, and the reply is
  sent when someone typing at 60 wpm would be done, or as soon as generation
  finishes if that takes longer. The fixed 2.5 s debounce, the "thinking" delay
  and the complexity bonus (6-10 s before anything happened) are gone.
- `TimingProfile` is speed-based: `typing_speed_wpm` (default 60) and
  `reading_speed_wpm` replace the old characters-per-second fields;
  `thinking_seconds`, `debounce_seconds` and `max_debounce_seconds` are removed
  and `max_reading_seconds` is new. Personalities must be updated.
- **Prompt layout for speed.** The system prompt is now static per channel; the
  time, focus, memories and task travel in a short note appended to the last
  user turn, and the history window's first message stays fixed for several
  turns. Ollama re-reads a prompt only from the first changed token, so
  consecutive replies reuse the cached system prompt and history instead of
  re-reading the whole conversation.
- Memory extraction uses the same `num_ctx` as replies, so Ollama no longer
  reloads the model between them; default `context_window` is 4096 and
  `max_tokens` 200; default history is 16 messages.
- **Conversation continuity.** A conversation partner's plain messages (no
  reply, no mention) keep the conversation going. Only a Discord reply to
  someone else or a message opening with an @mention of someone else counts as
  "addressed elsewhere"; replying to yourself or mentioning someone in passing
  no longer drops the bot out of the conversation.
- Reply limits raised to 12 per user and 15 per channel per minute (the old
  4 per user cut off normal back-and-forth); conversation timeout 10 minutes,
  focus timeout 2 minutes.
- Ground rules: the bot can only read and type text, declines voice calls,
  video, games, pictures, friend requests, meetups and socials with a casual
  excuse, makes no real-world commitments, and keeps it family friendly.

- **mp3 is now a normal 18-year-old guy on the internet** (chosen by a judge
  panel over three drafts): short lowercase messages, never an exclamation
  point, light casual swearing, family friendly, declines voice calls, games,
  pictures, meetups and socials with a casual excuse, honest about being a bot
  when sincerely asked, and still into history, space and what-if arguments.
- `keep_alive` defaults to 24h and the model is preloaded at startup with the
  same `num_ctx` replies use.

### Added

- `arcane/ai/guards.py`: detects 14 kinds of requests the bot can't fulfil
  (voice/video calls, streaming, games, media, friend requests, socials, phone
  calls, meetups, links, reminders, pings, reactions, server invites) and adds
  a direct per-turn instruction to decline. Validated against 455 example
  messages in `tests/data/guard_cases.json`.
- Reply logs show prompt and cached token counts; a warning is logged when
  Ollama reloads the model mid-session.
- `StyleProfile.allow_exclamation_points`, `lowercase_starts` and
  `blocked_patterns`.
- `IncomingMessage.addressed_user_ids` (leading @mentions).

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
