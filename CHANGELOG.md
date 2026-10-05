# Changelog

All notable changes to Arcane are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

Human pacing, chiming into active chats, faster local inference, and a new
debate-club mp3.

### Changed

- **Pacing.** The bot reads first and types once the reply is ready. The model
  starts drafting as soon as a message is accepted, with no typing indicator,
  while the bot notices (0.3-1 s) and reads it (500 wpm, at most 6 s; 2 s more
  after a bare "yo"). Only then does the typing indicator show, for 1.2 s plus
  the reply's length at 210 wpm, so "nothing much" takes about 2 s and a
  107-character sentence about 7 s (0.8-15 s per message). Follow-ups from the
  same person before typing starts discard the draft and regenerate it for
  everything they said (within 20 s, at most 3 times). The debounce, the
  "thinking" delay and the typing indicator during generation are gone.
- `TimingProfile` is speed-based: `typing_speed_wpm` (210), `typing_start_seconds`,
  `reading_speed_wpm` (500), `follow_up_wait_seconds` and `max_reading_seconds`
  replace `typing_speed_cps`, `reading_speed_cps`, `thinking_seconds`,
  `debounce_seconds` and `max_debounce_seconds`. Personalities must be updated.
- **Chiming in replaces quiet-channel openers.** When idle, the bot joins the
  channel where people talked most recently: it replies to a debatable recent
  message or posts a take on one of its topics. `InitiativeProfile` fields are
  now `check_interval_seconds`, `idle_seconds`, `active_window_seconds`,
  `reply_max_age_seconds`, `channel_cooldown_minutes`, `min_interval_minutes`,
  `max_per_channel_per_day`, `chance`, `reply_chance` and `active_hours_utc`;
  `min_quiet_minutes` and `recent_activity_hours` are removed.
  `opener_reply_window_seconds` defaults to 180.
- `ARCANE_INITIATIVE_CHANNEL_IDS` now **restricts** where bots chime in instead
  of opting channels in: empty means any allowed channel, except ones whose
  names contain a word suggesting unprompted chatter is unwelcome (vent,
  support, mod, log, rules, ...).
- **mp3 is an 18-year-old debate-club regular** (chosen by a judge panel over
  three drafts): argues either side, keeps score, concedes cleanly, takes debates
  as text debates, chimes in about debate and philosophy, types short and
  lowercase, never uses exclamation points, swears mildly, and stays family
  friendly. The persona and the static prompt contain nothing about bots, AI
  or code.
- **Prompt layout for speed.** The system prompt is static per channel; time,
  focus, memories, task and guard notes travel in a note appended to the last
  user turn, and the history window's first message stays fixed for several
  turns, so Ollama reuses its cached prompt. The history budget shrinks when
  the context window can't hold it.
- **Ollama.** Thinking is turned off automatically for models with a thinking
  mode (`think: "low"` where it can't be switched off); `ARCANE_OLLAMA_THINK`
  overrides it. `keep_alive` defaults to 24h, every request uses the same
  `num_ctx`, and the model is loaded in the background while the bots log in.
  Defaults: `context_window` 4096, `max_tokens` 200, 16 history messages.
- **Conversation continuity.** A partner's plain messages keep the
  conversation going; only a Discord reply to someone else or a message
  opening with an @mention of someone else counts as addressed elsewhere.
  Conversation timeout 10 minutes, focus timeout 2 minutes, reply limits 12 per
  user and 15 per channel per minute, counted per reply sent.
- Memory extraction waits while the bot is talking and is cancelled when a
  reply is needed (forced after 30 minutes).

### Added

- `arcane/ai/guards.py`: detects 14 kinds of requests the bot can't fulfil
  (voice/video calls, streaming, games, media, friend requests, socials, phone
  calls, meetups, links, reminders, pings, reactions, server invites) and adds
  a per-turn instruction to decline; replies that agree anyway are regenerated
  with a correction. `detect_debate_request` turns debate invitations into text
  debates (vc debates are declined with a text offer), and
  `detect_identity_question` adds a note to laugh off "are you a bot" jokes and,
  when sincere, say briefly that it's a bot account. Validated against the cases
  in `tests/data/guard_cases.json`.
- Ground rules: text debates only, never step out of character to explain how
  it works, don't claim to be a real person when seriously asked, no real-world
  commitments, family friendly.
- Post-processing flags implementation talk (context, training, prompt, code,
  model) and echoes of the private note, and the reply is regenerated.
- `StyleProfile.allow_exclamation_points`, `lowercase_starts` and
  `blocked_patterns`; `IncomingMessage.addressed_user_ids` (leading @mentions).
- Per-reply log line `read and generated in Xs (model ... took Ys, prompt N
  tokens, M cached, attempt K), typed for Zs`; a warning when the prompt nearly
  fills the context window and when Ollama reloads the model; the Ollama server
  version in `python main.py check`.
- Database migration 2: `conversations.awaiting_reply_from`.

### Fixed

- People's text can't forge the private note: brackets and quotes are
  neutralised; note echoes in replies are cut.
- Guard regexes no longer backtrack exponentially on runs of mentions (ReDoS).
- Every Unicode exclamation glyph is removed, not just "!"; banned openers
  match whole words only ("ah" no longer eats "ahaha").
- Memory extraction output cut off at the token limit is salvaged instead of
  discarded.
- Replying to the partner refreshes their focus; the partner using the bot's
  name is always answered; leading @mentions after filler words or emoji count
  as addressing someone.
- A reserved rate-limit slot closes the race where several channels passed
  the check before any recorded a reply.
- Model warm-up no longer delays login.

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
