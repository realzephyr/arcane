# Arcane Architecture

This document records the requirements analysis, the architecture, and the design
decisions behind Arcane. It is the reference for anyone extending the framework.

---

## 1. Requirements analysis

Arcane is a framework for AI-driven Discord participants. Distilled from the
project brief, the system must:

| Area | Requirement |
|------|-------------|
| Identity | Run a personality (first: **mp3**) with a stable identity, interests, and writing style. |
| Conversation | Respond to mentions, replies, and ongoing conversations; prefer one conversation partner; chime into active chats when idle; avoid spam. |
| Decisions | Decide whether to respond, ignore, treat a conversation as active, or chime in. |
| Realism | Pace replies like a person: read first, then show the typing indicator for as long as typing the reply takes, with random variation. |
| Prompting | Combine personality, recent history, memory, Discord context, and user info into each request. |
| Output | Produce raw, Discord-ready text: no AI labels, no markdown scaffolding, length matched to context. |
| Memory | Short-term conversational memory with automatic expiry and bounded growth; a framework for long-term memory of important facts only. |
| Extensibility | Multiple personalities, multiple bot accounts, different styles/behaviours/memory profiles/models, and swappable AI providers. |
| Quality | Clean architecture, security, error handling, performance, tests, documentation. |

Non-functional constraints that shaped the design:

* **Local inference is slow and serial.** Ollama on consumer hardware can take
  several seconds per reply and typically processes one request at a time. The
  design must bound concurrency, tolerate timeouts, and never block the Discord
  gateway.
* **Discord is noisy.** Most messages in a server are not addressed to the bot.
  Ignoring correctly matters as much as replying well.
* **Untrusted input.** Every message is user-controlled text that ends up in a
  prompt and possibly in the database. Mentions must not be weaponised
  (`@everyone`), secrets must never be logged, and stored data must expire.

---

## 2. High-level architecture

Arcane is organised in layers. Dependencies point inwards: the Discord layer
depends on the conversation core, never the other way round.

```
                    ┌────────────────────────────────────────┐
                    │              arcane.app                │  composition root:
                    │  builds one BotRuntime per personality │  wires everything,
                    └───────────────┬────────────────────────┘  owns lifecycle
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
┌────▼─────────────┐     ┌──────────▼───────────┐       ┌──────────▼──────────┐
│   arcane.bot     │     │  arcane.conversation │       │  arcane.personalities│
│ discord.py layer │────►│  platform-agnostic   │◄──────│  identity, style,    │
│ adapters,        │     │  decision engine,    │       │  behaviour, memory & │
│ transport, events│     │  state, timing,      │       │  model profiles      │
└──────────────────┘     │  handler, initiative │       └─────────────────────┘
                         └─────┬──────────┬─────┘
                               │          │
                  ┌────────────▼──┐   ┌───▼──────────────┐
                  │   arcane.ai   │   │  arcane.memory   │
                  │ prompts,      │   │ short-term,      │
                  │ post-process, │   │ long-term,       │
                  │ providers/*   │   │ extraction       │
                  └──────┬────────┘   └───┬──────────────┘
                         │                │
                 ┌───────▼──────┐   ┌─────▼──────────┐
                 │ Ollama (HTTP)│   │ arcane.database│
                 │ future: any  │   │ SQLite + WAL,  │
                 │ LLM provider │   │ migrations     │
                 └──────────────┘   └────────────────┘
```

### Package layout

```
arcane/
├── main.py                     # entry point: `python main.py`
├── arcane/
│   ├── app.py                  # composition root & lifecycle
│   ├── runtime.py              # wires one personality's conversation stack
│   ├── cli.py                  # `run`, `check`, `chat` commands
│   ├── config/                 # settings (env-driven), logging
│   ├── core/                   # domain models, errors, clock, background tasks
│   ├── ai/
│   │   ├── providers/          # LLMProvider interface + Ollama implementation
│   │   ├── prompts.py          # prompt assembly
│   │   ├── guards.py           # impossible requests, debates, "are you a bot"
│   │   ├── postprocess.py      # turns raw model output into Discord messages
│   │   └── response_manager.py # orchestrates generation
│   ├── conversation/
│   │   ├── decision.py         # respond / ignore decisions
│   │   ├── state.py            # active-conversation tracking & focus
│   │   ├── rate_limit.py       # anti-spam limits
│   │   ├── timing.py           # human-like delay model
│   │   ├── initiative.py       # when and where to chime in
│   │   ├── transport.py        # interface the core uses to talk to a platform
│   │   └── handler.py          # per-channel pipeline tying it all together
│   ├── bot/                    # discord.py: client, events, adapters, transport
│   ├── memory/                 # short-term, long-term, extraction
│   ├── database/               # connection, migrations, row models
│   └── personalities/
│       ├── base.py             # Personality schema
│       ├── registry.py         # discovery & loading
│       └── mp3/personality.py  # the first personality
├── tests/                      # pytest suite
├── docs/                       # architecture & guides
└── data/                       # SQLite database (git-ignored)
```

### Why the layout differs from the original sketch

The brief proposed a top-level `discord/` folder. A local package named
`discord` shadows the `discord.py` library whenever code runs from the project
root, which breaks `import discord` in confusing ways. All code therefore lives
in a single importable `arcane` package and the Discord layer is named
`arcane.bot`. The conversation logic was split out of the Discord folder into
`arcane.conversation` so it can be unit-tested without Discord and reused by
future platforms.

---

## 3. Runtime flow

### 3.1 Incoming message

```
Discord gateway
   │ on_message
   ▼
bot.events ──► bot.adapters: discord.Message → IncomingMessage (plain dataclass)
   │
   ▼
conversation.handler.ConversationHandler.handle_message
   1. channel allow-list check
   2. memory.short_term.record_message   (bounded, expiring)
   3. decision.DecisionEngine.decide      → RESPOND / IGNORE + reason
   4. if RESPOND → update focus, enqueue on the channel's ChannelSession, and
      interrupt background model work (chime-in drafts, memory extraction)
   ▼
ChannelSession (one asyncio task per channel, serialised)
   1. pick whose messages to answer (priority, then the partner, then oldest)
   2. reserve a rate-limit slot (released again if nothing gets sent)
   3. read and think, with nothing shown in the channel:
        ai.ResponseManager.generate(context) starts at once
          prompts.PromptBuilder → provider.chat() → postprocess
        ready_at = start + reaction + min(reading, max_reading_seconds)
                   + follow_up_wait (only after a bare "yo")
        follow-ups from the same person before the draft is used
          → cancel the draft, merge them, generate again
        wait until the draft is done and ready_at has passed
   4. for each part: (pause) → typing indicator for its typing time → send
      (the first part as a Discord reply only when it disambiguates)
   5. update conversation state, log the reply timings
```

### 3.2 Background work (per bot)

* **Maintenance loop** – expires inactive conversations, hands finished
  conversations to long-term memory extraction (which gives way to live
  replies, see 4.11), prunes short-term messages by age and per-channel cap,
  prunes expired memories and old chime-in records.
* **Initiative loop** – every `check_interval_seconds` (45 s for mp3, with
  jitter) asks the `InitiativePlanner` whether to chime into the most recently
  active channel (see 4.8).
* **Model warm-up** – at startup, after the preflight check, the model is
  loaded with the `num_ctx` replies use. It runs in the background while the
  bots log in, so a slow model load never keeps them offline.

---

## 4. Key design decisions

### 4.1 Provider abstraction for AI

`arcane.ai.providers.base.LLMProvider` is a small async interface
(`chat()`, `health_check()`, `close()`). Ollama is one implementation using
`aiohttp` against `/api/chat` (no extra SDK dependency; `aiohttp` already ships
with discord.py). Each personality declares its provider and model in a
`ModelProfile`, and the provider factory creates and caches provider instances.
Adding OpenAI-compatible, Anthropic, or llama.cpp backends means adding one
module and one factory entry; nothing else changes.

Providers enforce a concurrency semaphore, timeouts, and bounded retries with
backoff so a slow model degrades gracefully instead of piling up requests.

The Ollama provider also handles thinking models. Unless `ARCANE_OLLAMA_THINK`
is set, it asks `/api/show` once per model: for a model with the `thinking`
capability it sends `think: false`, or the lightest level it offers ("low"
for gpt-oss style models) when thinking can't be switched off, and logs the
choice. A failed check is not remembered, so it is asked again next time. The health check reports
the Ollama server version.

### 4.2 Personalities are data, not code paths

A personality is a validated, immutable `Personality` model composed of
profiles: identity, `StyleProfile`, `TimingProfile`, `BehaviorProfile` (with its
`InitiativeProfile`), `MemoryProfile`, `ModelProfile`, plus interests,
conversation topics, and example messages. The engine never branches on a
personality's name. Adding a personality means adding
`arcane/personalities/<id>/personality.py` that exports `PERSONALITY`; the
registry discovers it by id.

### 4.3 Multiple bots in one process

`ARCANE_BOTS=mp3,debate_bot` starts one Discord client per personality, each
with its own token (`ARCANE_BOT_MP3_TOKEN`, `ARCANE_BOT_DEBATE_BOT_TOKEN`, …).
Clients share one event loop, one database connection, and one provider pool.
All stored rows carry a `bot_id`, so memories and conversation state stay
isolated per personality.

### 4.4 Platform-agnostic conversation core

The conversation layer works with plain dataclasses (`IncomingMessage`,
`ChannelInfo`) and talks to the platform through a `MessageTransport` protocol
(typing indicator, send, channel info). `arcane.bot.transport.DiscordTransport`
is the only implementation today. This keeps the most complex logic fully
testable with fakes and leaves room for other platforms.

### 4.5 Deterministic, testable decisions

`DecisionEngine` is a pure function of the message, the conversation state,
the rate limiter, the personality's behaviour profile, the current time, and an
injected `random.Random`. Every decision carries a machine-readable reason
(`mention`, `reply_to_bot`, `continuation`, `opener_reply`,
`focused_on_other_user`, `rate_limited`, …) that is logged, which makes
behaviour debuggable.

Decision priorities (first match wins):

1. Ignore own messages, other bots (unless enabled), and empty messages.
2. DMs → respond (configurable).
3. Direct @mention or a reply to one of the bot's messages → respond.
4. The bot's name used in text → respond with high probability; always when it
   comes from the current conversation partner.
5. Active conversation:
   * the current partner keeps talking → respond, even without a reply or a
     mention, unless the message is clearly addressed to someone else (a Discord
     reply to another person, or a message that opens with an @mention of
     someone else; replying to oneself or mentioning someone in passing does
     not count);
   * a participant talks after the partner went quiet → respond;
   * another user who isn't engaging the bot → ignore (focus on the partner).
6. Someone answers a chime-in within `opener_reply_window_seconds` →
   respond (`opener_reply`) and adopt them as partner. Whose plain message counts
   is stored in `awaiting_reply_from`: `None` for anyone, a user id for the
   person the bot replied to (or the only person active in the channel), or the
   `NOBODY` sentinel (`0`) in a busy channel, where only replies, mentions and
   its name (rules 3 and 4) count.
7. Idle channel + message strongly matching the personality's interests →
   small chance to join in.
8. Otherwise ignore.

Any "respond" outcome is still subject to rate limits. Limits count replies
actually sent: the handler re-checks them just before generating, reserves the
slot right away (so several channels working on replies at once can't all pass
the check), and releases it if nothing gets sent.

### 4.6 Conversation focus

`ConversationTracker` keeps per-channel state: the primary partner, other
participants, last exchange times, and whether the bot is waiting for answers
to a chime-in. The partner only changes when a new user directly engages the
bot and the current partner has been quiet longer than the focus timeout.
Replying to the partner refreshes their focus, so they keep priority while they
read the reply and type back. This produces the "stick with one person, but
don't ignore people who talk to me" behaviour.

Chiming in starts a fresh conversation in that channel; the previous one is
retired and handed to memory extraction at the next maintenance run.

### 4.7 Human-like timing

`HumanTiming` follows what a person does in a chat: read the message while
working out the answer, then type it.

* **reaction** – a short moment to notice new messages (0.3-1.0 s for mp3);
* **reading** – the length of what was received at `reading_speed_wpm` (500),
  capped at `max_reading_seconds` (6). A message that is only an
  attention-getter (nothing but greetings, the bot's names and mentions, like
  "yo" or "hey mp3") adds `follow_up_wait_seconds` (2), because the real
  message usually follows. Replies in an ongoing exchange (continuation,
  reply to the bot, answer to a chime-in) and absorbed follow-ups never wait;
* **generation overlaps both.** The model starts drafting as soon as the
  message is accepted, silently. The reply is used once the draft is ready and
  `ready_at = start + reaction + min(reading, max_reading) + follow_up_wait`
  has passed. If the same person sends more before then (within 20 s of the
  start, at most 3 times, only with humanised timing), the draft is cancelled
  and regenerated for the merged messages;
* **typing** – only now does the typing indicator show, for
  `typing_start_seconds + length / (typing_speed_wpm × 5 / 60)`: 1.2 s plus
  17.5 characters per second for mp3, so "nothing much" takes about 2 s and a
  107-character sentence about 7.3 s. Clamped to 0.8-15 s per message. The time
  to start the indicator on Discord counts as typing;
* **inter-message pauses** – when a reply is split into several messages, each
  further part gets a short pause (0.4-1.2 s) and its own typing time.

Reading and typing times get median-preserving log-normal jitter. Humanisation
can be disabled (`ARCANE_HUMANIZE=false`) for development: every delay is zero
and follow-ups are no longer folded into a draft.

Each reply is logged with its timings, e.g. `read and generated in 3.4s (model
llama3.1:8b took 2.9s, prompt 1830 tokens, 1702 cached, attempt 1), typed for
2.1s`. A warning is logged when `prompt_tokens + max_tokens` exceeds 95% of
`context_window`.

### 4.8 Chiming in

When a personality isn't talking with anyone, it joins chats on its own, the
way a regular does. `InitiativePlanner` makes the decisions without I/O; the
handler carries them out.

1. **May it chime in at all?** Initiative is enabled, within `active_hours_utc`,
   the bot is not engaged (no channel session busy, no conversation activity
   within `idle_seconds`, no chime-in waiting for answers) and its last
   chime-in anywhere was at least `min_interval_minutes` ago. Attempts are kept
   in the `initiatives` table, so this survives restarts.
2. **Where?** Channels with a person's message within `active_window_seconds`,
   most recent first. Skipped: DMs, channels outside the allow-list, channels
   with a busy session, and, when `ARCANE_INITIATIVE_CHANNEL_IDS` is set,
   channels not in it (threads count under their parent). With an empty list,
   channels whose names contain one of the words vent, support, serious,
   rules, announcements, staff, mod, admin, log, ticket, report, welcome or
   verify are skipped instead (`is_skipped_channel`, whole words only). A
   channel must
   also pass `evaluate_channel`: a person spoke last (not the bot or another
   bot), bots wrote at most 25% of the last 20 messages, fewer than
   `max_per_channel_per_day` chime-ins in the last 24 hours, and the channel
   cooldown has passed. The cooldown is `channel_cooldown_minutes` doubled for
   every chime-in in a row nobody answered (at most 8x). The count lives in
   memory: it resets on restart and whenever the bot answers someone in the
   channel.
3. **Roll the dice.** Only the first eligible channel is considered, with
   probability `chance`.
4. **Reply or post?** `choose_reply_target` scores people's messages younger
   than `reply_max_age_seconds` and newer than the bot's own last message
   (`discussion_score`). A message needs a debate or philosophy hook (interest
   keywords only add to the score); commands, bare links, short messages,
   messages led by an @mention or replying to someone else, venting, and
   impossible requests score zero. Messages someone already replied to, whose
   author posted again, or with more than three messages after them are
   skipped.
   Questions, debate words and freshness add to the score. With a good enough
   target the bot replies to it `reply_chance` of the time (prompt mode
   `join`, sent as a Discord reply); otherwise it posts a take on a topic from
   `conversation_topics` that it hasn't used in its last ten (mode `initiate`).
5. **Write it, but yield.** The attempt is logged before generating, so it
   counts for cooldowns even if generation fails. Generation is cancelled when
   any message anywhere needs a reply. The finished draft is dropped if more
   than two people's messages arrived in the channel meanwhile, or the reply
   target posted again.
6. **Send** with the same typing model as replies, then start a new
   conversation that awaits answers (rule 6 in 4.5): from the reply target; or,
   after a message of its own, from the only person active in the channel; or,
   in a busy channel, from nobody's plain messages (`NOBODY`).

### 4.9 Prompt layout and model caching

Local inference is the slowest step, and most of its cost is reading the
prompt. Ollama (like llama.cpp) keeps the previous prompt in its KV cache and
only re-evaluates from the first token that differs, and its chat templates
render every `system` message at the top of the prompt. The prompt is laid out
around both facts:

1. a **static system prompt** (personality, ground rules, server/channel/topic)
   that is byte-identical across turns in a channel. It describes a person: no
   wording about bots, AI or code, so nothing invites the model to talk about
   how it works;
2. the **history**, whose first message the handler keeps fixed for several
   turns (`MemoryProfile.history_slack`) instead of sliding it by one message.
   Its character budget is lowered automatically when the context window,
   minus `max_tokens`, the system prompt and a reserve for the note, can't hold
   `history_char_budget`;
3. a **turn note** (time, focus, profile, memories, task, guard notes)
   appended to the last user turn, so it never invalidates the cached prefix.

Guard notes come from `arcane.ai.guards`: an instruction to decline an
impossible request, to take a debate invitation as a text debate (or to
decline vc and offer a text debate instead), and, when someone asks whether the
bot is a bot, to laugh off jokes and, if they are sincere, not claim to be human
but say briefly that it's a bot account, without details about how it works.

People can't forge the note: square brackets in their messages and names become
parentheses, and text quoted inside the note (names, the message being
answered, memories) also has double quotes and newlines replaced.

Every request also uses the same `num_ctx`, memory extraction included, because
Ollama reloads the model when the context size changes.

### 4.10 Output post-processing

Models drift. `ResponsePostProcessor` makes output safe and natural regardless:

* strips `<think>` reasoning blocks (DeepSeek-R1, Qwen3, …);
* cuts an echo of the private turn note and everything after it;
* strips speaker labels (`mp3:`, `Assistant:`), lines written for other people,
  and wrapping quotes;
* removes markdown scaffolding (headers, bold, bullet markers);
* removes assistant clichés at the start of a reply ("Great question!"), as
  whole words only;
* applies the personality's style switches: no exclamation points (every
  Unicode exclamation glyph; interrobangs become "?"), lowercase message
  starts;
* neutralises `@everyone`/`@here` (also blocked via `AllowedMentions`);
* splits on blank lines into at most N messages and enforces Discord's
  2000-character limit at sentence boundaries.

It also reports problems cleaning can't fix: a match of the personality's
`blocked_patterns`, leftovers of the private note, and talk about the bot's own
implementation (its context, training, prompt, code, developers or the model
behind it; first-person phrasing, so debate talk about AI in general passes).
`ResponseManager` regenerates such replies once, at a slightly higher
temperature and with a correction for implementation talk, and drops them if
the retry fails too. Near-verbatim repeats of its recent messages, and replies
that agree to an impossible request ("omw", "sent", "maybe later"), are
regenerated the same way.

### 4.11 Memory model

**Short-term memory** (`messages`, `conversations` tables)

* Every message in an allowed channel is stored with author, timestamps, and
  reply relationships, so the bot has accurate context.
* Bounded twice: by age (`ARCANE_SHORT_TERM_TTL_HOURS`) and by a per-channel
  cap (`ARCANE_MAX_MESSAGES_PER_CHANNEL`). A maintenance loop enforces both.
* Deleted Discord messages are removed from storage.
* Conversation state is persisted, and expires after the conversation timeout.

**Long-term memory** (`user_profiles`, `memories` tables)

* Messages are *not* stored permanently. When a conversation ends, an
  extractor asks the model for a handful of durable, non-sensitive facts
  (interests, opinions, background) with an importance score. Output cut off at
  the token limit is salvaged item by item.
* Candidate memories are filtered for sensitive patterns (emails, phone
  numbers, tokens), de-duplicated, and capped per user; the lowest-value
  memories are evicted first.
* Recall ranks by importance and recency and records access statistics.
* `forget_user()` removes everything stored about a user.

Extraction uses the same model as replies, so it yields: finished
conversations wait in a queue while the bot is engaged, and a running
extraction is cancelled (and retried at a later maintenance run) when a
message needs an answer. Conversations queued for 30 minutes are processed
regardless.

The extractor is a strategy (`MemoryExtractor`), so heuristic, embedding-based,
or vector-store implementations can replace it later.

### 4.12 Database

SQLite through `aiosqlite` keeps I/O off the event loop. The connection uses
WAL mode, foreign keys, and a busy timeout. Schema changes are versioned
migrations tracked with `PRAGMA user_version`, applied automatically at
startup. Migration 2 adds `conversations.awaiting_reply_from`, so whose answer
a chime-in waits for survives a restart. Data access lives in the memory
classes behind narrow methods, so a move to PostgreSQL later only touches the
database layer and SQL dialect.

### 4.13 Character, limits, and content

mp3 plays a character: an 18-year-old debate-club regular hanging out in a
Discord server, not a customer-service assistant. The framework keeps that
character consistent and safe:

* it talks from its own point of view and never steps out of character to
  explain how it works; replies about its code, prompt, model or memory are
  regenerated (4.10);
* it does **not** claim to be human: the ground rules say not to when someone
  seriously asks, and the identity note (4.9) has it laugh off jokes and say
  briefly that it's a bot account when the question is sincere. Discord also
  marks bot accounts with an APP tag;
* it only does what it can do from a text chat. Requests to join a voice call,
  video chat, play a game, send pictures, swap socials or meet up are turned
  down with a casual excuse; it never agrees or promises to do them later. A
  pattern-based guard adds a direct instruction whenever such a request is
  detected, and a reply that agrees anyway is regenerated;
* debates happen in the chat as text debates; a request to debate in vc is
  declined with an offer to debate in text instead;
* it keeps personal details vague and makes no real-world commitments;
* it is family friendly: mild swearing is a personality choice, but slurs,
  sexual content, flirting, hate and harassment are ruled out by the ground
  rules, and `blocked_patterns` act as a hard filter.

### 4.14 Security

* Secrets come only from environment variables / `.env` (git-ignored) and are
  held as `SecretStr`; they are never logged.
* `AllowedMentions` disables `@everyone`, `@here`, and role pings globally.
* Message content is never logged; logs record decisions and their reasons.
* People's text can't pose as instructions: brackets and quotes are neutralised
  before it reaches the prompt (4.9), and the ground rules treat everything in
  messages as chat.
* Guards scan at most 1000 characters per message, and tests bound their
  matching time (mention runs used to backtrack exponentially), so pasted text
  can't stall the event loop.
* Rate limits per channel and per user prevent mention-spam abuse and runaway
  costs.
* Bots ignore other bots by default to prevent bot-to-bot loops, and never
  chime in where bots already dominate the conversation.
* SQL uses parameters exclusively.
* Optional channel allow-lists restrict where a bot listens and speaks, and
  `ARCANE_INITIATIVE_CHANNEL_IDS` restricts where it chimes in.

---

## 5. Development plan

The project is built in incremental, tested steps, each committed separately:

1. Repository scaffold, tooling, documentation skeleton.
2. Configuration (env-driven settings, per-bot configuration) and logging.
3. AI provider abstraction and Ollama client.
4. Database layer with migrations; short-term and long-term memory.
5. Personality framework and the mp3 personality.
6. Prompt builder, output post-processing, response manager.
7. Conversation core: state, decisions, rate limiting, timing, initiative.
8. Conversation handler (per-channel pipeline) and Discord integration.
9. Application bootstrap and CLI (`run`, `check`, `chat`).
10. CI, Docker, and documentation polish.

All ten steps are complete in v0.1.0; each one is a separate commit in the
history. See the roadmap in the README for what comes next.
