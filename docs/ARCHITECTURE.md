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
| Conversation | Respond to mentions, replies, and ongoing conversations; prefer one conversation partner; avoid spam. |
| Decisions | Decide whether to respond, ignore, treat a conversation as active, or start a conversation. |
| Realism | Show typing indicators; delay responses based on reading, thinking, and typing time with random variation. |
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
│   │   ├── postprocess.py      # turns raw model output into Discord messages
│   │   └── response_manager.py # orchestrates generation
│   ├── conversation/
│   │   ├── decision.py         # respond / ignore decisions
│   │   ├── state.py            # active-conversation tracking & focus
│   │   ├── rate_limit.py       # anti-spam limits
│   │   ├── timing.py           # human-like delay model
│   │   ├── initiative.py       # when to start a conversation
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
   3. state.ConversationTracker.observe   (channel activity)
   4. decision.DecisionEngine.decide      → RESPOND / IGNORE + reason
   5. if RESPOND → enqueue on the channel's ChannelSession
   ▼
ChannelSession (one asyncio task per channel, serialised)
   1. read: reaction delay + reading time of unread messages; follow-ups that
      arrive meanwhile are read too (no typing indicator)
   2. pick whose messages to answer (priority, then the partner, then oldest)
   3. typing indicator ON and ai.ResponseManager.generate(context) at once
        prompts.PromptBuilder → provider.chat() → postprocess
   4. keep typing until a 60 wpm typist would be done (generation time counts)
   5. send part 1 (as a Discord reply only when it disambiguates)
   6. for each further part: pause → typing → send
   7. update conversation state & rate limiter
```

### 3.2 Background loops (per bot)

* **Maintenance loop** – expires inactive conversations, hands finished
  conversations to long-term memory extraction, prunes short-term messages by
  age and per-channel cap, prunes expired memories.
* **Initiative loop** – periodically checks opted-in channels and, when the
  `InitiativePlanner` agrees (quiet channel, recent human presence, daily cap,
  random chance), posts a conversation opener on one of the personality's topics.

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

### 4.2 Personalities are data, not code paths

A personality is a validated, immutable `Personality` model composed of
profiles: identity, `StyleProfile`, `BehaviorProfile`, `MemoryProfile`,
`ModelProfile`, plus interests, conversation topics, and example messages. The
engine never branches on a personality's name. Adding a personality means
adding `arcane/personalities/<id>/personality.py` that exports `PERSONALITY`;
the registry discovers it by id.

### 4.3 Multiple bots in one process

`ARCANE_BOTS=mp3,debate_bot` starts one Discord client per personality, each
with its own token (`MP3_DISCORD_TOKEN`, …). Clients share one event loop, one
database connection, and one provider pool. All stored rows carry a `bot_id`,
so memories and conversation state stay isolated per personality.

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
(`mention`, `reply_to_bot`, `continuation`, `focused_on_other_user`,
`rate_limited`, …) that is logged, which makes behaviour debuggable.

Decision priorities (first match wins):

1. Ignore own messages, other bots (unless enabled), and empty messages.
2. DMs → respond (configurable).
3. Direct @mention or a reply to one of the bot's messages → respond.
4. The bot's name used in text → respond with high probability.
5. Active conversation:
   * the current partner keeps talking → respond, even without a reply or a
     mention, unless the message is clearly addressed to someone else (a Discord
     reply to another person, or a message that opens with an @mention of
     someone else; replying to oneself or mentioning someone in passing does
     not count);
   * another user who isn't engaging the bot → ignore (focus on the partner).
6. Someone answers an opener the bot posted → respond and adopt them as partner.
7. Idle channel + message strongly matching the personality's interests →
   small chance to join in.
8. Otherwise ignore.

Any "respond" outcome is still subject to rate limits.

### 4.6 Conversation focus

`ConversationTracker` keeps per-channel state: the primary partner, other
participants, last exchange times, and whether the bot is waiting for answers
to an opener. The partner only changes when a new user directly engages and the
current partner has been quiet longer than the focus timeout. This produces the
"stick with one person, but don't ignore people who talk to me" behaviour.

### 4.7 Human-like timing

`HumanTiming` follows what a person does in a chat:

* **reaction** – a short moment to notice new messages;
* **reading** – proportional to the length of what was received, at the
  personality's reading speed. No typing indicator is shown. Messages that
  arrive while reading extend the reading phase, so a burst is answered once;
* **typing** – the typing indicator starts at the same moment the model starts
  generating. The reply is sent when a person typing at `typing_speed_wpm`
  (60 by default, five characters per word) would be done, or as soon as
  generation finishes if that takes longer;
* **inter-message pauses** – when a reply is split into several messages, each
  further part gets a short pause and its own typing time.

Reading and typing times get log-normal jitter. Humanisation can be disabled
(`ARCANE_HUMANIZE=false`) for development.

### 4.8 Prompt layout and model caching

Local inference is the slowest step, and most of its cost is reading the
prompt. Ollama (like llama.cpp) keeps the previous prompt in its KV cache and
only re-evaluates from the first token that differs, and its chat templates
render every `system` message at the top of the prompt. The prompt is laid out
around both facts:

1. a **static system prompt** (personality, ground rules, server/channel/topic)
   that is byte-identical across turns in a channel;
2. the **history**, whose first message the handler keeps fixed for several
   turns (`MemoryProfile.history_slack`) instead of sliding it by one message;
3. a **turn note** (time, focus, profile, memories, task, impossible-request
   instruction) appended to the last user turn, so it never invalidates the
   cached prefix.

Every request also uses the same `num_ctx`, memory extraction included, because
Ollama reloads the model when the context size changes.

### 4.9 Output post-processing

Models drift. `ResponsePostProcessor` makes output safe and natural regardless:

* strips `<think>` reasoning blocks (DeepSeek-R1, Qwen3, …);
* strips speaker labels (`mp3:`, `Assistant:`) and wrapping quotes;
* removes markdown scaffolding (headers, bold, bullet markers);
* removes assistant clichés at the start of a reply ("Great question!");
* applies the personality's style switches: no exclamation points, lowercase
  message starts;
* neutralises `@everyone`/`@here` (also blocked via `AllowedMentions`);
* flags replies matching the personality's `blocked_patterns`, which are then
  regenerated or dropped;
* splits on blank lines into at most N messages and enforces Discord's
  2000-character limit at sentence boundaries.

### 4.10 Memory model

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
  (interests, opinions, background) with an importance score.
* Candidate memories are filtered for sensitive patterns (emails, phone
  numbers, tokens), de-duplicated, and capped per user; the lowest-value
  memories are evicted first.
* Recall ranks by importance and recency and records access statistics.
* `forget_user()` removes everything stored about a user.

The extractor is a strategy (`MemoryExtractor`), so heuristic, embedding-based,
or vector-store implementations can replace it later.

### 4.11 Database

SQLite through `aiosqlite` keeps I/O off the event loop. The connection uses
WAL mode, foreign keys, and a busy timeout. Schema changes are versioned
migrations tracked with `PRAGMA user_version`, applied automatically at
startup. Data access lives in the memory classes behind narrow methods, so a
move to PostgreSQL later only touches the database layer and SQL dialect.

### 4.12 Honesty, limits, and content

mp3 plays a character: a normal 18-year-old guy hanging out in a Discord
server, not a customer-service assistant. The framework keeps that character
safe and honest:

* it does **not** claim to be human when sincerely asked: Discord marks bot
  accounts with an APP badge, and the ground rules tell the model to be honest,
  in its own voice, if someone sincerely asks whether it is an AI;
* it only does what a text bot can do. Requests to join a voice call, video
  chat, play a game, send pictures, swap socials or meet up are turned down
  with a casual excuse; it never agrees or promises to do them later. A
  pattern-based guard adds a direct instruction whenever such a request is
  detected;
* it keeps personal details vague and makes no real-world commitments;
* it is family friendly: casual swearing is a personality choice, but slurs,
  sexual content, flirting, hate and harassment are ruled out by the ground
  rules, and operators can add `blocked_patterns` as a hard filter.

### 4.13 Security

* Secrets come only from environment variables / `.env` (git-ignored) and are
  held as `SecretStr`; they are never logged.
* `AllowedMentions` disables `@everyone`, `@here`, and role pings globally.
* Message content is never logged; logs record decisions and their reasons.
* Rate limits per channel and per user prevent mention-spam abuse and runaway
  costs.
* Bots ignore other bots by default to prevent bot-to-bot loops.
* SQL uses parameters exclusively.
* Optional channel allow-lists restrict where a bot listens and speaks.

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
