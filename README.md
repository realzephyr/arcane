# Arcane

**Arcane** is a framework for AI-powered Discord bots that take part in conversations
the way people do: with a consistent personality, memory of what was said, a sense of
who they are talking to, and the patience to read before they type.

The first personality is **mp3**: an 18-year-old debate-club regular who got hooked on
philosophy through 2am youtube rabbit holes. He'll argue either side just to see which
one holds up, keeps score when someone lands a point, concedes cleanly, and loves being
made to rethink something. Debates happen right there in chat: he calls them **text
debates** and never does vc. When he isn't talking with anyone, he chimes into whichever
channel is most active with a take or a question about debate or philosophy. He types
short and lowercase, never uses exclamation points, swears mildly but keeps it family
friendly, and turns down anything he can't do from a text chat ("nah i dont do vc"). The
framework is built so that more personalities, more bot accounts, and other AI models can
be added without touching the core.

> Status: **v0.2.0** (unreleased). Single process, local inference via Ollama, SQLite
> storage. See the [changelog](CHANGELOG.md) and the [roadmap](#roadmap).

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running](#running)
- [Architecture](#architecture)
- [Adding a personality](#adding-a-personality)
- [Development](#development)
- [Roadmap](#roadmap)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Security & privacy](#security--privacy)

---

## Features

**Conversation**
- Responds to @mentions, to replies to its own messages, and to its name in text.
- Understands Discord reply chains and includes them in the model's context.
- Keeps ongoing conversations going: once it's talking with someone, it follows
  their plain messages too (no reply or mention needed), and it prefers to stay
  with one partner instead of hopping between users, while still answering anyone
  who engages it directly.
- Chimes in on its own when it isn't talking with anyone: in the channel where
  people talked most recently, it replies to a message worth debating or drops a
  take on a debate or philosophy topic (see [Chiming in](#chiming-in)).
- Takes debate invitations as **text debates** in the chat; asked to debate in
  vc, it declines and offers a text debate instead.
- Reads bursts in full: three quick messages from someone get one answer.
- Turns down things it can't do from a text chat (voice calls, games, pictures,
  meetups, socials) with a casual excuse instead of agreeing.

**Human-like interaction**
- Reads first, types once the reply is ready: it notices and reads the message
  while the model drafts the reply, with nothing shown in the channel. Only then
  does `mp3 is typing...` appear, for as long as typing that reply takes.
- Short replies are quick, long ones take longer: "nothing much" shows about 2
  seconds of typing, a 107-character sentence about 7 seconds.
- Splits longer replies into separate messages with natural pauses.
- Replies via Discord's reply feature only when it helps disambiguate.

**AI**
- Prompts combine the personality, recent conversation, relevant long-term memory,
  Discord context (server, channel, topic, time), and what it knows about the user.
- Output is cleaned into raw Discord text: no AI labels, no markdown scaffolding,
  no `<think>` blocks, no assistant clichés, no mass mentions. Per-personality
  switches remove exclamation points, lowercase message starts, and block terms.
  Replies that talk about how the bot works, or echo its private instructions,
  are regenerated.
- Prompts are laid out for Ollama's prompt cache (static system prompt, stable
  history, per-turn note at the end), so replies don't re-read the whole
  conversation every time.
- Provider-agnostic AI layer; Ollama is the first backend, with thinking turned off
  automatically for models that have a thinking mode.

**Memory**
- Short-term memory of messages, participants, timestamps, and conversation state,
  with automatic expiry and per-channel caps so the database cannot grow unbounded.
- Long-term memory framework that stores only durable, important facts extracted when
  a conversation ends, with de-duplication, sensitive-data filtering, and eviction.
  Extraction waits until the bot isn't busy talking, so it never delays a reply.

**Framework**
- Multiple personalities, each with its own identity, style, behaviour, memory
  profile, and model.
- Multiple bot accounts in one process, sharing the database and model backend.
- Every decision (respond / ignore / chime in) is logged with a reason.

---

## How it works

```
message ──► adapter ──► short-term memory ──► decision engine ──┬─► ignore
                                                                 │
                                                                 ▼
                                                        channel session
             notice + read while the model drafts (nothing shown) │
               (static persona prompt + cached history            │
                + per-turn note; follow-ups → draft again)        │
                                                                 ▼
                                      post-process → split into messages
                                                                 │
                                                                 ▼
               per message: typing indicator for its typing time → send
```

Each channel has its own session that serialises replies, so the bot never talks
over itself. Background loops expire stale conversations, prune old data, extract
long-term memories, and look for a chance to chime in.

### A reply, step by step

```
"hey, what's up?" arrives
│
├─ notice + read ─────────────┐  nothing shown in the channel
│  model drafts the reply ────┤  they send more before typing starts?
│                             │  → the draft is thrown away and redone
│                             ▼    for everything they said
├─ mp3 is typing...  for as long as the reply takes to type
│      "nothing much"                    about 2 s
│      a 107-character sentence          about 7 s
▼
send
```

1. **Notice and read.** A short reaction moment (0.3-1 s), then reading time at 500
   words per minute (at most 6 s). The model starts drafting the moment the message
   is accepted, so it works during this phase instead of after it. A bare
   attention-getter like "yo" or "mp3" (15 characters or less, no question mark) gets
   an extra 2 s, because the real message usually follows.
2. **Fold in follow-ups.** If the same person sends more before typing starts (within
   20 s, at most 3 times), the draft is thrown away and a new one is written for
   everything they said, so a burst gets one answer.
3. **Type.** Once the draft is ready and the messages have had time to be read, the
   typing indicator shows for 1.2 s plus the reply's length at 210 words per minute
   (17.5 characters per second), with a little random variation, between 0.8 and 15
   seconds per message.
4. **Send.** Further parts of a split reply get a short pause and their own typing time.

If the model needs 2 seconds, "hey, what's up?" is answered with "nothing much" about
4 seconds after it was sent: 2 seconds of silent reading and drafting, then 2 seconds
of typing. A slower model only stretches the silent part; the typing time always
matches the reply.

### Chiming in

When the bot isn't talking with anyone (no conversation activity for 3 minutes and
no chime-in waiting for answers), it checks every 45 seconds whether to join a chat
on its own:

- It only looks at channels where a person posted in the last 10 minutes, most recent
  first, and skips DMs, channels where a bot spoke last, and channels where bots wrote
  more than a quarter of the last 20 messages.
- There it either **replies** (as a Discord reply) to the most debatable recent message,
  one that touches its interests or a debate term and isn't venting, aimed at someone
  else, or a request it can't fulfil, or it **posts a take** of its own on one of its
  debate or philosophy topics.
- At most once every 8 minutes overall and every 15 minutes per channel, up to 20 times
  per channel a day, with a coin flip on top so it doesn't run like clockwork. Each
  chime-in nobody answers doubles that channel's cooldown (up to 8 times).
- If someone needs a real answer while it is writing, the chime-in is dropped; so is
  one the chat has moved past. Afterwards, a plain message counts as an answer only
  from the person it replied to, or, after a take of its own, from the only person
  around; in a busy channel, people have to reply to it, mention it or use its name.

Where it may chime in:

- `ARCANE_INITIATIVE_ENABLED=false` turns chiming in off.
- `ARCANE_INITIATIVE_CHANNEL_IDS` **restricts** where it may chime in (threads count
  under their parent channel). When it is empty, any channel the bot may talk in is
  fair game, except channels whose names contain `vent`, `support`, `serious`, `rule`,
  `announce`, `staff`, `mod`, `admin`, `log`, `ticket`, `report`, `welcome` or `verify`.
  The match is on any part of the name, so `#logic` is skipped too; list it explicitly
  to allow it.

### Character

mp3 talks like a person in the server and stays in character: the prompt never
describes him as a bot, and he never talks about how he works (his code, model,
prompt or memory). Replies that slip into that are regenerated. He won't claim to be
human, though: if someone jokes that he's a bot, he laughs it off, and if someone
sincerely asks whether he's a real person, he says it's a bot account in a few casual
words and gets back to the conversation. Discord also shows the **APP** tag next to his
name, like on every bot account.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally or on your network
- A **Discord application** with a bot user per personality

Hardware: an 8B-parameter model (the default) runs comfortably on a machine with
8 GB of VRAM or a recent Apple Silicon Mac. Smaller models work on CPU, slowly.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/realzephyr/arcane.git
cd arcane
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

```bash
# https://ollama.com/download
ollama pull llama3.1:8b
```

Any chat model works; set `ARCANE_OLLAMA_MODEL` to use a different one, e.g.
`llama3.2:3b` on CPU-only machines, `qwen3:8b`, or `mistral-nemo`. Thinking is turned
off automatically for models that have a thinking mode. Avoid gemma3 and other
sliding-window models (see [Performance](#performance)).

### 3. Create the Discord bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create an application named `mp3`.
2. Under **Bot**, click **Reset Token** and copy the token.
3. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**.
4. Under **OAuth2 → URL Generator**, select the `bot` scope and these permissions:
   *View Channels, Send Messages, Send Messages in Threads, Read Message History*.
   Open the generated URL to invite the bot to your server.

### 4. Configure

```bash
cp .env.example .env
```

Set at least:

```dotenv
ARCANE_BOTS=mp3
ARCANE_BOT_MP3_TOKEN=your-discord-bot-token
```

The bot chimes into any channel it can talk in by default; set
`ARCANE_INITIATIVE_CHANNEL_IDS` to limit where (see [Chiming in](#chiming-in)).

### 5. Verify and run

```bash
python main.py check     # validates config, database, Ollama, and the model
python main.py           # starts the bot(s)
```

To talk to a personality in your terminal without Discord (handy for tuning):

```bash
python main.py chat --personality mp3
```

### Docker

```bash
cp .env.example .env                                  # add your token
docker compose up -d                                  # Arcane + an Ollama container
docker compose exec ollama ollama pull llama3.1:8b    # once, to download the model
docker compose run --rm arcane check                  # preflight inside the container
docker compose logs -f arcane
```

The image runs as an unprivileged user; the database and logs live in named volumes.
To use an Ollama server that already runs on the host, see the comments in
[`docker-compose.yml`](docker-compose.yml).

---

## Configuration

All configuration comes from environment variables (or `.env`). See
[`.env.example`](.env.example) for the full annotated list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARCANE_BOTS` | `mp3` | Comma-separated personality ids to run, one bot account each. |
| `ARCANE_BOT_<ID>_TOKEN` | — | Discord token for that bot (required). |
| `ARCANE_BOT_<ID>_MODEL` | — | Model override for that bot. |
| `ARCANE_BOT_<ID>_ALLOWED_CHANNEL_IDS` | — | Per-bot override of the channel allow-list. |
| `ARCANE_BOT_<ID>_INITIATIVE_CHANNEL_IDS` | — | Per-bot override of the chime-in channel list. |
| `ARCANE_ALLOWED_CHANNEL_IDS` | *(all)* | Channels bots may read and talk in. |
| `ARCANE_INITIATIVE_CHANNEL_IDS` | *(any allowed)* | Restricts where bots may chime in on their own. Empty = any allowed channel, minus names like `vent`, `support`, `mod`, `log`, `rules`. |
| `ARCANE_INITIATIVE_ENABLED` | `true` | Master switch for chiming in on its own. |
| `ARCANE_RESPOND_IN_DMS` | `true` | Whether bots answer direct messages. |
| `ARCANE_HUMANIZE` | `true` | Human-like pacing: reading time, typing indicator and typing time, folding in follow-ups. `false` replies as fast as the model allows. |
| `ARCANE_CONVERSATION_TIMEOUT_SECONDS` | *(personality)* | Override for when a conversation is considered over. |
| `ARCANE_AI_PROVIDER` | `ollama` | Default AI provider. |
| `ARCANE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `ARCANE_OLLAMA_MODEL` | `llama3.1:8b` | Default model. |
| `ARCANE_OLLAMA_TIMEOUT_SECONDS` | `120` | Per-request timeout. |
| `ARCANE_OLLAMA_KEEP_ALIVE` | `24h` | How long Ollama keeps the model loaded (`-1m` = forever). |
| `ARCANE_OLLAMA_THINK` | *(auto)* | Unset: thinking is turned off for models that have a thinking mode. `true`/`false` forces it for every model. |
| `ARCANE_OLLAMA_MAX_CONCURRENT_REQUESTS` | `1` | Concurrent requests sent to Ollama. |
| `ARCANE_DATABASE_PATH` | `data/arcane.db` | SQLite database file. |
| `ARCANE_SHORT_TERM_TTL_HOURS` | `24` | Age after which stored messages are deleted. |
| `ARCANE_MAX_MESSAGES_PER_CHANNEL` | `300` | Hard cap of stored messages per channel. |
| `ARCANE_LONG_TERM_MEMORY_ENABLED` | `true` | Extract durable facts when conversations end. |
| `ARCANE_LOG_LEVEL` | `INFO` | `DEBUG` logs every respond/ignore/chime-in decision with its reason. |
| `ARCANE_LOG_FILE` | — | Rotating log file path. |

Personality-specific behaviour (response probabilities, timing, chime-in cooldowns and
caps, memory sizes, sampling parameters) lives in each personality's definition.

---

## Running

```bash
python main.py                 # same as `python main.py run`
python main.py run             # start all bots in ARCANE_BOTS
python main.py check           # preflight checks, non-zero exit on failure
python main.py chat -p mp3     # terminal chat with a personality
python -m arcane --help        # equivalent entry point
```

Stop with `Ctrl+C` (or `SIGTERM` in containers); Arcane shuts down cleanly. The bots
log in right away; the model is loaded in the background meanwhile.

---

## Architecture

```
arcane/
├── main.py                     # entry point
├── arcane/
│   ├── app.py                  # composition root & lifecycle (multi-bot)
│   ├── runtime.py              # wires one personality's conversation stack
│   ├── cli.py                  # run / check / chat
│   ├── config/                 # settings.py (env-driven), logging.py
│   ├── core/                   # domain models, errors, clock, background tasks
│   ├── ai/
│   │   ├── providers/          # LLMProvider interface, Ollama client, factory
│   │   ├── prompts.py          # prompt assembly
│   │   ├── guards.py           # impossible requests, debates, "are you a bot"
│   │   ├── postprocess.py      # raw model output → Discord messages
│   │   └── response_manager.py # generation orchestration
│   ├── conversation/           # platform-agnostic conversation core
│   │   ├── decision.py         # respond / ignore decisions
│   │   ├── state.py            # active conversations & focus
│   │   ├── rate_limit.py       # anti-spam
│   │   ├── timing.py           # human-like delays
│   │   ├── initiative.py       # chiming in on its own
│   │   ├── transport.py        # platform interface
│   │   └── handler.py          # per-channel pipeline
│   ├── bot/                    # discord.py client, events, adapters, transport
│   ├── memory/                 # short-term, long-term, extraction
│   ├── database/               # SQLite connection, migrations, row models
│   └── personalities/          # Personality schema, registry, mp3/
├── tests/                      # pytest suite (no network, no Discord needed)
├── docs/                       # ARCHITECTURE.md, PERSONALITIES.md
├── data/                       # SQLite database (git-ignored)
├── Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml
```

Design principles:

- **Layered.** Discord code depends on the conversation core, never the reverse.
  The core speaks to platforms through a small `MessageTransport` interface.
- **Personalities are data.** The engine never branches on a personality's name;
  everything personality-specific is a validated profile.
- **Providers are pluggable.** The AI layer depends on an `LLMProvider` interface.
- **Bounded by design.** Concurrency, rate, storage, and prompt size all have limits.

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the requirements analysis,
runtime flow, and the reasoning behind each design decision.

---

## Adding a personality

1. Create `arcane/personalities/<id>/__init__.py` and `personality.py`.
2. In `personality.py`, define `PERSONALITY = Personality(...)` (see `mp3`).
3. Add the id to `ARCANE_BOTS` and set `ARCANE_BOT_<ID>_TOKEN`.

See [`docs/PERSONALITIES.md`](docs/PERSONALITIES.md) for every field and tips on
writing a persona that stays in character.

---

## Development

```bash
pip install -r requirements-dev.txt

ruff check .                          # lint
ruff format .                         # format
python -m mypy arcane tests main.py   # type-check (strict)
pytest                                # tests
```

The test suite needs neither Discord nor Ollama: it uses an in-process fake Ollama
server, a fake transport, and spec'd discord.py mocks. It covers configuration,
the Ollama client, the database and both memory layers, personalities, prompts,
guards, post-processing, the decision engine, timing, chiming in, the full
conversation pipeline, the Discord adapters, the CLI, and the application lifecycle.

Set `ARCANE_HUMANIZE=false` and `ARCANE_LOG_LEVEL=DEBUG` while iterating to remove
delays and see every decision with its reason. `python main.py chat -p mp3` runs the
real pipeline against your model without Discord; add `--humanize` to feel the
reading and typing delays.

CI runs lint, format check, strict type-check, tests on Python 3.11 to 3.13, and a
Docker build on every push and pull request.

---

## Roadmap

**v0.1 — Foundation**
- [x] Modular architecture with provider, personality, memory, and platform layers
- [x] mp3 personality
- [x] Decision engine, conversation focus, rate limiting
- [x] Typing indicators and human-like timing
- [x] Short-term memory with expiry; long-term memory framework with extraction
- [x] Conversation initiation in opted-in channels
- [x] Multiple bots per process
- [x] CLI (`run`, `check`, `chat`), Docker, CI

**v0.2 — Natural pacing & voice** *(current, unreleased)*
- [x] Read while the model drafts, then type for as long as the reply takes
- [x] Follow-ups before typing starts are folded into one answer
- [x] Chiming into the most active channel when idle, with cooldowns and backoff
- [x] Prompt layout that reuses Ollama's prompt cache; consistent `num_ctx`;
      thinking turned off automatically
- [x] Conversation continuity for plain follow-up messages
- [x] Declines impossible requests (voice calls, games, pictures, meetups); text debates
- [x] mp3 rewritten as an 18-year-old debate-club regular; family friendly,
      no exclamation points, never talks about how it works

**v0.3 — Quality of conversation**
- [ ] Rolling conversation summaries to extend context beyond the history window
- [ ] Embedding-based memory recall (semantic search over long-term memory)
- [ ] Better addressee detection in busy group conversations
- [ ] Image and attachment understanding via multimodal models
- [ ] Per-guild personality tuning

**v0.4 — Operations**
- [ ] Slash commands: `/forget-me`, `/memories`, admin controls
- [ ] Metrics (latency, decisions, token usage) and health endpoint
- [ ] Additional providers (OpenAI-compatible APIs, llama.cpp, Anthropic)
- [ ] Provider fallback chains

**v0.5 — Multi-personality**
- [ ] `philosophy_bot` and `debate_bot` personalities
- [ ] Safe bot-to-bot conversations with loop protection
- [ ] Shared world knowledge between personalities

**Later**
- [ ] PostgreSQL backend for multi-instance deployments
- [ ] Sharding and horizontal scaling
- [ ] Web dashboard for personalities and memory inspection

---

## Performance

The typing indicator only shows once the reply is ready, and only for as long as
typing it takes (about 2 seconds for a short reply). Everything before that is the
reaction and reading time (usually 1-2 seconds) overlapped with the model writing the
reply. So if replies feel slow, the silent part before typing starts is the model.

Every reply is logged like this:

```
[mp3] #general: replied to alice with 1 message(s): read and generated in 3.4s
(model llama3.1:8b took 2.9s, prompt 1830 tokens, 1702 cached, attempt 1), typed for 2.1s
```

- **read and generated in** is the silent part, from accepting the message to starting
  to type. When it is close to the reading time (1-2 s for a short message), the model
  keeps up; when it is much longer, the model is the bottleneck.
- **model took** is the generation time of the draft that was sent, retries included.
- **prompt / cached**: when most of the prompt is cached, the model only read the new
  messages. A low cached count on consecutive replies means the cache is being missed.
- **attempt 2** means the first draft was discarded (the reason is logged just before)
  and generated again, which doubles the model time.
- **typed for** is the typing indicator time for all parts, by design.

A warning is logged when the prompt plus the reply nearly fill the context window
(more than 95% of `context_window`): older messages then get cut off and the cache
stops working. Raise `context_window` or lower `history_char_budget` in the personality.

- **Use a GPU and a model that fits it.** A model that spills out of VRAM into system
  RAM, or runs on CPU, is many times slower than one that fits on the GPU. On CPU,
  prefer a 3B model (`llama3.2:3b`, `qwen2.5:3b`) and keep `max_tokens` around 120-160.
- **Leave thinking off.** Models with a thinking mode (qwen3, deepseek-r1, ...) think
  before every answer. Arcane asks Ollama about each model and turns thinking off
  (or down to "low" for models that can't switch it off); the log says so the first
  time the model is used. `ARCANE_OLLAMA_THINK=true` or `false` overrides this for every model.
- **Use Ollama 0.30 or newer.** It reuses the cached prompt prefix between requests;
  0.33.3+ also reports cached tokens in the logs. `python main.py check` shows the
  server version.
- **Keep the model loaded.** `ARCANE_OLLAMA_KEEP_ALIVE=24h` is the default (`-1m` keeps it
  forever), and Arcane loads the model in the background at startup. A warning is logged
  if Ollama reloads it mid-session.
- **Don't fight the cache.** Arcane keeps the system prompt identical across turns, moves
  the conversation window in jumps, and sends the same `num_ctx` on every request (a
  different `num_ctx` makes Ollama reload the model). Bots that share a model should use
  the same `context_window`.
- **Side requests stay out of the way.** Memory extraction waits while the bot is
  talking and chime-ins are dropped when someone needs an answer. On older Ollama
  versions, `OLLAMA_NUM_PARALLEL=2` on the server additionally gives them their own
  cache slot so they don't evict the conversation's cached prompt, at the cost of the
  memory for a second context.
- **On CPU, reading the prompt dominates.** Leave flash attention on auto (or set
  `OLLAMA_FLASH_ATTENTION=1`) and consider `OLLAMA_KV_CACHE_TYPE=q8_0`.
- **Avoid sliding-window models** such as gemma3: llama.cpp may re-read the whole prompt
  for them on every request, which defeats the cache.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `Message Content Intent is not enabled` | Enable it under **Bot → Privileged Gateway Intents** in the Developer Portal. |
| `Discord rejected the token` | Wrong or reset token in `ARCANE_BOT_<ID>_TOKEN`. |
| Bot is online but never answers | The channel isn't in `ARCANE_ALLOWED_CHANNEL_IDS`, or the bot lacks *Send Messages* there. Run with `ARCANE_LOG_LEVEL=DEBUG` to see each decision. |
| `AI backend 'ollama' is unavailable` | Ollama isn't running or `ARCANE_OLLAMA_BASE_URL` is wrong. Bots stay silent until it's reachable. |
| `model '...' is not installed` | Run `ollama pull <model>`. |
| Replies are slow | Check the per-reply log line (see [Performance](#performance)). If "read and generated" is much longer than "typed for", the model is the bottleneck. `ARCANE_HUMANIZE=false` removes the simulated delays entirely. |
| Bot chimes in where it shouldn't | Set `ARCANE_INITIATIVE_CHANNEL_IDS` to the channels it may chime into, or `ARCANE_INITIATIVE_ENABLED=false`. |
| Output contains reasoning text | Thinking is turned off automatically when Ollama reports the model's capabilities; on an older Ollama, set `ARCANE_OLLAMA_THINK=false`. |
| `the prompt ... nearly fills the context window` | Raise the personality's `context_window` or lower its `history_char_budget`. |

`python main.py check` diagnoses most of these in one go.

---

## Security & privacy

- Tokens and secrets are read from the environment only and never logged (a log
  filter also redacts anything shaped like a Discord token). `.env` is git-ignored.
- Bots can never ping `@everyone`, `@here`, roles, or users.
- Message content is never written to logs; logs contain decisions, display names,
  and channel names.
- Bots request only the gateway intents they need (guilds, messages, message
  content), run as an unprivileged user in Docker, and use parameterised SQL.
- Short-term messages expire automatically and are capped per channel; messages
  deleted on Discord are deleted from storage.
- Long-term memory stores only extracted facts, filters obvious sensitive data
  (emails, phone numbers, secrets), and can be wiped per user
  (`LongTermMemory.forget_user`; a user-facing `/forget-me` command is on the roadmap).
- Per-channel and per-user rate limits stop mention-spam and runaway inference.
- People's messages can't imitate the bot's private per-turn instructions: square
  brackets and quotes in their text are neutralised before it reaches the prompt.
- mp3 talks like a person in the server, but it does not claim to be human: if someone
  sincerely asks, it says it's a bot account. Discord marks it with the APP tag.

All data stays on the machine running Arcane and your Ollama server.

---

## License

No license has been chosen yet. Until one is added, all rights are reserved by the
repository owner.
