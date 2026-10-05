# Arcane

**Arcane** is a framework for AI-powered Discord bots that take part in conversations
the way people do: with a consistent personality, memory of what was said, a sense of
who they are talking to, and the patience to type before they speak.

The first personality is **mp3**: a curious, intellectually motivated regular who likes
education, philosophy, science, history, and a good debate. The framework is built so
that more personalities, more bot accounts, and other AI models can be added without
touching the core.

> Status: **v0.1.0 (foundation)**. Single process, local inference via Ollama, SQLite storage.
> See the [changelog](CHANGELOG.md) and the [roadmap](#roadmap).

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
- [Troubleshooting](#troubleshooting)
- [Security & privacy](#security--privacy)

---

## Features

**Conversation**
- Responds to @mentions, to replies to its own messages, and to its name in text.
- Understands Discord reply chains and includes them in the model's context.
- Keeps ongoing conversations going and prefers to stay with one partner instead of
  hopping between users, while still answering anyone who engages it directly.
- Occasionally joins discussions that strongly match its interests.
- Starts conversations on its own in opted-in channels when they've been quiet,
  with daily caps so it never becomes noise.
- Debounces bursts: three quick messages from someone get one considered answer.

**Human-like interaction**
- Shows `mp3 is typing...` before replying.
- Delays are modelled from reading time, thinking time (message complexity), and
  typing time (reply length), with random human variation.
- Splits longer replies into separate messages with natural pauses.
- Replies via Discord's reply feature only when it helps disambiguate.

**AI**
- Prompts combine the personality, recent conversation, relevant long-term memory,
  Discord context (server, channel, topic, time), and what it knows about the user.
- Output is cleaned into raw Discord text: no AI labels, no markdown scaffolding,
  no `<think>` blocks, no assistant clichés, no mass mentions.
- Provider-agnostic AI layer; Ollama is the first backend.

**Memory**
- Short-term memory of messages, participants, timestamps, and conversation state,
  with automatic expiry and per-channel caps so the database cannot grow unbounded.
- Long-term memory framework that stores only durable, important facts extracted when
  a conversation ends, with de-duplication, sensitive-data filtering, and eviction.

**Framework**
- Multiple personalities, each with its own identity, style, behaviour, memory
  profile, and model.
- Multiple bot accounts in one process, sharing the database and model backend.
- Every decision (respond / ignore / initiate) is logged with a reason.

---

## How it works

```
message ──► adapter ──► short-term memory ──► decision engine ──┬─► ignore
                                                                 │
                                                                 ▼
                                                        channel session
                                       debounce → read delay → typing… → generate
                                                  (prompt: personality + history
                                                   + memory + Discord context)
                                                                 │
                                                post-process → split → type → send
```

Each channel has its own session that serialises replies, so the bot never talks
over itself. Background loops expire stale conversations, prune old data, extract
long-term memories, and consider starting new conversations.

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

Any chat model works; set `ARCANE_OLLAMA_MODEL` to use a different one
(e.g. `qwen3:8b`, `gemma3:12b`, `mistral-nemo`).

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
| `ARCANE_BOT_<ID>_INITIATIVE_CHANNEL_IDS` | — | Per-bot override of initiative channels. |
| `ARCANE_ALLOWED_CHANNEL_IDS` | *(all)* | Channels bots may read and talk in. |
| `ARCANE_INITIATIVE_CHANNEL_IDS` | *(none)* | Channels where bots may start conversations. |
| `ARCANE_INITIATIVE_ENABLED` | `true` | Master switch for starting conversations. |
| `ARCANE_RESPOND_IN_DMS` | `true` | Whether bots answer direct messages. |
| `ARCANE_HUMANIZE` | `true` | Typing indicators and human-like delays. |
| `ARCANE_CONVERSATION_TIMEOUT_SECONDS` | *(personality)* | Override for when a conversation is considered over. |
| `ARCANE_AI_PROVIDER` | `ollama` | Default AI provider. |
| `ARCANE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `ARCANE_OLLAMA_MODEL` | `llama3.1:8b` | Default model. |
| `ARCANE_OLLAMA_TIMEOUT_SECONDS` | `120` | Per-request timeout. |
| `ARCANE_OLLAMA_MAX_CONCURRENT_REQUESTS` | `1` | Concurrent requests sent to Ollama. |
| `ARCANE_DATABASE_PATH` | `data/arcane.db` | SQLite database file. |
| `ARCANE_SHORT_TERM_TTL_HOURS` | `24` | Age after which stored messages are deleted. |
| `ARCANE_MAX_MESSAGES_PER_CHANNEL` | `300` | Hard cap of stored messages per channel. |
| `ARCANE_LONG_TERM_MEMORY_ENABLED` | `true` | Extract durable facts when conversations end. |
| `ARCANE_LOG_LEVEL` | `INFO` | `DEBUG` logs every respond/ignore decision with its reason. |
| `ARCANE_LOG_FILE` | — | Rotating log file path. |

Personality-specific behaviour (response probabilities, timing, initiative caps,
memory sizes, sampling parameters) lives in each personality's definition.

---

## Running

```bash
python main.py                 # same as `python main.py run`
python main.py run             # start all bots in ARCANE_BOTS
python main.py check           # preflight checks, non-zero exit on failure
python main.py chat -p mp3     # terminal chat with a personality
python -m arcane --help        # equivalent entry point
```

Stop with `Ctrl+C` (or `SIGTERM` in containers); Arcane shuts down cleanly.

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
│   │   ├── postprocess.py      # raw model output → Discord messages
│   │   └── response_manager.py # generation orchestration
│   ├── conversation/           # platform-agnostic conversation core
│   │   ├── decision.py         # respond / ignore decisions
│   │   ├── state.py            # active conversations & focus
│   │   ├── rate_limit.py       # anti-spam
│   │   ├── timing.py           # human-like delays
│   │   ├── initiative.py       # starting conversations
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
post-processing, the decision engine, timing, initiative, the full conversation
pipeline, the Discord adapters, the CLI, and the application lifecycle.

Set `ARCANE_HUMANIZE=false` and `ARCANE_LOG_LEVEL=DEBUG` while iterating to remove
delays and see every decision with its reason. `python main.py chat -p mp3` runs the
real pipeline against your model without Discord.

CI runs lint, format check, strict type-check, tests on Python 3.11 to 3.13, and a
Docker build on every push and pull request.

---

## Roadmap

**v0.1 — Foundation** *(current)*
- [x] Modular architecture with provider, personality, memory, and platform layers
- [x] mp3 personality
- [x] Decision engine, conversation focus, rate limiting
- [x] Typing indicators and human-like timing
- [x] Short-term memory with expiry; long-term memory framework with extraction
- [x] Conversation initiation in opted-in channels
- [x] Multiple bots per process
- [x] CLI (`run`, `check`, `chat`), Docker, CI

**v0.2 — Quality of conversation**
- [ ] Rolling conversation summaries to extend context beyond the history window
- [ ] Embedding-based memory recall (semantic search over long-term memory)
- [ ] Better addressee detection in busy group conversations
- [ ] Image and attachment understanding via multimodal models
- [ ] Per-guild personality tuning

**v0.3 — Operations**
- [ ] Slash commands: `/forget-me`, `/memories`, admin controls
- [ ] Metrics (latency, decisions, token usage) and health endpoint
- [ ] Additional providers (OpenAI-compatible APIs, llama.cpp, Anthropic)
- [ ] Provider fallback chains

**v0.4 — Multi-personality**
- [ ] `philosophy_bot` and `debate_bot` personalities
- [ ] Safe bot-to-bot conversations with loop protection
- [ ] Shared world knowledge between personalities

**Later**
- [ ] PostgreSQL backend for multi-instance deployments
- [ ] Sharding and horizontal scaling
- [ ] Web dashboard for personalities and memory inspection

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `Message Content Intent is not enabled` | Enable it under **Bot → Privileged Gateway Intents** in the Developer Portal. |
| `Discord rejected the token` | Wrong or reset token in `ARCANE_BOT_<ID>_TOKEN`. |
| Bot is online but never answers | The channel isn't in `ARCANE_ALLOWED_CHANNEL_IDS`, or the bot lacks *Send Messages* there. Run with `ARCANE_LOG_LEVEL=DEBUG` to see each decision. |
| `AI backend 'ollama' is unavailable` | Ollama isn't running or `ARCANE_OLLAMA_BASE_URL` is wrong. Bots stay silent until it's reachable. |
| `model '...' is not installed` | Run `ollama pull <model>`. |
| Replies are slow | Expected with large models on modest hardware; try a smaller model, or set `ARCANE_HUMANIZE=false` to remove simulated delays. |
| Output contains reasoning text | Use a non-reasoning model, or set `ARCANE_OLLAMA_THINK=false` for qwen3 / deepseek-r1. |

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
- mp3 talks like a person in the server, but it does not claim to be human. If
  someone sincerely asks whether it is an AI, it answers honestly.

All data stays on the machine running Arcane and your Ollama server.

---

## License

No license has been chosen yet. Until one is added, all rights are reserved by the
repository owner.
