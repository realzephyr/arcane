# Writing personalities

A personality is everything that makes one Arcane bot different from another:
who it is, what it cares about, how it writes, how it times its messages, when
it speaks up, how much it remembers, and which model it runs on. Personalities
are pure data. The engine never checks a personality's name, so adding one
never means changing framework code.

---

## Quick start

```
arcane/personalities/
└── debate_bot/
    ├── __init__.py          # can be empty
    └── personality.py       # defines PERSONALITY
```

```python
# arcane/personalities/debate_bot/personality.py
from arcane.personalities.base import BehaviorProfile, ModelProfile, Personality, StyleProfile

PERSONALITY = Personality(
    id="debate_bot",  # must match the folder name
    name="vex",  # what it's called in chat
    description="Sharp, contrarian debater who steelmans before attacking.",
    identity=(
        "You are vex, a regular in this Discord server who can't resist a good argument. "
        "You steelman the other side before you take it apart, and you concede gracefully "
        "when you lose."
    ),
    traits=("precise", "contrarian but fair", "concedes good points"),
    interests=("rhetoric and logic", "politics of the ancient world", "game theory"),
    interest_keywords={"debate", "argument", "fallacy", "logic", "game theory"},
    conversation_topics=("whether democracy is the least bad system",),
    style=StyleProfile(
        guidelines=(
            "Short, punchy messages. Plain text only.",
            "Name the strongest version of the other view before disagreeing.",
        ),
        max_messages_per_reply=2,
    ),
    behavior=BehaviorProfile(spontaneous_reply_chance=0.03),
    model=ModelProfile(model="qwen3:8b", temperature=0.7),
)
```

Run it next to mp3:

```dotenv
ARCANE_BOTS=mp3,debate_bot
ARCANE_BOT_DEBATE_BOT_TOKEN=...      # a second Discord application
```

Try it without Discord first:

```bash
python main.py chat -p debate_bot
```

---

## Field reference

Every profile has defaults; override only what makes the personality distinct.
All values are validated at startup, so typos and out-of-range values fail fast.

### Identity

| Field | Purpose |
|-------|---------|
| `id` | Lowercase identifier; must match the package folder. Used in env vars (`ARCANE_BOT_<ID>_TOKEN`) and to scope stored data. |
| `name` | The name it goes by. Messages containing it count as addressing the bot. |
| `aliases` | Other names that count as addressing it. |
| `description` | One line for logs and `python main.py check`. |
| `identity` | Second-person core description ("You are ..."). The most important text you'll write. |
| `traits` | Short behavioural traits, shown as a list in the prompt. |
| `interests` | What it cares about. Shapes what it says and which topics it brings up. |
| `interest_keywords` | Lowercase words or phrases. A message matching these may make it join a conversation uninvited (see `spontaneous_reply_chance`), and makes a recent message a candidate for a chime-in reply. |
| `conversation_topics` | Seeds for the messages it posts when it chimes in on its own. Topics rotate and avoid the last ten used. |
| `example_messages` | A handful of lines in its voice. Shown as a style reference, not to be copied. |

### `StyleProfile`: how it writes

| Field | Default | Purpose |
|-------|---------|---------|
| `guidelines` | `()` | Concrete writing rules for the prompt. |
| `max_response_chars` | 900 | Hard cap for a whole reply; cut at a sentence boundary. |
| `max_messages_per_reply` | 3 | How many Discord messages one reply may be split into. |
| `banned_openers` | `()` | Extra phrases stripped from the start of replies (added to the defaults, such as "Great question" and "Certainly"). |
| `allow_exclamation_points` | `True` | When `False`, every exclamation mark character (emoji, full-width and other Unicode forms included) is removed from replies; interrobangs become "?" and links are left alone. |
| `lowercase_starts` | `False` | When `True`, a capitalised first word of each message is lowercased ("Yeah" -> "yeah"); all-caps words are kept. |
| `blocked_patterns` | `()` | Case-insensitive regexes that must never be posted. A matching reply is regenerated; if every attempt matches, the bot stays silent. |

### `TimingProfile`: how it paces itself

The bot paces itself like a person: it notices new messages and reads them while
the model drafts the reply, with nothing shown in the channel. Only once the reply
is ready does the typing indicator appear, for `typing_start_seconds` plus the
reply's length at `typing_speed_wpm`, and then the message is sent. Short replies
are quick, long ones take longer: with the defaults, "nothing much" shows about 2
seconds of typing and a 107-character sentence about 7 seconds. Messages the same
person sends before typing starts are folded in: the draft is thrown away and
redone for everything they said (within 20 seconds, at most 3 times).

| Field | Default | Purpose |
|-------|---------|---------|
| `typing_speed_wpm` | 210 | How fast replies appear to be typed, in words per minute (a word is 5 characters, so 210 wpm = 17.5 characters per second). This is the speed people perceive in a chat window, not a keyboard test. |
| `typing_start_seconds` | 1.2 | The moment before the first keystroke, added to every message. |
| `reading_speed_wpm` | 500 | Reading speed for incoming messages (people skim chat). |
| `reaction_seconds` | (0.3, 1.0) | Time to notice new messages before reading them. |
| `follow_up_wait_seconds` | 2.0 | Extra wait after a message that is only an attention-getter ("yo", "mp3", "hey mp3 u there"), because the real message usually follows. Never applies to replies in an ongoing exchange. |
| `pause_between_messages_seconds` | (0.4, 1.2) | Pause between the parts of a split reply. |
| `min_typing_seconds` / `max_typing_seconds` | 0.8 / 15 | Bounds for typing a single message, so long replies never drag. |
| `max_reading_seconds` | 6 | Upper bound for the reading time of the messages being answered; follow-ups get their own reading time when they arrive. |
| `variation` | 0.15 | Spread of the log-normal noise applied to reading and typing times. |

### `BehaviorProfile`: when it talks

| Field | Default | Purpose |
|-------|---------|---------|
| `conversation_timeout_seconds` | 600 | Silence after which a conversation is over (overridable via `ARCANE_CONVERSATION_TIMEOUT_SECONDS`). While it lasts, the partner's plain messages (no reply, no mention) are answered. |
| `focus_timeout_seconds` | 120 | How long the current partner keeps priority after their last message. |
| `name_mention_reply_chance` | 0.85 | Chance to answer when its name is used without an @mention. The current conversation partner is always answered. |
| `spontaneous_reply_chance` | 0.05 | Chance to join a matching message in an idle channel. |
| `spontaneous_min_keyword_hits` / `spontaneous_min_words` | 1 / 6 | Minimum relevance and substance for joining uninvited. |
| `spontaneous_cooldown_seconds` | 900 | Minimum time since its last message before joining uninvited. |
| `opener_reply_window_seconds` | 180 | After chiming in, a plain message within this window counts as an answer: from the person it replied to; after a message of its own, from the only person active in the channel; in a busy channel, nobody's plain message counts, only replies to it, mentions and its name. |
| `respond_to_bots` | `False` | Whether other bots' messages can trigger it. Leave off unless you add loop protection. |
| `max_replies_per_channel_per_minute` / `max_replies_per_user_per_minute` | 15 / 12 | Anti-spam limits on replies actually sent; they apply even to @mentions. Generous enough for a fast one-on-one conversation. |
| `stale_trigger_seconds` | 180 | Queued messages older than this are dropped instead of answered late. |
| `initiative` | see below | Chiming in on its own. |

### `InitiativeProfile`: chiming in on its own

Whenever it isn't talking with anyone, the bot looks for the channel where people
talked most recently and either replies to a recent message worth discussing or
posts a message of its own about one of its `conversation_topics`. Deployments
choose where with `ARCANE_INITIATIVE_ENABLED` and `ARCANE_INITIATIVE_CHANNEL_IDS`
(or `ARCANE_BOT_<ID>_INITIATIVE_CHANNEL_IDS`), which restricts it to the listed
channels; when the list is empty, any channel the bot may talk in qualifies except
ones whose names contain a word suggesting unprompted chatter is unwelcome (`vent`,
`support`, `mod`, `log`, `rules` and similar; whole words, so `#logic` is fine).

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `True` | Master switch for this personality. |
| `check_interval_seconds` | 45 | How often it looks for a chance to chime in. |
| `idle_seconds` | 180 | Only chime in after not having talked with anyone for this long (and while no earlier chime-in is waiting for answers). |
| `active_window_seconds` | 600 | A channel counts as active when a person posted within this window. |
| `reply_max_age_seconds` | 300 | Only messages younger than this (and newer than its own last message) are replied to. |
| `channel_cooldown_minutes` | 15 | Minimum time between two chime-ins in the same channel. Doubles with every chime-in in a row that nobody answered, up to 8x. |
| `min_interval_minutes` | 8 | Minimum time between two chime-ins anywhere. |
| `max_per_channel_per_day` | 20 | Cap per channel within any 24 hours. |
| `chance` | 0.5 | Probability of acting on a check when every other condition is met. |
| `reply_chance` | 0.65 | When a recent message is worth answering, the chance of replying to it rather than posting a message of its own. |
| `active_hours_utc` | `None` | Optional `(start, end)` hour window in UTC; may wrap midnight. |

A message is worth replying to when it has a clear debate or philosophy hook ("hot
take", "would you rather", "morally", "free will", ...; `interest_keywords` only add
to the score), nobody has answered it yet, and it isn't a command, a bare link, aimed at someone else (a leading @mention or a Discord
reply to another person), someone venting, or a request the bot can't fulfil.

It never chimes in where its own message or another bot's message is the latest,
or where bots wrote more than a quarter of the last 20 messages. Every attempt is
logged before the model writes it, so cooldowns hold even when generation fails;
the attempt is dropped if someone needs a real answer meanwhile or the chat has
moved on.

### `MemoryProfile`: what it remembers

| Field | Default | Purpose |
|-------|---------|---------|
| `history_messages` | 16 | Recent channel messages in each prompt. The window may grow by half again before it advances, so the start of the prompt stays the same for several turns and Ollama can reuse its cache. |
| `history_char_budget` | 4000 | Character budget for history; oldest messages drop first. Lowered automatically when the context window, minus the reply, the system prompt and the note, can't hold that much. |
| `long_term_enabled` | `True` | Extract durable facts when conversations end. |
| `recall_limit` | 6 | Long-term memories about the user included per prompt. |
| `max_memories_per_user` | 40 | Cap per user; lowest-value memories are evicted. |
| `min_messages_for_extraction` | 3 | Skip extraction for very short conversations. |

### `ModelProfile`: which model

| Field | Default | Purpose |
|-------|---------|---------|
| `provider` | `None` | Provider name; `None` uses `ARCANE_AI_PROVIDER`. |
| `model` | `None` | Model name; `None` uses the provider default (`ARCANE_OLLAMA_MODEL`). `ARCANE_BOT_<ID>_MODEL` overrides both. |
| `temperature`, `top_p`, `top_k`, `repeat_penalty` | 0.85, 0.92, `None`, 1.1 | Sampling. |
| `max_tokens` | 200 | Generation cap per reply. Short caps keep local models fast. |
| `context_window` | 4096 | `num_ctx` for Ollama, used for every request (replies and memory extraction) because changing it between requests makes Ollama reload the model. Raise it if you raise the history budget. |

---

## Writing a persona that stays in character

* **Write the identity in second person** and give the persona a point of view:
  what it believes, what bores it, what it gets excited about. Vague personas
  collapse into a generic assistant voice.
* **Never mention bots, AI or code in a persona.** Identity, traits, guidelines
  and examples describe a person; anything about being a bot, a model, memory or
  context invites the model to talk about how it works, which gives it away. The
  framework already handles the rare "are you a bot" question with a per-turn note
  (laugh off jokes; if sincere, don't claim to be human and say it's a bot account
  in a few casual words), and replies that talk about the bot's code, prompt,
  model, context or training are regenerated.
* **Show, don't tell, with `example_messages`.** Four to six short lines in the
  persona's voice do more than a paragraph about tone.
* **Make style rules concrete.** "Keep it short" is weak; "default to one short
  line, under 15 words" works.
* **Write `conversation_topics` as debate seeds.** A question people can take sides
  on ("whether free will actually exists", "whether a hot dog is a sandwich") makes
  a far better chime-in than a bare subject ("music").
* **Keep `interest_keywords` specific.** Broad words like "why" or "think" make
  the bot barge into every conversation.
* **Use `blocked_patterns` as a backstop.** The ground rules keep replies family
  friendly; a few regexes for terms that must never be posted catch the rare slip,
  and a matching reply is regenerated or dropped.
* **Respect the framework's ground rules.** The prompt builder always adds rules
  that keep personas in character (they talk from their own point of view and never
  explain how they work, but don't claim to be a real person when seriously asked),
  keep them to what they can do from a text chat (they decline voice calls, games,
  pictures, meetups and socials with an excuse instead of agreeing), hold debates
  as text debates, keep them family friendly, block mass pings, and resist prompt
  injection. Don't write identities that contradict them.
* **Let the framework handle special requests.** `arcane/ai/guards.py` recognises
  common asks ("hop in vc", "send a pic", "whats your snap") and adds a direct
  instruction to decline for that turn; a reply that agrees anyway is regenerated
  with a correction. It also spots debate invitations ("debate me", "wanna debate")
  and adds a note to accept a text debate, or to offer one instead when they asked
  for vc. Persona example lines that show how the character says no help the model
  keep the refusal in voice.
* **Tune with the terminal chat.** `python main.py chat -p <id>` runs the full
  pipeline (prompt, model, post-processing) without Discord. Add `--humanize` to
  feel the timing, and set `ARCANE_LOG_LEVEL=DEBUG` to see every decision.
