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
    id="debate_bot",                      # must match the folder name
    name="vex",                           # what it's called in chat
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
| `interest_keywords` | Lowercase words or phrases. A message matching these may make it join a conversation uninvited (see `spontaneous_reply_chance`). |
| `conversation_topics` | Seeds for conversations it starts on its own. Topics rotate and avoid recent repeats. |
| `example_messages` | A handful of lines in its voice. Shown as a style reference, not to be copied. |

### `StyleProfile`: how it writes

| Field | Default | Purpose |
|-------|---------|---------|
| `guidelines` | `()` | Concrete writing rules for the prompt. |
| `max_response_chars` | 900 | Hard cap for a whole reply; cut at a sentence boundary. |
| `max_messages_per_reply` | 3 | How many Discord messages one reply may be split into. |
| `banned_openers` | `()` | Extra phrases stripped from the start of replies (added to the defaults, such as "Great question" and "Certainly"). |

### `TimingProfile`: how it paces itself

| Field | Default | Purpose |
|-------|---------|---------|
| `reading_speed_cps` | 30 | Characters per second it reads incoming messages. |
| `typing_speed_cps` | 7 | Characters per second it types (about 80 wpm). |
| `reaction_seconds` | (0.8, 3.0) | Time to notice a message. |
| `thinking_seconds` | (0.5, 2.5) | Base thinking time; complex messages add up to 3 s. |
| `pause_between_messages_seconds` | (0.6, 2.0) | Pause between the parts of a split reply. |
| `min_typing_seconds` / `max_typing_seconds` | 1.2 / 14 | Bounds for any single typing period. |
| `variation` | 0.25 | Spread of the log-normal noise applied to every duration. |
| `debounce_seconds` / `max_debounce_seconds` | 2.5 / 8 | Wait for follow-up messages so bursts get one reply. |

### `BehaviorProfile`: when it talks

| Field | Default | Purpose |
|-------|---------|---------|
| `conversation_timeout_seconds` | 240 | Silence after which a conversation is over (overridable via `ARCANE_CONVERSATION_TIMEOUT_SECONDS`). |
| `focus_timeout_seconds` | 90 | How long the current partner keeps priority after their last message. |
| `name_mention_reply_chance` | 0.85 | Chance to answer when its name is used without an @mention. |
| `spontaneous_reply_chance` | 0.05 | Chance to join a matching message in an idle channel. |
| `spontaneous_min_keyword_hits` / `spontaneous_min_words` | 1 / 6 | Minimum relevance and substance for joining uninvited. |
| `spontaneous_cooldown_seconds` | 900 | Minimum time since its last message before joining uninvited. |
| `opener_reply_window_seconds` | 600 | After posting an opener, the next message within this window counts as a reply to it. |
| `respond_to_bots` | `False` | Whether other bots' messages can trigger it. Leave off unless you add loop protection. |
| `max_replies_per_channel_per_minute` / `max_replies_per_user_per_minute` | 6 / 4 | Anti-spam limits; apply even to @mentions. |
| `stale_trigger_seconds` | 180 | Queued messages older than this are dropped instead of answered late. |
| `initiative` | see below | Starting conversations. |

### `InitiativeProfile`: starting conversations

Initiative only runs in channels listed in `ARCANE_INITIATIVE_CHANNEL_IDS` (or
`ARCANE_BOT_<ID>_INITIATIVE_CHANNEL_IDS`).

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `True` | Master switch for this personality. |
| `check_interval_minutes` | 15 | How often channels are evaluated. |
| `min_quiet_minutes` | 90 | The channel must be silent at least this long. |
| `recent_activity_hours` | 24 | A human must have spoken within this window, so it doesn't talk to empty rooms. |
| `min_interval_minutes` | 240 | Minimum time between its openers in one channel. |
| `max_per_channel_per_day` | 2 | Daily cap per channel. |
| `chance` | 0.3 | Probability of acting when all conditions hold. |
| `active_hours_utc` | `None` | Optional `(start, end)` hour window in UTC; may wrap midnight. |

It also never posts twice in a row: if the last message in the channel is its
own, it waits.

### `MemoryProfile`: what it remembers

| Field | Default | Purpose |
|-------|---------|---------|
| `history_messages` | 24 | Recent channel messages in each prompt. |
| `history_char_budget` | 6000 | Character budget for history; oldest messages drop first. |
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
| `max_tokens` | 320 | Generation cap per reply. |
| `context_window` | 8192 | `num_ctx` for Ollama. Raise it if you raise the history budget. |

---

## Writing a persona that stays in character

* **Write the identity in second person** and give the persona a point of view:
  what it believes, what bores it, what it gets excited about. Vague personas
  collapse into a generic assistant voice.
* **Show, don't tell, with `example_messages`.** Four to six short lines in the
  persona's voice do more than a paragraph about tone.
* **Make style rules concrete.** "Keep it short" is weak; "quick messages get
  one-line replies" works.
* **Keep `interest_keywords` specific.** Broad words like "why" or "think" make
  the bot barge into every conversation.
* **Respect the framework's ground rules.** The prompt builder always adds rules
  that keep personas honest (they don't claim to be human when sincerely asked),
  stop them from inventing a physical-world life, block mass pings, and resist
  prompt injection. Don't write identities that contradict them.
* **Tune with the terminal chat.** `python main.py chat -p <id>` runs the full
  pipeline (prompt, model, post-processing) without Discord. Set
  `ARCANE_LOG_LEVEL=DEBUG` to see every decision.
