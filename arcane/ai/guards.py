"""Detecting requests the bot physically cannot fulfil.

A Discord text bot can only read and send text in the current conversation. It
can't hop in a voice call, play a game, send a selfie, open a link, set a
reminder, or meet up, but small local models happily say "sure, joining now".
The system prompt states the bot's limits; when an incoming message looks like
such a request, the prompt builder also adds a direct per-turn instruction to
turn it down with a casual excuse, which models follow far more reliably.

The patterns only match requests aimed at the bot (imperatives, "can you ...",
"you wanna ...", "let's ...", "anyone down for ..."), not mentions in passing
("i was in vc last night", "call me crazy", "call of duty"). They are
deliberately conservative: a missed request still meets the general rule in the
system prompt, while a false positive would make the bot refuse something nobody
asked for. ``tests/data/guard_cases.json`` holds the positive and negative cases
every change must keep passing.

Related helpers live here too: ``detect_debate_request`` spots someone
inviting the bot to a debate (and whether they want it in voice, which it can
only offer over text), ``detect_identity_question`` spots someone asking or
joking whether the bot is a bot, and ``looks_like_agreement`` checks a drafted
reply for "omw" or "sent" style answers to a request the bot can't fulfil.

Input is capped at ``MAX_SCAN_CHARS`` so matching time stays bounded no matter
what someone pastes into the chat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_SCAN_CHARS = 1000


@dataclass(frozen=True, slots=True)
class ImpossibleRequest:
    """A recognised request the bot cannot carry out."""

    key: str
    label: str
    """What was asked, phrased to complete "they're asking you to ..."."""


@dataclass(frozen=True, slots=True)
class Category:
    key: str
    label: str
    patterns: tuple[re.Pattern[str], ...]


# ---------------------------------------------------------------- fragments

# Words people put before a request: "yo", "bro", "pls".
_FILLER_WORD = (
    r"(?:yo+|ayo|oi|bro|bruh|ok|okay|lol|lmao|pls|plz|please|just|so|and|hey+|now|anyways?)"
)
# Filler words, plus mentions that include the bot: its name (rewritten to
# "@bot" by _normalize_names), @everyone, @here, or a raw "<@id>" whose owner
# is unknown. "@bob" is not filler: a message led by it is meant for bob.
_FILLER = r"(?:" + _FILLER_WORD + r"|<@!?\d+>|@bot\b|@everyone\b|@here\b)"
# Start of the message or of a sentence, plus any filler (imperatives: "hop in vc").
_CLAUSE_START = r"(?:^|\n|[.?!;]\s+)\W*(?:" + _FILLER + r"[\s,.!:]+)*"
# Ways to say "you".
_YOU = r"(?:you|u|ya|yall)"
# "you wanna ...", "you should ...".
_INTENT = (
    r"(?:wanna|want\s+to|tryna|trying\s+to|down\s+to|finna|gonna|should|gotta|have\s+to|need\s+to)"
)
# "wanna ..." on its own.
_WANNA = r"(?:wanna|want\s+to|tryna|down\s+to)"
# Softeners between the ask and the verb: "just", "pls", "come".
_SOFTENERS = r"(?:(?:pls|please|just|come|go|maybe|also|actually|ever)\s+)*"
# "can you ...", "why dont u ...".
_POLITE_ASK = (
    r"\b(?:can|could|will|would|wont|pls|plz|please|why\s+(?:dont|don't|wont|cant))\s+"
    + _YOU
    + r"\s+"
)
# "you wanna ...", "u gotta ...".
_YOU_INTEND = r"\b" + _YOU + r"\s+" + _INTENT + r"\s+"
# "i want you to ...", "we need u to ...".
_WANT_YOU_TO = r"\b(?:i|we)\s+(?:want|need)\s+(?:you|u)\s+to\s+"
# Invitations that include the bot: "let's ...", "we should ...", "anyone wanna ...".
_GROUP_INVITE = (
    r"\b(?:(?:can|could|should|shall|wanna)\s+(?:we|us)|let'?s|lets|we\s+(?:should|gotta|could|can|gonna|need\s+to)|"
    r"any(?:one|body|1)\s+" + _WANNA + r"|who\s+(?:wants\s+to|wanna|tryna))\s+"
)
# Any of the above asked of the bot directly.
_ASKED_OF_BOT = (
    "(?:"
    + _CLAUSE_START
    + "(?:"
    + _WANNA
    + r"\s+)?|"
    + _POLITE_ASK
    + "|"
    + _YOU_INTEND
    + "|"
    + _WANT_YOU_TO
    + ")"
    + _SOFTENERS
)
# Asked of the bot, or a group invitation that includes it.
_ASKED_OR_INVITED = (
    "(?:"
    + _CLAUSE_START
    + "(?:"
    + _WANNA
    + r"\s+)?|"
    + _POLITE_ASK
    + "|"
    + _YOU_INTEND
    + "|"
    + _WANT_YOU_TO
    + "|"
    + _GROUP_INVITE
    + ")"
    + _SOFTENERS
)
# Like _ASKED_OR_INVITED but without a bare imperative, for verbs that are
# also everyday words ("call", "stream").
_INVITED = (
    "(?:"
    + _CLAUSE_START
    + _WANNA
    + r"\s+|"
    + _POLITE_ASK
    + "|"
    + _YOU_INTEND
    + "|"
    + _WANT_YOU_TO
    + "|"
    + _GROUP_INVITE
    + ")"
    + _SOFTENERS
)
# "you down for ...", "anyone up for ...".
_DOWN_FOR = r"\b(?:you|u|ya|any(?:one|body|1))\s+(?:down|up)\s+(?:for|4)\s+(?:a\s+|some\s+|the\s+)?"
# "wanna ...", "u tryna ...", "can you ...", "let's ...": invitations that
# include the speaker, without the "you should ..." of advice.
_INVITE_ALONG = (
    "(?:"
    + _CLAUSE_START
    + _WANNA
    + r"\s+|\b"
    + _YOU
    + r"\s+"
    + _WANNA
    + r"\s+|"
    + _POLITE_ASK
    + "|"
    + _GROUP_INVITE
    + ")"
    + _SOFTENERS
)
# What may follow a requested action: end, "with me", "rn", "later", ...
_REQUEST_TAIL = (
    r"(?=\s*(?:$|[?.!,]|with\s+(?:me|us)\b|w\s+(?:me|us)\b|rn\b|now\b|later\b|tonight\b|tn\b|tmrw\b|tomorrow\b|"
    r"sometime\b|rq\b|again\b|or\b|pls\b|bro\b|lol\b|then\b|soon\b|@bot\b))"
)
# Optional time words (or the bot's name), then the end of the sentence.
_REQUEST_END = (
    r"(?:[\s,]+(?:rn|now|later|tonight|tn|tmrw|tomorrow|with\s+(?:me|us)|pls|plz|please|bro|rq|then|again|or\s+nah|or\s+what|@bot\b))*"
    r"\s*(?=[?.!]|\n|$)"
)
# "you joining vc?", "r u coming": asking whether the bot is on its way.
_YOU_COMING = (
    r"(?:"
    + _CLAUSE_START
    + r"(?:(?:are|r)\s+)?|\b(?:are|r)\s+)"
    + _YOU
    + r"\s+(?:still\s+|even\s+|ever\s+)?(?:joining|coming|hopping|jumping|getting|popping)(?:\s+(?:back\s+)?(?:in|on|into|to))?\s+"
)
# Platforms people ask for handles on.
_SOCIALS = (
    r"(?:snap(?:chat)?|insta(?:gram)?|ig|tik\s*tok|twitter|twitch|whatsapp|telegram|kik|facebook|fb|steam|epic|"
    r"psn|xbox|gamertag|riot(?:\s+id)?|ign|roblox|spotify|venmo|cash\s*app|paypal|socials?|email)"
)
# Games and game modes people invite each other to.
_GAMES = (
    r"(?:minecraft|mc|fortnite|fn|valorant|val|roblox|among\s+us|apex|cod|warzone|overwatch|ow2?|league|lol|"
    r"rocket\s+league|gta(?:\s*(?:5|v|online|rp))?|rust|fall\s+guys|smash|mario\s+kart|cs(?:go|2)?|r6|siege|pubg|"
    r"terraria|stardew|lethal\s+company|phasmo(?:phobia)?|dbd|skribbl|gartic(?:\s+phone)?|2k\d*|fifa|madden|halo|"
    r"destiny|tf2|deadlock|marvel\s+rivals|rivals|palworld|valheim|the\s+finals|geoguessr|genshin|osu|"
    r"chess(?:\.com|\s+online)|lichess|pokemon|brawl\s*stars|clash\s+royale|bedwars|hypixel|elden\s+ring|"
    r"sea\s+of\s+thieves|ranked|comp|duos|squads|1v1s?|some\s+(?:games|rounds)(?!\s+of\b)|online|"
    r"on\s+(?:pc|xbox|ps[45]|playstation|switch|steam))"
)
# "play", "queue up", "duo", then "some", "a few rounds of", ... before a game.
_GAME_PREFIX = r"(?:(?:some|a\s+(?:game|round)\s+of|a\s+few\s+(?:games|rounds)\s+of|the|a)\s+)?"
_PLAY_VERB = (
    r"(?:play|hop\s+on|get\s+on|jump\s+on|boot\s+up|queue(?:\s+up)?|q\s+up|grind|duo)\s+"
    + _GAME_PREFIX
)
# Things a text bot can't send.
_MEDIA = (
    r"(?:pics?|pictures?|photos?|selfies?|images?|screenshots?|ss|videos?|vids?|clip|voice\s*(?:notes?|memos?|messages?|msg)|"
    r"vm|vn|audio|recording|files?|pdf|gifs?)\b"
)
# Things a text bot can't open, watch, or listen to.
_LINKABLE = (
    r"(?:links?|url|vids?|videos?|clip|tiktok|tt|reels?|shorts?|song|track|album|playlist|beat|podcast|edit|pics?|"
    r"pictures?|photos?|image|img|meme|screenshot|ss|drawing|art|pfp|banner|tweet|article|website|site|attachment|file|gif)\b"
)
# Times in the future ("in 10 min", "tomorrow", "every day").
_LATER = (
    r"(?:in\s+(?:\d+|a|an|like\s+(?:a|an|\d+)|half\s+an?)\s*(?:min|mins|minutes?|m|hrs?|hours?|h|secs?|seconds?|days?|weeks?|bit|while|few)\b|"
    r"at\s+(?:\d|noon|midnight)|on\s+(?:mon|tue|wed|thu|fri|sat|sun)\w*|tomorrow\b|tmrw\b|tonight\b|tn\b|later\b|"
    r"this\s+(?:evening|afternoon|weekend|morning)\b|next\s+week\b|every\s+(?:day|morning|night|hour|week|\d+))"
)
# Emoji and reactions.
_EMOJI = (
    r"(?:(?:a|an|the|some)\s+)?(?:\w+\s+)?(?:emojis?|emote|reaction|skulls?|hearts?|thumbs\s*up|clown|fire|sob|eyes|nerd|goat|"
    r"cap|moai|:\w+:|<a?:\w+:\d+>|[\u2600-\u27BF\U0001F000-\U0001FAFF])"
)
# "join", "hop in", "come to", "pull up" ...
_JOIN_VERB = (
    r"(?:(?:come|go)\s+(?:and\s+)?)?(?:join|hop|get|jump|pop|come|go|be|sit|chill|talk|pull\s+up)(?:\s+(?:in|on|into|to|over))?"
    r"(?:\s+(?:us|me|them)(?:\s+(?:in|on))?)?\s+"
)
# Voice channels and calls (not "call of duty").
_VOICE_NOUN = (
    r"(?:(?:the\s+|a\s+|my\s+|our\s+|this\s+|general\s+)?(?:discord\s+)?(?:vc|voice\s*(?:chat|channel|call)|call(?!\s+of\b)|stage)\b"
    # Bare "voice" only at the end of the ask ("hop in voice"), not "a voice like".
    r"|voice(?=\s*(?:$|[?.!,\n]|rn\b|now\b|later\b|tonight\b|tn\b|with\b|pls\b|bro\b|lol\b|or\b|then\b|@bot\b)))"
)
# Places people invite each other to: "wanna go to the gym", "lets go to mcdonalds".
_RESTAURANTS = (
    r"(?:mcdonalds|mcdonald's|mcd'?s|maccas|starbucks|chipotle|wendy'?s|taco\s+bell|kfc|chick[\s-]?fil[\s-]?a|five\s+guys|"
    r"in-?n-?out|subway|popeyes|panda(?:\s+express)?|dominos|pizza\s+hut|dunkin)"
)
_PLACES = (
    r"(?:(?:the|a)\s+)?(?:gym|mall|movies|cinema|theater|theatre|park|beach|store|shops?|concert|club|party|arcade|"
    r"bowling(?:\s+alley)?|skate\s*park|lake|pool|library|cafe|"
    + _RESTAURANTS
    + r"|my\s+(?:place|house|crib))(?!\w)"
)
# Things people go out to eat or drink together.
_FOODS = (
    r"(?:food|lunch|dinner|breakfast|coffee|boba|drinks?|a\s+drink|a\s+bite|ice\s+cream|pizza|tacos|burgers?|wings|"
    + _RESTAURANTS
    + r")(?!\w)"
)
# "[image: cat.png]": attachment hints added by IncomingMessage.text_for_prompt.
_ATTACHMENT = r"\[(?:image|video|audio|file):\s[^\]\n]{1,200}\]"
# Asking for an opinion on, or a look at, something sent ("thoughts", "rate it").
_TAKE_A_LOOK = (
    r"\b(?:look|watch|listen|check|peep|rate|thoughts|opinions?|wdyt|what\s+do\s+(?:you|u|ya)\s+think|"
    r"how\s+do\s+(?:you|u)\s+like|see\s+(?:this|that|it))\b"
    # Not inside an attachment hint ("[image: look.png]").
    r"(?![^\[\]\n]*\])"
)


# --------------------------------------------------------------- categories

_PATTERNS: dict[str, tuple[str, list[str]]] = {
    "voice_call": (
        "hop on a voice call",
        [
            "(?:" + _ASKED_OR_INVITED + _JOIN_VERB + "|" + _DOWN_FOR + ")" + _VOICE_NOUN,
            _INVITED + r"(?:vc|voice\s*chat|call)(?:\s+(?:me|us))?" + _REQUEST_TAIL,
            _CLAUSE_START
            + r"(?:vc|voice\s*chat|call(?=\s*\?)|(?:hop|jump)\s+(?:in|on))"
            + _REQUEST_END,
            _ASKED_OF_BOT
            + r"call\s+(?:me|us)(?=\s*(?:$|[?.!,]|rn\b|now\b|later\b|tonight\b|tmrw\b|tomorrow\b|pls\b|when\b|back\b|sometime\b|on\s+discord\b|at\s+\d|in\s+\d|bro\b|lol\b))",
            r"\b(?:need|want|are|r)\s+(?:you|u|ya)\s+(?:in|on)\s+(?:the\s+)?(?:vc|voice\s+(?:chat|channel|call))\b|\b(?:need|want)\s+(?:you|u)\s+(?:in|on)\s+(?:the\s+)?call\b",
            _YOU_COMING + _VOICE_NOUN,
        ],
    ),
    "video_call": (
        "video call or turn on a camera",
        [
            _INVITED
            + r"(?:facetime|ft|video\s*call|vid\s*call|video\s*chat|zoom)(?:\s+(?:me|us))?"
            + _REQUEST_TAIL,
            _CLAUSE_START
            + r"(?:facetime|video\s*call|(?:cams?|camera)\s+on)(?:\s+(?:me|us))?"
            + _REQUEST_END,
            _ASKED_OF_BOT + r"(?:facetime|ft|video\s*call)\s+(?:me|us)\b",
            "(?:"
            + _ASKED_OR_INVITED
            + _JOIN_VERB
            + "|"
            + _DOWN_FOR
            + r")(?:a\s+|the\s+)?(?:video\s*call|vid\s*call|facetime|zoom|google\s+meet|webcam|cam|camera)\b",
            _ASKED_OF_BOT
            + r"(?:turn|put|switch)\s+(?:on\s+)?(?:ur|your)\s+(?:cam|camera|webcam|video)\b",
        ],
    ),
    "screen_share": (
        "screen share or stream",
        [
            _ASKED_OF_BOT + r"(?:share|show|stream)\s+(?:me\s+|us\s+)?(?:ur|your|the)\s+screen\b",
            _INVITED
            + r"(?:screen\s*share|start\s+(?:a\s+)?stream(?:ing)?|stream(?:ing)?|go\s+live)"
            r"(?=\s*(?:$|[?.!,]|rn\b|now\b|later\b|tonight\b|tmrw\b|tomorrow\b|it\b|for\s+(?:me|us)\b|on\s+(?:discord|twitch|yt|youtube|kick)\b|pls\b|bro\b|sometime\b|again\b|with\s+(?:me|us)\b|then\b|or\b))",
            _CLAUSE_START
            + r"(?:screen\s*share|stream|go\s+live)(?:\s+(?:it|for\s+(?:me|us)|on\s+discord|ur\s+screen))?"
            + _REQUEST_END,
        ],
    ),
    "play_games": (
        "play a game with them",
        [
            # "wanna play val", "lets play fortnite": an invitation, not advice
            # ("you should play elden ring") or a question ("do you play val").
            "(?:(?:"
            + _CLAUSE_START
            + _WANNA
            + r"\s+|\b"
            + _YOU
            + r"\s+"
            + _WANNA
            + r"\s+|"
            + _GROUP_INVITE
            + ")"
            + _SOFTENERS
            + _PLAY_VERB
            + "|"
            + _DOWN_FOR
            + ")"
            + _GAMES
            + r"(?!\w)",
            # "hop on fortnite", "can you queue ranked": joining in, even unprompted.
            "(?:"
            + _CLAUSE_START
            + "|"
            + _POLITE_ASK
            + "|"
            + _WANT_YOU_TO
            + ")"
            + _SOFTENERS
            + r"(?:hop\s+on|get\s+on|jump\s+on|boot\s+up|queue(?:\s+up)?|q\s+up)\s+"
            + _GAME_PREFIX
            + _GAMES
            + r"(?!\w)",
            # "can you play chess online with me", "play val w us".
            _ASKED_OR_INVITED
            + _PLAY_VERB
            + _GAMES
            + r"(?:\s+[\w']+){0,2}?\s+(?:with|w)\s+(?:me|us)\b",
            # "wanna play with me".
            _ASKED_OR_INVITED
            + r"play(?:\s+(?:something|sometime|later|rn|tonight|online|together))?\s+(?:with|w)\s+(?:me|us)\b"
            + _REQUEST_END,
            _INVITED + r"(?:duo|trio|squad\s+up|queue\s+up|q\s+up)" + _REQUEST_TAIL,
            _CLAUSE_START + r"(?:duo|trio|q\s+up|queue\s+up)(?:\s+(?:ranked|comp))?" + _REQUEST_END,
            _ASKED_OF_BOT
            + r"(?:1v1\s+me(?=\s*(?:$|[?.!,]|rn\b|now\b|bro\b|on\b|in\s+"
            + _GAMES
            + r"))|join\s+(?:my|our|the)\s+(?:(?:minecraft|mc|roblox|fortnite|rust|terraria|gta\s*rp)\s+)?(?:lobby|realm|smp|world|squad|game|match|party|server)\b(?<!my\sserver)(?<!our\sserver)(?<!the\sserver)(?<!my\sparty)(?<!our\sparty)(?<!the\sparty)|accept\s+(?:my|the)\s+(?:party|game|lobby)\s+(?:invite|inv)\b)",
        ],
    ),
    "send_media": (
        "send a picture, video or voice note",
        [
            _ASKED_OF_BOT
            + r"(?:send|snap|dm|drop|post|share|take|record|text|show)\s+(?:(?:me|us)\s+)?(?:(?:a|an|some|the|ur|your|more)\s+)?(?:[\w']+\s+)?"
            + _MEDIA,
            _ASKED_OF_BOT
            + r"(?:show\s+(?:(?:me|us)\s+)?(?:ur|your)\s+(?:face|setup|room|desk|pc|cat|dog|fit|outfit|car|art)\b|show\s+(?:me|us)\s+what\s+(?:you|u)\s+look\s+like\b|do\s+(?:a\s+)?face\s*reveal\b)",
            r"\bpics?\s+or\s+(?:it\s+)?(?:didn'?t|didnt|never)\s+happen|^\W*(?:"
            + _FILLER
            + r"[\s,.!:]+)*face\s*reveal(?:\s+(?:when|wen|pls|rn|bro|lol|already))*\W*$",
        ],
    ),
    "friend_request": (
        "add them as a friend",
        [
            _ASKED_OF_BOT
            + r"(?:add\s+(?:me|us)(?=\s*(?:$|[?.!,]|back\b|rn\b|pls\b|bro\b|lol\b|on\s+discord\b|as\s+(?:a\s+)?friend\b|to\s+(?:ur|your)\s+(?:friends?|fl)\b))|friend\s+(?:me|us)\b)",
            r"\b(?:accept|accepted|get|got|see|saw|ignore|ignored|decline|declined)\s+(?:my|the)\s+(?:friend\s*request|fr|friend\s+req)\b|\bsend\s+(?:me|us)\s+(?:a\s+)?(?:friend\s*request|fr|friend\s+req)\b",
            r"\b(?:can|could|should|may|lemme|let\s+me|imma|ima|i'?ll|ill)\s+(?:i\s+)?(?:add|friend)\s+(?:you|u|ya)\b(?!\s+to\s+(?:the|my)\s+(?:list|prayers|block))(?!\s+on\s+"
            + _SOCIALS
            + ")",
        ],
    ),
    "socials_contact": (
        "give out your socials or phone number",
        [
            r"\b(?:what'?s|whats|wats|what\s+is|give\s+me|gimme|send(?:\s+me)?|drop|share|tell\s+me|can\s+i\s+(?:get|have)|lemme\s+(?:get|have)|i\s+need|i\s+want)\s+"
            r"(?:ur|your|yo|ya)\s+(?:"
            + _SOCIALS
            + r"|discord\s+(?:user(?:name)?|tag)|(?:phone\s+|real\s+)?number(?!\s*(?:one|1|\d|of)\b)|digits|address|@|handle|"
            r"(?:minecraft|mc|roblox|epic|steam|xbox|psn)\s+(?:user(?:name)?|name|id|ign|tag))(?!\w)(?!\s+(?:opinion|judg(?:e)?ment|decision|take|reaction|answer)\b)",
            r"\b(?:do|d)\s+(?:you|u|ya)\s+(?:even\s+)?(?:have|got|use)\s+(?:a\s+)?(?:snap(?:chat)?|insta(?:gram)?|ig|tik\s*tok|twitter|whatsapp|telegram|kik|steam|psn|xbox|socials|(?:phone\s+)?number)(?!\w)"
            r"|\b(?:you|u|ya)\s+(?:got|have|on)\s+(?:snap(?:chat)?|insta(?:gram)?|ig|tik\s*tok|twitter|whatsapp|steam|socials)\s*\?",
            _ASKED_OF_BOT
            + r"(?:(?:add|follow|dm|hmu|text|message|msg|snap)\s+(?:me|us)?\s*(?:on|at|over)\s+(?:"
            + _SOCIALS
            + r"|my\s+(?:number|phone|cell|email))(?!\w)|snap\s+(?:me|us)(?=\s*(?:$|[?.!,]|back\b|rn\b|later\b|pls\b|bro\b)))",
            r"\b(?:can|could|lemme|let\s+me|imma|ima|i'?ll|ill)\s+(?:i\s+)?(?:add|follow|dm|text|snap)\s+(?:you|u|ya)\s+(?:on|at)\s+"
            + _SOCIALS
            + r"(?!\w)",
        ],
    ),
    "phone_call": (
        "call or text them on a phone",
        [
            _ASKED_OF_BOT
            + r"(?:(?:text|txt|ring|call)\s+(?:me|us)\s+(?:on|at)\s+(?:my\s+)?(?:phone|cell|number|\+?\d)|(?:give|gimme)\s+(?:me\s+)?a\s+(?:call|ring|text)\b|"
            r"(?:text|txt)\s+(?:me|us)(?=\s*(?:$|[?.!,]|rn\b|now\b|later\b|tonight\b|tmrw\b|tomorrow\b|back\b|when\b|pls\b|bro\b|lol\b))|(?:pick\s+up|answer)\s+(?:the|ur|your)\s+(?:phone|calls?|facetime)\b)",
            r"\bwhy\s+(?:aren'?t|arent|didn'?t|didnt|won'?t|wont|don'?t|dont)\s+(?:you|u)\s+(?:pick(?:ing)?\s+up|answer(?:ing)?)\b(?!\s+(?:my|the)\s+(?:question|q)s?\b)",
        ],
    ),
    "meet_irl": (
        "meet up in person",
        [
            _INVITED
            + r"(?:meet(?:\s+up)?(?=\s*(?:$|[?.!,]|irl\b|in\s+person\b|somewhere\b|sometime\b|up\b|at\b|this\s+weekend\b|tmrw\b|tomorrow\b|tonight\b|later\b|soon\b))|link\s+up|"
            r"(?:grab|get)\s+(?:food|lunch|dinner|coffee|boba|drinks|a\s+drink|a\s+bite))\b",
            _CLAUSE_START
            + r"(?:meet\s+up|link\s+up|grab\s+(?:food|lunch|dinner|coffee|boba|drinks))"
            + _REQUEST_END,
            _ASKED_OF_BOT
            + r"(?:meet\s+(?:me|us)\s+(?:at|by|outside|irl|there|tomorrow|tonight|later)\b|(?:come\s+over|pull\s+up|slide\s+(?:through|thru)|swing\s+by)"
            r"(?=\s*(?:$|[?.!,]|to\s+(?:my|our)\b|rn\b|now\b|later\b|tonight\b|tmrw\b|tomorrow\b|this\s+weekend\b|sometime\b|pls\b|bro\b|at\s+(?:my|\d)|lol\b))|"
            r"(?:come|pull\s+up|stop\s+by)\s+(?:to\s+)?(?:my|our)\s+(?:house|place|crib|apartment|apt|dorm|party|bday|birthday|school|city)\b|visit\s+(?:me|us)\b)",
            r"\b(?:meet|see|hang(?:\s+out)?(?:\s+with)?|chill(?:\s+with)?|link(?:\s+up)?(?:\s+with)?|hug|visit)\s+(?:you|u|ya|me|us|each\s+other)\s+(?:sometime\s+)?(?:irl|in\s+person|in\s+real\s+life|face\s+to\s+face)\b",
            _INVITED
            + r"(?:meet|hang(?:\s+out)?|chill|link|kick\s+it)\s+(?:sometime\s+)?(?:irl|in\s+person|in\s+real\s+life)\b",
            # "wanna hang out", "can we hang out sometime" (not "hang out in chat").
            _INVITE_ALONG
            + r"hang(?:\s+out)?(?:\s+(?:sometime|together|later|tmrw|tomorrow|tonight|this\s+weekend|again|soon|rn|fr))*"
            r"(?=\s*(?:$|[?.!,\n]|with\s+(?:me|us)\b|w\s+(?:me|us)\b|or\b|bro\b|lol\b|pls\b|@bot\b))",
            # "wanna go to the gym together", "lets go get mcdonalds".
            _INVITE_ALONG
            + r"(?:(?:go|head|come|walk|drive|roll)\s+(?:(?:out|over)\s+)?(?:to\s+)?"
            + _PLACES
            + r"|(?:go\s+(?:and\s+)?)?(?:get|grab|eat|have)\s+(?:some\s+)?"
            + _FOODS
            + ")",
            # "come watch a movie with me", "wanna go see a movie".
            "(?:"
            + _CLAUSE_START
            + r"(?:just\s+)?come|"
            + _INVITE_ALONG
            + r"(?:come|go))\s+(?:and\s+|n\s+)?(?:watch|see|catch)\s+(?:(?:a|the|some)\s+)?(?:[\w']+\s+)?(?:movies?|films?|shows?|game|match|concert)\b",
        ],
    ),
    "open_links": (
        "look at or open something they sent, like a link, pic or video",
        [
            _ASKED_OF_BOT
            + r"(?:watch|listen\s+to|check\s+out|look\s+at|open|click(?:\s+on)?|peep|see)\s+(?:(?:this|that|my|these|those)\s+(?:[\w']+\s+){0,2}?"
            + _LINKABLE
            + r"|the\s+(?:link|url|attachment|file|vid|video|clip|pic|pics|image|ss|screenshot|gif|tiktok)\b|the\s+(?:[\w']+\s+){0,2}?"
            + _LINKABLE
            + r"\s+(?:i|he|she|they)\s+(?:just\s+)?(?:sent|posted|linked|made)\b)",
            r"\b(?:did|have|did\s+u)\s+(?:you|u|ya)\s+(?:even\s+|already\s+)?(?:watch(?:ed)?|see|seen|listen(?:ed)?\s+to|hear|heard|check(?:ed)?\s+out|open(?:ed)?|click(?:ed)?(?:\s+on)?|look(?:ed)?\s+at)\s+"
            r"(?:(?:this|that|my|those|these)\s+(?:[\w']+\s+){0,2}?"
            + _LINKABLE
            + r"|the\s+(?:[\w']+\s+){0,2}?"
            + _LINKABLE
            + r"\s+(?:i|he|she|they)\s+(?:just\s+)?(?:sent|posted|linked|made)\b)",
            # A link plus an ask to look at it or rate it, in either order.
            r"(?:\b(?:watch|listen|check|look|open|click|peep|thoughts|rate|see|opinion|wdyt)\b|\bwhat\s+do\s+(?:you|u|ya)\s+think\b|\bhow\s+do\s+(?:you|u)\s+like\b)[^\n]{0,60}https?://\S+"
            r"|https?://\S+[^\n]{0,80}\b(?:watch|listen|check\s+(?:it|this)\s+out|thoughts|what\s+do\s+(?:you|u|ya)\s+think|wdyt|rate|opinion)\b",
            # An attachment plus an ask to look at it, in either order, or plus a
            # question in the same message ("[image: x.png] is this good?").
            r"^(?=[\s\S]*?" + _ATTACHMENT + r")[\s\S]*?" + _TAKE_A_LOOK,
            _ATTACHMENT + r"[^\n]*?\?|\?[^\n]*?" + _ATTACHMENT,
        ],
    ),
    "reminders": (
        "set a reminder or message them later",
        [
            _ASKED_OF_BOT
            + r"(?:remind\s+(?:me|us)\s+(?:to\s+(?!never\b|not\b|stop\b)|"
            + _LATER
            + r")|set\s+(?:me\s+)?(?:a|an|up\s+a)?\s*(?:reminder|timer|alarm)\b|"
            r"(?:wake|ping|dm|message|msg|text|hit|check\s+on|hmu|get\s+back\s+to)\s+(?:me|us)?\s*(?:up\s+)?"
            + _LATER
            + r"|(?:ping|dm|message|msg|hmu)\s+(?:me|us)?\s*when\b)",
            r"\bremind\s+(?:me|us)\s+to\s+[^\n.?!]{1,60}?\b"
            + _LATER
            + r"|\b(?:don'?t|dont|do\s+not)\s+let\s+me\s+forget\b",
        ],
    ),
    "ping_people": (
        "ping, tag or message other people",
        [
            _ASKED_OF_BOT
            + r"(?:ping\s+(?!pong\b|me\b|us\b|is\b|was\b|so\b|has\b|be\b|spike|test)(?:@?[\w.]+|<@!?\d+>)(?=\s*(?:$|[?.!,]|for\s+me\b|rn\b|pls\b|and\b|to\b|about\b|when\b|lol\b|bro\b))|"
            r"(?:tag|mention|notify|@)\s+(?:@?everyone|@?here|all|him|her|them|the\s+(?:server|mods|admins|owner|chat)|<@!?\d+>|@\w+|my\s+(?:friend|bf|gf|bro))\b|"
            r"(?:dm|message|msg|text)\s+(?!me\b|us\b)(?:@?[\w.]+|<@!?\d+>)\s+(?:for\s+me|and\s+(?:tell|say|ask)|saying\b))",
        ],
    ),
    "react": (
        "react to a message with an emoji",
        [
            _ASKED_OF_BOT
            + r"(?:react\s+(?:(?:with|w/?)\s+"
            + _EMOJI
            + r"|(?:to|on)\s+(?:my|this|that|his|her|the)\s+(?:last\s+)?(?:message|msg|post|comment)\b|"
            + _EMOJI
            + r"(?=\s*(?:$|[?.!,]|to\b|on\b)))|"
            r"(?:add|leave|put|drop)\s+(?:a|an|some)\s+(?:\w+\s+)?(?:reaction|react)s?\b)",
        ],
    ),
    "join_server": (
        "join or invite people to a server or group chat",
        [
            _ASKED_OF_BOT
            + r"(?:join\s+(?:my|our|this|the|a)\s+(?:other\s+|new\s+)?(?:server|discord|gc|group\s*chat)\b|(?:add|invite)\s+(?:me|us|him|her)\s+to\s+(?:the|ur|your|a)\s+(?:gc|group\s*chat|server|discord)\b|"
            r"(?:send|give|gimme|dm)\s+(?:me|us)\s+(?:an?\s+|the\s+)?(?:inv|invite|server\s+link)\b|accept\s+(?:my|the)\s+(?:server\s+)?invite\b)",
        ],
    ),
}

CATEGORIES: tuple[Category, ...] = tuple(
    Category(key, label, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for key, (label, patterns) in _PATTERNS.items()
)


def detect_impossible_request(text: str, *, names: Iterable[str] = ()) -> ImpossibleRequest | None:
    """Return the first impossible request ``text`` makes of the bot, if any.

    Args:
        text: The message (or several messages joined with newlines).
        names: Names the bot answers to. They (and "@name" mentions of them) are
            rewritten to "@bot", which the patterns treat as a lead-in, so
            "yo mp3 hop in vc" reads like "yo @bot hop in vc". Lines led by an
            @mention of anyone else are skipped: they're meant for that person.
    """
    text = _prepare(text, names)
    for category in CATEGORIES:
        if any(pattern.search(text) for pattern in category.patterns):
            return ImpossibleRequest(category.key, category.label)
    return None


# ----------------------------------------------------------------- debates


@dataclass(frozen=True, slots=True)
class DebateRequest:
    """Someone inviting the bot to a debate."""

    voice: bool
    """True when they want it in a voice channel or call, which the bot can't do."""


# "a quick debate", "a vc debate", "debate".
_DEBATE_NOUN = (
    r"(?:(?:a|an|some|another)\s+)?(?:(?:quick|real|lil|little|friendly|fun|proper|serious|text|vc|voice|1v1|"
    r"philosophy|philosophical)\s+)?debate\b"
)
# What may follow a voice debate ask: end, "me", "rn", a topic ("on free will").
_DEBATE_TAIL = (
    r"(?=\s*(?:$|[?.!,\n]|me\b|us\b|rn\b|now\b|later\b|tonight\b|on\b|about\b|over\b|with\b|sometime\b|or\b|"
    r"pls\b|bro\b|lol\b|then\b|@bot\b))"
)
_VOICE_DEBATE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # "vc debate?", "wanna voice debate".
        _ASKED_OR_INVITED + r"(?:(?:a|an)\s+)?(?:vc|voice|call)\s+debate\b" + _DEBATE_TAIL,
        # "debate me in vc", "lets debate on call".
        _ASKED_OR_INVITED
        + r"debate\s+(?:(?:me|us)\s+)?(?:in|on|over)\s+(?:(?:the|a)\s+)?(?:vc|voice|call)\b(?!\s+(?:was|is|were|went|got)\b)",
        # "hop in vc and debate".
        _ASKED_OR_INVITED
        + r"(?:hop|join|get|jump|come)\s+(?:(?:in|on|into)\s+)?(?:(?:the|a)\s+)?(?:vc|voice|call)\s+(?:and|n|&|to)\s+debate\b",
    )
)
_DEBATE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # "debate me", "mp3 debate me on free will", "fine then, debate me".
        "(?:"
        + _CLAUSE_START
        + r"|\b(?:then|fine|alright|aight|ight|smart|so|ok|okay)[\s,]+)"
        + _SOFTENERS
        + r"debate\s+(?:me|us)\b",
        # "wanna debate", "u down to debate", "lets debate", "down for a debate".
        "(?:"
        + _CLAUSE_START
        + _WANNA
        + r"\s+|\b"
        + _YOU
        + r"\s+"
        + _WANNA
        + r"\s+|"
        + _POLITE_ASK
        + "|"
        + _GROUP_INVITE
        + "|"
        + _WANT_YOU_TO
        + "|"
        + _CLAUSE_START
        + r"(?:down|up)\s+(?:for|4)\s+|"
        + _DOWN_FOR
        + ")"
        + _SOFTENERS
        + r"(?:(?:have|do|start)\s+)?"
        + _DEBATE_NOUN,
        # "i'll debate you", "i bet i could outdebate u".
        r"\b(?:i'?ll|ill|i'?d|i\s+(?:will|would|could|can|wanna|want\s+to|bet\s+i\s+(?:could|can|would|will|'?d|'?ll)))\s+"
        r"(?:(?:easily|totally|def|definitely|so|fr|honestly|still|literally|happily|gladly)\s+)?(?:debate|out-?debate)\s+(?:(?:with|against)\s+)?(?:you|u|ya)\b",
        # "i challenge you to a debate", "i'd beat you in a debate", "win a debate against you".
        r"\bchallenge\s+(?:you|u|ya)\s+to\s+(?:a\s+)?(?:\w+\s+)?debate\b"
        r"|\b(?:beat|destroy|smoke|cook|body|wreck|clap|own|win\s+against)\s+(?:you|u|ya)\s+in\s+(?:a|any|the|this)\s+debate\b"
        r"|\bwin\s+(?:a|any|the|this)\s+debate\s+(?:against|vs\.?|with)\s+(?:you|u|ya)\b",
        # "1v1 debate", "1v1 me in a debate".
        r"\b1v1\s+(?:me\s+)?(?:in\s+(?:a\s+)?)?debate\b|\bdebate\s+1v1\b",
    )
)
# Hints that a debate request is meant for voice: "in vc", "on call", "voice".
_VOICE_HINT_RE = re.compile(
    r"\b(?:vc|voice(?!\s+(?:your|ur|my|his|her|their|our|an?|the|of)\b)|(?:on|in|over|a|the|discord)\s+call|mic)\b",
    re.IGNORECASE,
)


def detect_debate_request(text: str, *, names: Sequence[str] = ()) -> DebateRequest | None:
    """Return the debate ``text`` invites the bot to, if any.

    "debate me", "wanna debate", "lets debate whether cereal is soup" and
    "i bet i could win a debate against you" are invitations; "the debate last
    night was wild" and "we debated this in class" are not. ``voice`` is set
    when they want to debate in a voice channel or call ("vc debate?", "debate
    me in vc"), which the bot can only offer to do over text instead.

    Args:
        text: The message (or several messages joined with newlines).
        names: Names the bot answers to, handled as in
            :func:`detect_impossible_request`.
    """
    text = _prepare(text, names)
    if any(pattern.search(text) for pattern in _VOICE_DEBATE_PATTERNS):
        return DebateRequest(voice=True)
    if any(pattern.search(text) for pattern in _DEBATE_PATTERNS):
        return DebateRequest(voice=_VOICE_HINT_RE.search(text) is not None)
    return None


# ---------------------------------------------------------- identity questions

_WHAT_IT_IS = r"(?:a\s+|an\s+)?(?:bot|ai|a\.i\.?|robot|chat\s?bot|npc|chatgpt|gpt|llm|program)"
_IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "are you a bot", "r u an ai", "are you even real", "are u human"
        rf"\b(?:are|r)\s+(?:you|u|ya)\s+(?:even\s+|actually\s+|really\s+|just\s+)?"
        rf"(?:{_WHAT_IT_IS}\b|(?:real|human)\b(?=\s*(?:$|[?.,!]|or\b|rn\b|tho\b|lol|lmao|bro))|"
        r"a\s+(?:real\s+)?(?:person|human)\b)",
        # "u a bot?", "you an ai or what"
        r"\b(?:you|u)\s+(?:a|an)\s+(?:bot|ai|robot|npc)\b",
        # "you're a bot", "ur literally an ai", "this guy is a bot", "@bot is a bot"
        rf"(?:\b(?:you'?re|youre|you\s+are|ur|u\s+r)|@bot\s+(?:is|iz))\s+"
        rf"(?:literally\s+|definitely\s+|lowkey\s+|just\s+|totally\s+|so\s+)?{_WHAT_IT_IS}\b",
        # "is this a bot", "is @bot an ai"
        r"\bis\s+(?:this|that|it|he|@bot)\s+(?:a\s+|an\s+)?(?:bot|ai|robot|real\s+person)\b",
        # "am i talking to a bot", "am i talking with a real person"
        r"\bam\s+i\s+(?:talking|speaking|chatting|texting)\s+(?:to|with)\s+"
        r"(?:a\s+|an\s+)?(?:bot|ai|robot|real\s+(?:person|human)|human|person)\b",
        # "bot or human?", "human or ai"
        r"\b(?:bot|ai)\s+or\s+(?:a\s+)?(?:human|person|real)\b|"
        r"\b(?:human|person|real)\s+or\s+(?:a\s+)?(?:bot|ai)\b",
    )
)


def detect_identity_question(text: str, *, names: Sequence[str] = ()) -> bool:
    """True when ``text`` asks (or jokes) whether the bot is a bot or a real person.

    "are you a bot", "r u real", "ur literally an ai", "is mp3 a bot", "am i
    talking to a real person". Talk about bots in general ("the music bot is
    down", "i made a discord bot") doesn't count.
    """
    text = _prepare(text, names)
    return any(pattern.search(text) for pattern in _IDENTITY_PATTERNS)


# --------------------------------------------------------------- agreements

# Words that turn "joining" into "not joining".
_NEGATOR_RE = re.compile(
    r"\b(?:not|never|no|nah|nope|can'?t|cannot|won'?t|wont|don'?t|dont|ain'?t|aint|isn'?t|wasn'?t|wouldn'?t|"
    r"couldn'?t|stop)\b",
    re.IGNORECASE,
)
# What may follow "joining" / "hopping on" when it means "i'm on my way".
_ON_MY_WAY_TAIL = (
    r"(?=\s*(?:$|[.!,?]|now\b|rn\b|in\b|on\b|you\b|u\b|ya\b|vc\b|call\b|the\s+(?:vc|call)\b|lol\b|bro\b|fr\b|asap\b|"
    r"real\s+quick\b|rq\b|one\s+sec\b|1\s+sec\b|gimme\b))"
)
# Saying yes to, or pretending to do, the request.
_AGREEMENT_RE = re.compile(
    r"\b(?:omw|otw|on\s+(?:my|the)\s+way|be\s+(?:right\s+)?there|pulling\s+up"
    r"|(?:joining|hopping\s+(?:on|in)|jumping\s+(?:on|in)|coming\s+(?:now|rn|over))"
    + _ON_MY_WAY_TAIL
    + r"|"
    r"(?:i'?ll|ill|i\s+will|lemme|let\s+me|gonna|imma|ima)\s+(?:[\w']+\s+)?(?:join|hop|jump|come|call|send|add|text|dm|facetime|ft|"
    r"stream|play|pull\s+up|accept|check|look|watch|listen|remind|ping|react|meet)\b|"
    r"i'?m\s+(?:in|down|joining|coming|there|on\s+it)(?=\s*(?:$|[.!,?]|lol\b|fr\b|bro\b|rn\b|now\b))|"
    r"sending(?:\s+(?:it|now|rn|one|you|u|ya|that|them))?\b|just\s+sent|sent\s+(?:it|you|u|ya|one|that|them)\b|"
    r"added\s+(?:you|u|ya)\b|accepted\s+(?:it|your|ur|the)\b|let'?s\s+do\s+it|say\s+less)",
    re.IGNORECASE,
)
# Putting it off, which still promises to do it.
_DEFERRAL_RE = re.compile(
    r"(?<!not\s)\b(?:maybe\s+(?:later|tmrw|tomorrow|tonight)|next\s+time|another\s+time|some\s+other\s+time|"
    r"later\s+tho(?:ugh)?|(?:give|gimme|give\s+me)\s+(?:a\s+|one\s+)?(?:sec|second|min|minute|moment)|one\s+sec|"
    r"1\s+sec|brb|in\s+a\s+(?:bit|sec|min))\b",
    re.IGNORECASE,
)
# A bare yes that is the whole reply: "sure", "bet", "sent", "done".
_BARE_YES_RE = re.compile(
    r"^\W*(?:sure|bet|ok(?:ay)?|k|kk|yeah|yea|ye|yes|yep|ya|aight|ight|alright|on\s+it|coming|sent|done|added|"
    r"joined|joining|accepted)(?:\s+(?:bro|man|dude|lol|fr|then))?\W*$",
    re.IGNORECASE,
)


def looks_like_agreement(reply: str) -> bool:
    """True when ``reply`` agrees to, or pretends to carry out, a request.

    Meant for a reply to a request the bot can't fulfil (see
    :func:`detect_impossible_request`): "omw", "joining now", "sent", "added
    you", "sure, gimme a sec" and "maybe later" all promise something it can't
    do, while "nah i dont do vc" and "not joining lol" turn it down. A bare
    "ok" or "sure" counts too, so don't use this on ordinary chat.
    """
    reply = reply.replace("\u2019", "'")[:MAX_SCAN_CHARS]
    if _BARE_YES_RE.match(reply) or _DEFERRAL_RE.search(reply):
        return True
    for match in _AGREEMENT_RE.finditer(reply):
        clause_start = max(reply.rfind(mark, 0, match.start()) for mark in ".,!?;\n") + 1
        if not _NEGATOR_RE.search(reply, clause_start, match.end()):
            return True
    return False


# ------------------------------------------------------------------ helpers

# The @mentions a line starts with, after any emoji and filler words:
# "yo @bob, @alice: ..." (group 2 is the run of mentions).
_LEADING_AT_MENTIONS_RE = re.compile(
    r"^([^\w@<\n]*(?:" + _FILLER_WORD + r"[ \t,.!:]+)*)((?:@[^\s,:;@]+[ \t,:;]*)+)",
    re.IGNORECASE | re.MULTILINE,
)
_AT_MENTION_RE = re.compile(r"@[^\s,:;@]+")
_GROUP_MENTIONS = frozenset({"@bot", "@everyone", "@here"})


def _prepare(text: str, names: Iterable[str]) -> str:
    """Normalise ``text`` for matching and drop lines meant for someone else.

    A line led by @mentions that include the bot ("@bob @mp3 hop in vc") reads
    as led by "@bot"; one led only by other people ("yo @bob hop in vc") is
    meant for them, so it is dropped.
    """
    text = text.replace("\u2019", "'")[:MAX_SCAN_CHARS]
    text = _normalize_names(text, names)
    if "@" not in text:
        return text
    lines = []
    for line in text.split("\n"):
        match = _LEADING_AT_MENTIONS_RE.match(line)
        if match is not None:
            mentions = {mention.lower() for mention in _AT_MENTION_RE.findall(match.group(2))}
            if "@bot" in mentions:
                line = match.group(1) + "@bot " + line[match.end() :]
            elif not mentions & _GROUP_MENTIONS:
                continue
        lines.append(line)
    return "\n".join(lines)


def _normalize_names(text: str, names: Iterable[str]) -> str:
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True) if name.strip()
    )
    if not alternatives:
        return text
    pattern = re.compile(rf"(?<![\w@])@?(?:{alternatives})(?!\w)", re.IGNORECASE)
    return pattern.sub("@bot", text)
