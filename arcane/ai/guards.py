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

Input is capped at ``MAX_SCAN_CHARS`` so matching time stays bounded no matter
what someone pastes into the chat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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

# Words and mentions people put before a request: "yo", "bro", "pls", "@bob".
_FILLER = r"(?:yo|bro|bruh|ok|okay|lol|lmao|pls|please|just|so|and|hey|now|anyways?|<@!?\d+>|@\S+)"
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
# What may follow a requested action: end, "with me", "rn", "later", ...
_REQUEST_TAIL = (
    r"(?=\s*(?:$|[?.!,]|with\s+(?:me|us)\b|w\s+(?:me|us)\b|rn\b|now\b|later\b|tonight\b|tn\b|tmrw\b|tomorrow\b|"
    r"sometime\b|rq\b|again\b|or\b|pls\b|bro\b|lol\b|then\b|soon\b))"
)
# Optional time words, then the end of the sentence.
_REQUEST_END = (
    r"(?:\s+(?:rn|now|later|tonight|tn|tmrw|tomorrow|with\s+(?:me|us)|pls|please|bro|rq|then|again|or\s+nah|or\s+what))*"
    r"\s*(?=[?.!]|\n|$)"
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
    r"sea\s+of\s+thieves|ranked|comp|duos|squads|1v1s?|some\s+(?:games|rounds)|online|"
    r"on\s+(?:pc|xbox|ps[45]|playstation|switch|steam))"
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
_VOICE_NOUN = r"(?:the\s+|a\s+|my\s+|our\s+|this\s+|general\s+)?(?:discord\s+)?(?:vc|voice\s*(?:chat|channel|call)|call(?!\s+of\b)|stage)\b"


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
            "(?:"
            + _ASKED_OR_INVITED
            + r"(?:play|hop\s+on|get\s+on|jump\s+on|boot\s+up|queue(?:\s+up)?|q\s+up|grind|duo)\s+"
            r"(?:(?:some|a\s+(?:game|round)\s+of|a\s+few\s+(?:games|rounds)\s+of|the|a)\s+)?|"
            + _DOWN_FOR
            + ")"
            + _GAMES
            + r"(?!\w)",
            r"\b(?:do|d|you|u|ya)\s+(?:you\s+|u\s+)?(?:even\s+|ever\s+|still\s+)?play\s+"
            + _GAMES
            + r"(?!\w)",
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
        ],
    ),
    "open_links": (
        "open a link or look at, watch or listen to something they sent",
        [
            _ASKED_OF_BOT
            + r"(?:watch|listen\s+to|check\s+out|look\s+at|open|click(?:\s+on)?|peep|rate|see)\s+(?:(?:this|that|my|these|those)\s+(?:[\w']+\s+){0,2}?"
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
            r"\b(?:watch|listen|check|look|open|click|peep|thoughts|rate|see|opinion)\b[^\n]{0,60}https?://\S+|https?://\S+[^\n]{0,80}\b(?:watch|listen|check\s+(?:it|this)\s+out|thoughts|what\s+do\s+(?:you|u)\s+think|rate|opinion)\b",
            r"\b(?:what\s+do\s+(?:you|u)\s+think|thoughts|opinions?|rate|how\s+do\s+(?:you|u)\s+like)\s+(?:(?:of|on|about)\s+)?(?:this|my)\s+(?:[\w']+\s+)?"
            r"(?:pics?|pfp|photo|picture|video|vid|clip|song|track|beat|drawing|art|fit|outfit|setup|edit|tiktok|meme|banner|image|ss|screenshot)\b",
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
        names: Names the bot answers to. They are rewritten to the "@name" form
            Discord uses for a real mention, which the patterns already treat as
            a lead-in, so "yo mp3 hop in vc" reads like "yo @mp3 hop in vc".
    """
    text = text.replace("\u2019", "'")[:MAX_SCAN_CHARS]
    text = _normalize_names(text, names)
    for category in CATEGORIES:
        if any(pattern.search(text) for pattern in category.patterns):
            return ImpossibleRequest(category.key, category.label)
    return None


def _normalize_names(text: str, names: Iterable[str]) -> str:
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True) if name.strip()
    )
    if not alternatives:
        return text
    pattern = re.compile(rf"(?<![\w@])(?:{alternatives})(?!\w)", re.IGNORECASE)
    return pattern.sub("@bot", text)
