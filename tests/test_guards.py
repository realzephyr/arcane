from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from arcane.ai.guards import (
    CATEGORIES,
    MAX_SCAN_CHARS,
    DebateRequest,
    detect_debate_request,
    detect_impossible_request,
    looks_like_agreement,
)

CASES = json.loads((Path(__file__).parent / "data" / "guard_cases.json").read_text())
NAMES = ("mp3", "mp 3")


@pytest.mark.parametrize(
    ("category", "text"),
    [(category, text) for category, texts in CASES["positives"].items() for text in texts],
)
def test_requests_are_detected(category: str, text: str) -> None:
    request = detect_impossible_request(text, names=NAMES)
    assert request is not None, text
    assert request.key == category


@pytest.mark.parametrize("text", CASES["negatives"])
def test_mentions_in_passing_are_not_requests(text: str) -> None:
    assert detect_impossible_request(text, names=NAMES) is None


def test_every_category_has_cases_and_a_usable_label() -> None:
    assert {c.key for c in CATEGORIES} == set(CASES["positives"])
    for category in CATEGORIES:
        assert category.label and not category.label.endswith(".")
        assert category.patterns


@pytest.mark.parametrize(
    "text",
    [
        "yo mp3 wanna hop in vc",
        "mp3 hop in vc",
        "ok mp3, send a pic",
        "bro mp3 whats your snap",
        "mp 3 wanna play val",
        "@mp3 join vc",
    ],
)
def test_requests_addressing_the_bot_by_name(text: str) -> None:
    assert detect_impossible_request(text, names=NAMES) is not None


@pytest.mark.parametrize(
    "text", ["mp3 was in vc earlier", "send me the mp3 file", "i like mp3 more than wav"]
)
def test_name_mentions_that_are_not_requests(text: str) -> None:
    assert detect_impossible_request(text, names=NAMES) is None


def test_names_work_for_any_personality() -> None:
    assert detect_impossible_request("yo vex hop in vc", names=("vex",)) is not None
    assert detect_impossible_request("vex, wanna vc?", names=("vex",)) is not None
    assert detect_impossible_request("vex was in vc yesterday", names=("vex",)) is None


def test_curly_apostrophes_are_normalised() -> None:
    assert detect_impossible_request("what\N{RIGHT SINGLE QUOTATION MARK}s your snap") is not None


@pytest.mark.parametrize(
    "text",
    ["yo @bob hop in vc", "@bob, @alice: wanna play val", "\N{SKULL} @bob send a pic"],
)
def test_requests_led_by_someone_elses_mention_are_ignored(text: str) -> None:
    assert detect_impossible_request(text, names=NAMES) is None
    assert detect_impossible_request(text.replace("@bob", "@mp3"), names=NAMES) is not None


# ------------------------------------------------------------------ debates


@pytest.mark.parametrize("text", CASES["debate"]["text"])
def test_text_debate_requests(text: str) -> None:
    assert detect_debate_request(text, names=NAMES) == DebateRequest(voice=False)


@pytest.mark.parametrize("text", CASES["debate"]["voice"])
def test_voice_debate_requests(text: str) -> None:
    assert detect_debate_request(text, names=NAMES) == DebateRequest(voice=True)


@pytest.mark.parametrize("text", CASES["debate"]["negatives"])
def test_debate_mentions_in_passing_are_not_requests(text: str) -> None:
    assert detect_debate_request(text, names=NAMES) is None


def test_debate_names_and_lines() -> None:
    assert detect_debate_request("vex debate me", names=("vex",)) is not None
    assert detect_debate_request("@bob debate me\nlol", names=NAMES) is None
    assert detect_debate_request("@bob lol\nmp3 debate me", names=NAMES) is not None


# --------------------------------------------------------------- agreements


@pytest.mark.parametrize(
    "reply",
    [
        "omw",
        "on my way",
        "joining",
        "joining now",
        "hopping on",
        "hopping in",
        "ill join",
        "i'll hop on",
        "i\N{RIGHT SINGLE QUOTATION MARK}ll hop on in a sec",
        "sent",
        "sent!",
        "sending it",
        "just sent it",
        "added you",
        "sure, gimme a sec",
        "ok joining",
        "yeah lets do it",
        "bet, joining",
        "maybe later",
        "nah maybe later",
        "next time",
        "another time",
        "later tho",
        "sure",
        "bet",
    ],
)
def test_agreements_are_recognised(reply: str) -> None:
    assert looks_like_agreement(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "nah i dont do vc",
        "cant rn, my mic is busted",
        "not joining lol",
        "never joining vc",
        "nah, text debate here tho",
        "i sent my essay yesterday",
        "joining a club in college was the best decision i made",
        "hopping on the bandwagon i see",
        "free will is probably an illusion tbh",
        "i'm in class rn so im slow",
        "",
    ],
)
def test_refusals_and_chat_are_not_agreements(reply: str) -> None:
    assert not looks_like_agreement(reply)


def test_matching_time_is_bounded() -> None:
    adversarial = [
        "yo " * 2000,
        ("hey " * 50 + "you wanna ") * 50,
        "https://" + "a" * 5000 + " watch" * 200,
        "@bob " * 1000 + "hop in vc",
        "yo " * 500 + "@bob " * 500,
        "[image: " * 500 + "x]" * 500,
        "debate " * 1000,
        "lets debate me in " * 200,
        "i bet i could " * 300 + "debate you",
        "not " * 1000 + "joining",
        ", " * 2000 + "omw",
    ]
    for text in adversarial:
        started = time.perf_counter()
        detect_impossible_request(text, names=NAMES)
        detect_debate_request(text, names=NAMES)
        looks_like_agreement(text)
        assert time.perf_counter() - started < 0.5
    assert MAX_SCAN_CHARS <= 2000
