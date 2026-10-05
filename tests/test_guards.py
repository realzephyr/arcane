from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from arcane.ai.guards import CATEGORIES, MAX_SCAN_CHARS, detect_impossible_request

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


def test_matching_time_is_bounded() -> None:
    adversarial = [
        "yo " * 2000,
        ("hey " * 50 + "you wanna ") * 50,
        "https://" + "a" * 5000 + " watch" * 200,
    ]
    for text in adversarial:
        started = time.perf_counter()
        detect_impossible_request(text, names=NAMES)
        assert time.perf_counter() - started < 0.5
    assert MAX_SCAN_CHARS <= 2000
