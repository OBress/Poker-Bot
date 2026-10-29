"""Build policies from short spec strings, so configs and worker processes can name players.

Specs:
  fullhouse:<name>   a fullhouse reference bot (``opponents/fullhouse``), e.g. fullhouse:shark
  fullhouse:<path>   any fullhouse-style bot.py file
  style:<name>       the calibrated center of a style in the bots file, e.g. style:tag
  calling-station    always checks or calls
  random             RandomPolicy with default rates
  abc | hand-strength | blueprint:<path>   the Phase 4 baselines (``eval/duplicate.py``)
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

from pokerbot.engine.policies import CallingStation, RandomPolicy
from pokerbot.engine.table import Policy
from pokerbot.opponents.bot import StyleBot
from pokerbot.opponents.fullhouse import REFERENCE_BOTS, FullhouseBot, reference_path

DEFAULT_BOTS = Path("profiles/bots-v2.json")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@cache
def _styles(bots_path: Path, fmt: str) -> dict[str, dict]:
    doc = json.loads(Path(bots_path).read_text(encoding="utf-8"))
    return {slug(s["name"]): s for s in doc["formats"][fmt]["styles"]}


def make_policy(
    spec: str, seed: int, stack_bb: float = 100.0, bots_path: Path = DEFAULT_BOTS, fmt: str = "sh"
) -> Policy:
    kind, _, arg = spec.partition(":")
    if kind == "fullhouse":
        path = reference_path(arg) if arg in REFERENCE_BOTS else Path(arg)
        return FullhouseBot(path, seed, name=spec)
    if kind == "style":
        styles = _styles(Path(bots_path), fmt)
        if slug(arg) not in styles:
            raise ValueError(f"unknown style {arg!r}; choose from {sorted(styles)}")
        s = styles[slug(arg)]
        return StyleBot({**s["params"], "buyin": stack_bb}, seed, s["name"])
    if spec == "calling-station":
        return CallingStation()
    if spec == "random":
        return RandomPolicy(seed)
    from pokerbot.eval.duplicate import make_policy as baseline_policy

    return baseline_policy(spec, seed)
