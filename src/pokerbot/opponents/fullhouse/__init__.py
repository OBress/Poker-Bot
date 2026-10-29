"""Bots written for the fullhouse engine (e-x.club Showdown), played inside our engine.

A fullhouse bot is a Python file with ``decide(game_state: dict) -> dict``. ``FullhouseBot`` turns
our ``DecisionState`` into their ``game_state`` dict and their reply back into an ``Action``, so
any submission-style ``bot.py`` can sit at our tables. ``bots/`` holds the five reference bots that
ship with the engine (MIT, see ``bots/LICENSE``), frozen at ``VERSION``.

Their rules, mirrored here: "check" when facing a bet is a call, "call" when nothing is owed is a
check, raises below the minimum snap up, a raise of the whole stack or more is an all-in, and an
unknown action, a malformed reply or an exception is a fold. Differences that remain because the
hand is played by our engine (``engine/state.py``): a fold with nothing to call becomes a check,
and a raise when betting isn't reopened becomes a call. ``match_action_log`` (previous hands of the
match) isn't provided; the reference bots don't read it.

Their bots use the stdlib ``random`` module. Each instance gets its own copy of the module with
``random`` replaced by a seeded ``random.Random``, so play is reproducible and instances don't
share state (AGENTS.md rule 3).
"""

from __future__ import annotations

import random
import types
from functools import cache
from pathlib import Path
from typing import Any

from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.cards import card_str
from pokerbot.engine.events import BlindPosted, Event, HandStart
from pokerbot.engine.state import DecisionState

VERSION = "fullhouse-engine@a80b269"  # github.com/uzlez/fullhouse-engine, bots/ at this commit
BOTS_DIR = Path(__file__).parent / "bots"
REFERENCE_BOTS = ("template", "aggressor", "mathematician", "shark", "ref_bot_2")

_STREETS = {
    Street.PREFLOP: "preflop",
    Street.FLOP: "flop",
    Street.TURN: "turn",
    Street.RIVER: "river",
}


@cache
def _compiled(path: Path) -> types.CodeType:
    return compile(path.read_text(encoding="utf-8"), str(path), "exec")


def load_module(path: Path, seed: int) -> types.ModuleType:
    """A fresh module for one bot instance, with its ``random`` seeded."""
    module = types.ModuleType(f"fullhouse_bot_{path.stem}")
    module.__file__ = str(path)
    exec(_compiled(path.resolve()), module.__dict__)
    rng = random.Random(seed)
    for name, value in list(module.__dict__.items()):
        if value is random:
            setattr(module, name, rng)
    if not callable(getattr(module, "decide", None)):
        raise ValueError(f"{path} has no decide(game_state)")
    return module


def reference_path(name: str) -> Path:
    if name not in REFERENCE_BOTS:
        raise ValueError(f"unknown fullhouse bot {name!r}; choose from {REFERENCE_BOTS}")
    return BOTS_DIR / f"{name}.py"


class FullhouseBot:
    """Adapter: a fullhouse ``bot.py`` as one of our policies."""

    def __init__(self, path: Path | str, seed: int, name: str = "") -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.module = load_module(self.path, seed)
        self.errors = 0  # exceptions and malformed replies, each played as a fold
        self._hand_id = ""
        self._blinds: list[dict[str, Any]] = []

    def observe(self, event: Event) -> None:
        if isinstance(event, HandStart):
            self._hand_id = f"h{event.hand_id:05d}"
            self._blinds = []
        elif isinstance(event, BlindPosted):
            kind = "small_blind" if not self._blinds else "big_blind"
            self._blinds.append({"seat": event.seat, "action": kind, "amount": event.amount})

    def act(self, state: DecisionState) -> Action:
        try:
            reply = self.module.decide(game_state(state, self._hand_id, self._blinds))
            return to_action(reply, state)
        except Exception:
            self.errors += 1
            return Action.fold()


def game_state(
    state: DecisionState, hand_id: str = "", blinds: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The ``game_state`` dict the fullhouse engine sends for this decision."""
    seat = state.seat
    current_bet = max(state.bets)
    owed = current_bet - state.bets[seat]
    players = []
    for i in range(state.n_players):
        folded = state.folded[i]
        all_in = not folded and state.stacks[i] == 0
        status = "folded" if folded else "all_in" if all_in else "active"
        players.append(
            {
                "seat": i,
                "bot_id": f"seat{i}",
                "stack": state.stacks[i],
                "state": status,
                "is_folded": folded,
                "is_all_in": all_in,
                "bet_this_street": state.bets[i],
                "hole_cards": None,
            }
        )
    return {
        "type": "action_request",
        "hand_id": hand_id,
        "street": _STREETS[state.street],
        "seat_to_act": seat,
        "pot": state.pot,
        "community_cards": [card_str(c) for c in state.board],
        "current_bet": current_bet,
        "min_raise_to": state.legal.min_raise_to,
        "amount_owed": owed,
        "can_check": owed == 0,
        "your_cards": [card_str(c) for c in state.hole],
        "your_stack": state.stacks[seat],
        "your_bet_this_street": state.bets[seat],
        "players": players,
        "action_log": list(blinds or []) + [_logged(a) for a in state.actions],
        "match_action_log": [],
    }


def _logged(a) -> dict[str, Any]:
    if a.type == ActionType.FOLD:
        return {"seat": a.seat, "action": "fold", "amount": 0}
    if a.type == ActionType.CHECK_CALL:
        if a.to_call == 0:
            return {"seat": a.seat, "action": "check", "amount": 0}
        return {"seat": a.seat, "action": "call", "amount": a.to_call}
    return {"seat": a.seat, "action": "all_in" if a.is_all_in else "raise", "amount": a.to}


def to_action(reply: Any, state: DecisionState) -> Action:
    """Their reply -> our action, following their validator."""
    if not isinstance(reply, dict):
        raise TypeError(f"decide() returned {type(reply).__name__}, not dict")
    kind = str(reply.get("action", "fold")).lower().strip()
    legal = state.legal
    if kind in ("check", "call"):
        return Action.check_call()
    if kind == "all_in":
        return Action.raise_to(legal.max_raise_to)
    if kind == "raise":
        try:
            amount = int(reply.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        return Action.raise_to(min(max(amount, legal.min_raise_to), legal.max_raise_to))
    return Action.fold()
