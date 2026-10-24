"""An 8-seat no-limit hold'em table for the demo, driven by the project's engine.

Seats are empty, human (a connected browser), PokerBot (``EngineSeat``), a calibrated style bot
from the full-ring population (``profiles/bots-v3.json``), or the TAG baseline. A hand is dealt
to every occupied seat; the engine's ``HandState`` uses hand-local indices 0..n-1, which map to
table seats in order. Busted seats rebuy to 100 big blinds before the next hand.

``Table`` is pure game logic. ``main.py`` drives it with asyncio, asks bots for actions in a
worker thread, waits for humans, and sends every client its own view (``view``), which only
ever contains public information plus that client's hole cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pokerbot.baselines.heuristic import abc_tag
from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.cards import card_str, shuffled_deck
from pokerbot.engine.events import BoardDealt, HandEnd, Muck, PlayerAction, Showdown
from pokerbot.engine.state import HandState
from pokerbot.opponents.sampler import StyleSampler

from .engine_seat import EngineSeat, fr_profiles

ROOT = Path(__file__).resolve().parents[2]
BOTS = ROOT / "profiles/bots-v3.json"
SEATS = 8
SB, BB = 50, 100
BUYIN = 100 * BB
KINDS = ("empty", "human", "pokerbot", "bot", "tag")
STREETS = {
    Street.PREFLOP: "preflop",
    Street.FLOP: "flop",
    Street.TURN: "turn",
    Street.RIVER: "river",
}


@dataclass
class Seat:
    kind: str = "empty"
    name: str = ""
    stack: int = BUYIN
    style: int | None = None  # bot: index into the full-ring styles; None = random draw
    client: str | None = None  # human: the client id that holds the seat
    policy: object = None
    style_name: str = ""  # bot: its true style (revealed in the UI only when asked)
    sitting_out: bool = False
    last_action: str = ""
    net: int = 0  # chips won or lost since the table started


@dataclass
class HandInfo:
    state: HandState
    seats: list[int]  # hand-local index -> table seat
    log: list[str] = field(default_factory=list)


class Table:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.sampler = StyleSampler(BOTS, "fr")
        self.seats = [Seat() for _ in range(SEATS)]
        self.button = 0
        self.hand_no = 0
        self.hand: HandInfo | None = None
        self.running = False
        self.log: list[str] = []
        self.results: dict | None = None  # last hand's showdown and winners
        self.composition: tuple | None = None
        self.reads: dict[int, dict] = {}  # PokerBot's read per table seat, refreshed per hand
        self.default_setup()

    # ------------------------------------------------------------- setup

    def default_setup(self) -> None:
        """You in seat 0, two PokerBots, five calibrated bots of different styles."""
        layout = ["human", "bot", "pokerbot", "bot", "bot", "pokerbot", "bot", "bot"]
        styles = [None, 0, None, 4, 1, None, 2, 3]
        for i, (kind, style) in enumerate(zip(layout, styles, strict=True)):
            self.configure(i, kind, style)

    def configure(self, i: int, kind: str, style: int | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown seat kind {kind!r}")
        s = self.seats[i]
        keep_client = s.client if kind == "human" else None
        self.seats[i] = s = Seat(kind=kind, style=style, client=keep_client)
        if kind == "pokerbot":
            n = sum(1 for x in self.seats if x.kind == "pokerbot")
            s.name = "PokerBot" if n == 1 else f"PokerBot {n}"
            s.policy = EngineSeat(int(self.rng.integers(2**31)))
        elif kind == "bot":
            draw = self.sampler.draw(self.rng, style=style)
            s.policy, s.style_name = draw.bot, self.sampler.styles[draw.style]["name"]
            used = {x.name for x in self.seats}
            s.name = str(self.rng.choice([n for n in _NAMES if n not in used]))
            s.stack = int(round(min(draw.bot.buyin_bb, 200) * BB))
        elif kind == "tag":
            s.policy, s.name = abc_tag(int(self.rng.integers(2**31))), "TAG baseline"
        elif kind == "human":
            s.name = "Open seat"

    def sit(self, client: str, name: str, i: int) -> None:
        s = self.seats[i]
        if s.kind not in ("empty", "human") or (s.client and s.client != client):
            raise ValueError("seat taken")
        for other in self.seats:  # one seat per client
            if other.client == client and other is not s:
                other.client, other.name = None, "Open seat"
        if s.kind == "empty":
            self.configure(i, "human")
        s.client, s.name, s.sitting_out = client, name[:16] or "Player", False

    def leave(self, client: str) -> None:
        for s in self.seats:
            if s.client == client:
                s.client, s.name = None, "Open seat"

    def seat_of(self, client: str) -> int | None:
        return next((i for i, s in enumerate(self.seats) if s.client == client), None)

    # ------------------------------------------------------------- hands

    def dealt_seats(self) -> list[int]:
        """Occupied seats that play the next hand (open human seats sit out)."""
        return [
            i
            for i, s in enumerate(self.seats)
            if s.kind in ("pokerbot", "bot", "tag") or (s.kind == "human" and s.client)
        ]

    def start_hand(self) -> bool:
        seats = self.dealt_seats()
        if len(seats) < 2:
            return False
        for i in seats:  # rebuy
            if self.seats[i].stack < BB:
                self.seats[i].stack = BUYIN
                self._note(f"{self.seats[i].name} rebuys for {BUYIN // BB} bb")
        comp = tuple((i, self.seats[i].kind, id(self.seats[i].policy)) for i in seats)
        if comp != self.composition:
            self.composition = comp
            for i in seats:
                if isinstance(self.seats[i].policy, EngineSeat):
                    self.seats[i].policy.reset(len(seats))
            self.reads = {}
        while self.button not in seats:
            self.button = (self.button + 1) % SEATS
        state = HandState(
            [self.seats[i].stack for i in seats],
            seats.index(self.button),
            shuffled_deck(int(self.rng.integers(2**63))),
            SB,
            BB,
            self.hand_no,
        )
        self.hand = HandInfo(state, seats)
        self.results = None
        for s in self.seats:
            s.last_action = ""
        self._note(f"Hand #{self.hand_no + 1}: {self.seats[self.button].name} has the button")
        return True

    def actor_seat(self) -> int | None:
        if not self.hand or self.hand.state.finished or self.hand.state.actor is None:
            return None
        return self.hand.seats[self.hand.state.actor]

    def bot_action(self) -> Action:
        st = self.hand.state
        return self.seats[self.hand.seats[st.actor]].policy.act(st.decision_state())

    def apply(self, action: Action) -> None:
        st = self.hand.state
        seat = self.hand.seats[st.actor]
        board_before = len(st.board)
        st.apply(action)
        ev = next(e for e in reversed(st.events) if isinstance(e, PlayerAction))
        self.seats[seat].last_action = _describe(ev)
        self._note(f"{self.seats[seat].name}: {_describe(ev)}")
        if len(st.board) > board_before and not st.finished:
            for s in self.seats:
                s.last_action = "" if s.last_action != "Fold" else s.last_action
        for e in st.events:
            if isinstance(e, BoardDealt) and len(e.board) > board_before:
                self._note(f"{STREETS[e.street].title()}: {' '.join(map(card_str, e.board))}")
                board_before = len(e.board)
        if st.finished:
            self._finish()

    def _finish(self) -> None:
        h = self.hand
        st = h.state
        shown = {
            h.seats[e.seat]: [card_str(c) for c in e.cards]
            for e in st.events
            if isinstance(e, Showdown)
        }
        mucked = [h.seats[e.seat] for e in st.events if isinstance(e, Muck)]
        end = next(e for e in st.events if isinstance(e, HandEnd))
        winners = []
        for j, pay in enumerate(end.payoffs):
            seat = h.seats[j]
            self.seats[seat].stack += pay
            self.seats[seat].net += pay
            if pay > 0:
                winners.append({"seat": seat, "amount": pay + st.contrib[j]})
                self._note(
                    f"{self.seats[seat].name} wins {(pay + st.contrib[j]) / BB:g} bb"
                    + (f" with {' '.join(shown[seat])}" if seat in shown else "")
                )
        for j, seat in enumerate(h.seats):
            p = self.seats[seat].policy
            if isinstance(p, EngineSeat):
                p.observe_hand(st.events, j)
        self.results = {"shown": shown, "mucked": mucked, "winners": winners}
        self.reads = self._reads()
        self.hand_no += 1
        seats = self.dealt_seats()
        nxt = [i for i in seats if i > self.button] or seats
        self.button = nxt[0] if nxt else self.button

    def _note(self, text: str) -> None:
        self.log.append(text)
        del self.log[:-200]

    # ------------------------------------------------------------- views

    def view(self, client: str | None) -> dict:
        """Everything one client may see."""
        me = self.seat_of(client) if client else None
        h = self.hand
        st = h.state if h else None
        local = {t: j for j, t in enumerate(h.seats)} if h else {}
        reads = self.reads
        seats = []
        for i, s in enumerate(self.seats):
            j = local.get(i)
            in_hand = st is not None and j is not None
            cards = None
            if in_hand:
                if self.results and i in self.results["shown"]:
                    cards = self.results["shown"][i]
                elif i == me:
                    cards = [card_str(c) for c in st.hole[j]]
            seats.append(
                {
                    "index": i,
                    "kind": s.kind,
                    "name": s.name,
                    "human": s.kind == "human",
                    "occupied": s.kind != "empty" and (s.kind != "human" or s.client is not None),
                    "you": i == me,
                    "stack": st.stacks[j] if in_hand and not st.finished else s.stack,
                    "bet": st.bets[j] if in_hand and not st.finished else 0,
                    "inHand": in_hand,
                    "folded": bool(in_hand and st.folded[j]),
                    "allIn": bool(
                        in_hand and not st.finished and st.stacks[j] == 0 and not st.folded[j]
                    ),
                    "cards": cards,
                    "hasCards": in_hand and not (st.folded[j]),
                    "lastAction": s.last_action,
                    "net": s.net,
                    "read": reads.get(i),
                    "style": s.style_name if s.kind == "bot" else None,
                    "engineMode": s.policy.mode if isinstance(s.policy, EngineSeat) else None,
                }
            )
        legal = None
        actor = self.actor_seat()
        if actor is not None and actor == me:
            lg = st.legal()
            legal = {
                "toCall": lg.to_call,
                "canRaise": lg.can_raise,
                "minRaiseTo": lg.min_raise_to,
                "maxRaiseTo": lg.max_raise_to,
                "bet": st.bets[st.actor],
                "pot": st.pot,
            }
        return {
            "type": "state",
            "running": self.running,
            "handNo": self.hand_no + (0 if st is None or st.finished else 1),
            "seats": seats,
            "button": self.button,
            "board": [card_str(c) for c in st.board] if st else [],
            "pot": st.pot if st else 0,
            "street": STREETS[st.street] if st and not st.finished else None,
            "actor": actor,
            "you": me,
            "legal": legal,
            "results": self.results,
            "log": self.log[-40:],
            "bigBlind": BB,
            "styles": [s["name"] for s in self.sampler.styles],
        }

    def _reads(self) -> dict[int, dict]:
        """PokerBot's live read of every other player (from the first PokerBot seat)."""
        h = self.hand
        bot = next((s.policy for s in self.seats if isinstance(s.policy, EngineSeat)), None)
        if bot is None or h is None or bot.n != len(h.seats):
            return {}
        out = {}
        for j, r in bot.reads().items():
            seat = h.seats[j]
            if isinstance(self.seats[seat].policy, EngineSeat):
                continue
            top = sorted(r.probs.items(), key=lambda kv: -kv[1])[:3]
            out[seat] = {
                "style": r.style,
                "confidence": r.confidence,
                "hands": r.hands,
                "top": [{"style": k, "p": v} for k, v in top],
            }
        return out


def parse_action(msg: dict) -> Action:
    kind = msg.get("action")
    if kind == "fold":
        return Action.fold()
    if kind in ("check", "call"):
        return Action.check_call()
    if kind == "raise":
        return Action.raise_to(int(msg.get("amount", 0)))
    raise ValueError(f"unknown action {kind!r}")


def _describe(e: PlayerAction) -> str:
    if e.type == ActionType.FOLD:
        return "Fold"
    if e.type == ActionType.CHECK_CALL:
        return (
            "Check"
            if e.amount == 0
            else ("All-in" if e.is_all_in else f"Call {e.amount / BB:g} bb")
        )
    if e.is_all_in:
        return f"All-in {e.to / BB:g} bb"
    return f"{'Bet' if e.to_call == 0 else 'Raise to'} {e.to / BB:g} bb"


_NAMES = [
    "Ava",
    "Ben",
    "Cleo",
    "Dev",
    "Eli",
    "Fay",
    "Gus",
    "Hana",
    "Ivo",
    "Jun",
    "Kai",
    "Lena",
    "Milo",
    "Nia",
    "Omar",
    "Pia",
    "Quin",
    "Rosa",
    "Sam",
    "Tess",
    "Uma",
    "Vik",
    "Wren",
    "Yara",
    "Zed",
]


def style_names() -> list[str]:
    return list(fr_profiles().names)
