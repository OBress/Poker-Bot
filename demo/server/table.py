"""An 8-seat no-limit hold'em table for the demo, driven by the project's engine.

Seats are empty, human (a connected browser), PokerBot (``EngineSeat``), a calibrated style bot
from the full-ring population (``profiles/bots-v3.json``), or the TAG baseline. A hand is dealt
to every occupied seat; the engine's ``HandState`` uses hand-local indices 0..n-1, which map to
table seats in order. Busted seats rebuy to 100 big blinds before the next hand.

Two modes:
- ``cash``: open-ended; seats can be changed between hands.
- ``read``: a fixed table for ``target`` hands, after which the game ends and shows how
  PokerBot classified each human.

Besides the PokerBot seats, the table runs its own ``reader`` (the same classifier, not
seated) over every hand, so reads of every player exist whoever sits at the table. Reads only
use public information.

``Table`` is pure game logic. ``main.py`` drives it with asyncio, asks bots for actions in a
worker thread, waits for humans, and sends every client its own view (``view``), which only
ever contains public information plus that client's hole cards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
BB = 100  # the engine's big blind in chips, always (the heads-up exploiter assumes it)
# The player-facing money: blinds and buy-in are shown in the units the game was set up with
# (``money``), but the engine always plays a 100-chip big blind; the two are proportional.
DEFAULT_SETUP = {
    "players": 8,
    "seats": [
        {"kind": k, "style": st}
        for k, st in zip(
            ["human", "bot", "pokerbot", "bot", "bot", "pokerbot", "bot", "bot"],
            [None, 0, None, 4, 1, None, 2, 3],
            strict=True,
        )
    ],
    "sb": 1.0,
    "bb": 2.0,
    "buyin": 200.0,
}
KINDS = ("empty", "human", "pokerbot", "bot", "tag")
MODES = ("cash", "read")
STREETS = {
    Street.PREFLOP: "preflop",
    Street.FLOP: "flop",
    Street.TURN: "turn",
    Street.RIVER: "river",
}
# What each full-ring style looks like (profiles/styles-v1.json, "fr"), in plain words
STYLE_INFO = {
    "Loose-passive": "Plays a lot of hands but mostly calls; rarely takes the lead.",
    "Passive limper": "Limps into pots and almost never raises before the flop.",
    "Weak-tight": "Plays few hands and gives up easily when someone bets.",
    "TAG": "Tight-aggressive: picks good hands and plays them with bets and raises.",
    "Nit": "Very tight and cautious: waits for strong hands.",
    "Short-stacker": "Buys in short and plays a narrow, all-in-heavy game.",
}
# Stats shown next to a read: (tracker/profile key, label)
STATS = (
    ("vpip", "Plays the hand (VPIP)"),
    ("pfr", "Raises before the flop"),
    ("limp", "Limps in"),
    ("3bet", "3-bets"),
    ("fold_to_cbet_flop", "Folds to a flop bet"),
    ("agg_flop", "Bets or raises on the flop"),
    ("wtsd", "Goes to showdown"),
)


@dataclass
class Seat:
    kind: str = "empty"
    name: str = ""
    stack: int = 100 * BB
    buyin: int = 100 * BB  # chips this seat starts and rebuys with
    style: int | None = None  # bot: index into the full-ring styles; None = random draw
    client: str | None = None  # human: the client id that holds the seat
    policy: object = None
    style_name: str = ""  # bot: its true style (revealed in the UI only when asked)
    last_action: str = ""
    net: int = 0  # chips won or lost since the table started
    hands: int = 0  # hands dealt to this seat
    gen: int = 0  # bumped whenever the seat is reconfigured: who sits here, for the reads


@dataclass
class HandInfo:
    state: HandState
    seats: list[int]  # hand-local index -> table seat


class Table:
    def __init__(
        self,
        seed: int = 0,
        mode: str = "cash",
        target: int | None = None,
        setup: dict | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.setup = parse_setup(setup)
        self.money = {"sb": self.setup["sb"], "bb": self.setup["bb"]}
        self.sb = max(1, round(BB * self.setup["sb"] / self.setup["bb"]))  # engine chips
        self.buyin = round(BB * self.setup["buyin"] / self.setup["bb"])
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.mode = mode
        self.target = target if mode == "read" else None
        self.id = time.strftime("%Y%m%d-%H%M%S") + f"-{mode}"
        self.started = time.time()
        self.ended: float | None = None
        self.sampler = StyleSampler(BOTS, "fr")
        self.seats = [Seat() for _ in range(SEATS)]
        self.button = 0
        self.hand_no = 0
        self.hand: HandInfo | None = None
        self.running = False
        self.log: list[str] = []
        self.results: dict | None = None  # last hand's showdown and winners
        self.composition: tuple | None = None
        self.local: dict[int, int] = {}  # table seat -> hand-local index, current composition
        self.reader = EngineSeat(seed)  # classifies everyone, seated or not
        self.reads: dict[int, dict] = {}  # read per table seat, refreshed after every hand
        self.trajectory: dict[int, list[dict]] = {}  # human seat -> read after each hand
        self.default_setup()

    # ------------------------------------------------------------- setup

    def default_setup(self) -> None:
        """The seats from the game's setup (default: seat 1 for you, two PokerBots, five
        calibrated bots of different styles)."""
        for i, seat in enumerate(self.setup["seats"]):
            self.configure(i, seat["kind"], seat["style"])

    def configure(self, i: int, kind: str, style: int | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown seat kind {kind!r}")
        s = self.seats[i]
        keep_client = s.client if kind == "human" else None
        self.seats[i] = s = Seat(kind=kind, style=style, client=keep_client, gen=s.gen + 1,
                                 stack=self.buyin, buyin=self.buyin)  # fmt: skip
        if kind == "pokerbot":
            n = sum(1 for x in self.seats if x.kind == "pokerbot")
            s.name = "PokerBot" if n == 1 else f"PokerBot {n}"
            s.policy = EngineSeat(int(self.rng.integers(2**31)))
        elif kind == "bot":
            draw = self.sampler.draw(self.rng, style=style)
            s.policy, s.style_name = draw.bot, self.sampler.styles[draw.style]["name"]
            used = {x.name for x in self.seats}
            s.name = str(self.rng.choice([n for n in _NAMES if n not in used]))
            # its calibrated buy-in (a short-stacker buys in short), at most the table's
            s.stack = s.buyin = min(int(round(draw.bot.buyin_bb * BB)), self.buyin)
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
        s.client, s.name = client, name[:16] or "Player"

    def resume_as(self, client: str, name: str) -> int:
        """A saved game is being resumed by ``client``: everyone's old connection is gone. Seat
        them where a human named ``name`` sat (else the first human seat); others are open."""
        humans = [i for i, s in enumerate(self.seats) if s.kind == "human"]
        if not humans:
            raise ValueError("this game has no seat for a person")
        mine = next((i for i in humans if self.seats[i].name == name), humans[0])
        for i in humans:
            self.seats[i].client = None
            if i != mine:
                self.seats[i].name = "Open seat"
        self.seats[mine].client, self.seats[mine].name = client, name[:16] or "Player"
        self.running = False
        self._note(f"{self.seats[mine].name} resumed the game")
        return mine

    def leave(self, client: str) -> int | None:
        """Stand up; returns the seat that was left (its live hand is folded by the caller)."""
        for i, s in enumerate(self.seats):
            if s.client == client:
                s.client, s.name = None, "Open seat"
                return i
        return None

    def seat_of(self, client: str | None) -> int | None:
        if client is None:
            return None
        return next((i for i, s in enumerate(self.seats) if s.client == client), None)

    @property
    def finished(self) -> bool:
        return self.target is not None and self.hand_no >= self.target

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
        if len(seats) < 2 or self.finished:
            return False
        for i in seats:  # rebuy
            if self.seats[i].stack < BB:
                self.seats[i].stack = self.seats[i].buyin
                self._note(f"{self.seats[i].name} rebuys for {self.amount(self.seats[i].buyin)}")
        # who sits where; the classifiers' memory is keyed by seat, so a change resets it. Stable
        # across saving and resuming the game (a resumed player gets a new client id).
        comp = tuple((i, self.seats[i].kind, self.seats[i].gen) for i in seats)
        if comp != self.composition:
            self.composition = comp
            for i in seats:
                if isinstance(self.seats[i].policy, EngineSeat):
                    self.seats[i].policy.reset(len(seats))
            self.reader.reset(len(seats))
            self.reads = {}
        self.local = {t: j for j, t in enumerate(seats)}
        while self.button not in seats:
            self.button = (self.button + 1) % SEATS
        state = HandState(
            [self.seats[i].stack for i in seats],
            seats.index(self.button),
            shuffled_deck(int(self.rng.integers(2**63))),
            self.sb,
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
        self.seats[seat].last_action = self.describe(ev)
        self._note(f"{self.seats[seat].name}: {self.describe(ev)}")
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
            self.seats[seat].hands += 1
            if pay > 0:
                winners.append({"seat": seat, "amount": pay + st.contrib[j]})
                self._note(
                    f"{self.seats[seat].name} wins {self.amount(pay + st.contrib[j])}"
                    + (f" with {' '.join(shown[seat])}" if seat in shown else "")
                )
        for j, seat in enumerate(h.seats):
            p = self.seats[seat].policy
            if isinstance(p, EngineSeat):
                p.observe_hand(st.events, j)
        self.reader.observe_hand(st.events, 0)
        self.results = {"shown": shown, "mucked": mucked, "winners": winners}
        self.reads = self._reads()
        self.hand_no += 1
        for seat, r in self.reads.items():
            if self.seats[seat].kind == "human":
                self.trajectory.setdefault(seat, []).append(
                    {"hand": self.hand_no, "probs": r["probs"]}
                )
        if self.finished:
            self.running = False
            self.ended = time.time()
            self._note(f"Game over after {self.hand_no} hands")
        seats = self.dealt_seats()
        nxt = [i for i in seats if i > self.button] or seats
        self.button = nxt[0] if nxt else self.button

    def fold_out(self, seat: int) -> bool:
        """If ``seat`` is to act in the live hand, fold it (a player who stood up)."""
        if self.actor_seat() == seat:
            self.apply(Action.fold())
            return True
        return False

    def amount(self, chips: int) -> str:
        """Engine chips in the game's money units, e.g. 250 chips at blinds 1/2 -> "5"."""
        return fmt_money(chips * self.money["bb"] / BB)

    def describe(self, e: PlayerAction) -> str:
        if e.type == ActionType.FOLD:
            return "Fold"
        if e.type == ActionType.CHECK_CALL:
            if e.amount == 0:
                return "Check"
            return "All-in" if e.is_all_in else f"Call {self.amount(e.amount)}"
        if e.is_all_in:
            return f"All-in {self.amount(e.to)}"
        return f"{'Bet' if e.to_call == 0 else 'Raise to'} {self.amount(e.to)}"

    def _note(self, text: str) -> None:
        self.log.append(text)

    # ------------------------------------------------------------- reads

    def _reads(self) -> dict[int, dict]:
        """The reader's current read of every non-PokerBot seat in the current composition."""
        out = {}
        for j, r in self.reader.reads().items():
            seat = self.hand.seats[j]
            if isinstance(self.seats[seat].policy, EngineSeat):
                continue
            probs = dict(sorted(r.probs.items(), key=lambda kv: -kv[1]))
            out[seat] = {
                "style": r.style,
                "confidence": r.confidence,
                "hands": r.hands,
                "probs": probs,
            }
        return out

    def stats(self, seat: int) -> list[dict]:
        """This seat's observed rates next to every full-ring style's average."""
        j = self.local.get(seat)
        counts = self.reader.tracker.features(j).counts if j is not None else {}
        prof = fr_profiles()
        out = []
        for key, label in STATS:
            n, x = counts.get(key, (0, 0))
            col = prof.features.index(key)
            out.append(
                {
                    "key": key,
                    "label": label,
                    "n": int(n),
                    "x": int(x),
                    "rate": x / n if n else None,
                    "styles": {
                        name: float(prof.pooled_rate[k, col]) for k, name in enumerate(prof.names)
                    },
                }
            )
        return out

    def read_detail(self, seat: int) -> dict | None:
        r = self.reads.get(seat)
        if r is None:
            return None
        return {
            **r,
            "stats": self.stats(seat),
            "trajectory": self.trajectory.get(seat, []),
            "about": STYLE_INFO.get(r["style"], ""),
        }

    # ------------------------------------------------------------- views

    def game_info(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "target": self.target,
            "hands": self.hand_no,
            "finished": self.finished,
            "started": self.started,
        }

    def view(self, client: str | None) -> dict:
        """Everything one client may see."""
        me = self.seat_of(client)
        h = self.hand
        st = h.state if h else None
        local = {t: j for j, t in enumerate(h.seats)} if h else {}
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
                    "hands": s.hands,
                    "read": self.reads.get(i),
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
        humans = [i for i, s in enumerate(self.seats) if s.kind == "human" and s.client]
        return {
            "type": "state",
            "game": self.game_info(),
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
            "log": self.log[-60:],
            "bigBlind": BB,
            "money": self.money,
            "styles": [s["name"] for s in self.sampler.styles],
            "myRead": self.read_detail(me) if me is not None else None,
            # at the end of a read challenge everyone sees every human's read
            "finalReads": {str(i): self.read_detail(i) for i in humans} if self.finished else None,
        }

    # ------------------------------------------------------------- the record

    def record(self) -> dict:
        """The whole game, for history: settings, players, results, reads, log."""
        players = []
        for i, s in enumerate(self.seats):
            if s.kind == "empty" or (s.kind == "human" and not s.client and not s.hands):
                continue
            players.append(
                {
                    "seat": i,
                    "name": s.name,
                    "kind": s.kind,
                    "trueStyle": s.style_name or None,
                    "net": s.net,
                    "hands": s.hands,
                    "read": self.reads.get(i),
                    "stats": self.stats(i) if i in self.local else None,
                    "trajectory": self.trajectory.get(i),
                }
            )
        return {
            **self.game_info(),
            "ended": self.ended,
            "seed": self.seed,
            "bigBlind": BB,
            "money": self.money,
            "setup": self.setup,
            "players": players,
            "styleInfo": STYLE_INFO,
            "log": self.log,
        }


def fmt_money(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def parse_setup(setup: dict | None) -> dict:
    """A validated table setup: 2-8 players, seat 1 a person, each other seat one of
    ``KINDS`` (a calibrated bot may name its style), blinds and buy-in in any money units."""
    if not setup:
        return {**DEFAULT_SETUP, "seats": [dict(x) for x in DEFAULT_SETUP["seats"]]}
    players = int(setup.get("players", 8))
    if not 2 <= players <= SEATS:
        raise ValueError(f"a table has 2 to {SEATS} players")
    given = list(setup.get("seats") or [])
    seats = []
    for i in range(SEATS):
        x = given[i] if i < len(given) and given[i] else {}
        kind = "human" if i == 0 else str(x.get("kind", "pokerbot")) if i < players else "empty"
        if kind not in KINDS:
            raise ValueError(f"unknown seat kind {kind!r}")
        style = x.get("style")
        style = int(style) if kind == "bot" and style not in (None, "") else None
        seats.append({"kind": kind, "style": style})
    sb, bb, buyin = (float(setup.get(k, DEFAULT_SETUP[k])) for k in ("sb", "bb", "buyin"))
    if not (0 < sb <= bb and bb < 1e7):
        raise ValueError("blinds must be positive, with the small blind at most the big blind")
    if not 10 <= buyin / bb <= 1000:
        raise ValueError("the buy-in must be 10 to 1000 big blinds")
    return {"players": players, "seats": seats, "sb": sb, "bb": bb, "buyin": buyin}


def parse_action(msg: dict) -> Action:
    kind = msg.get("action")
    if kind == "fold":
        return Action.fold()
    if kind in ("check", "call"):
        return Action.check_call()
    if kind == "raise":
        return Action.raise_to(int(msg.get("amount", 0)))
    raise ValueError(f"unknown action {kind!r}")


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
