"""PokerBot, the project's engine, as a seat at the demo table.

- Heads-up hands use the evolved heads-up exploiter: the Phase 6/7 champion over the MCCFR
  blueprint, with the frozen heads-up classifier as its belief source, as evaluated in
  ``docs/RESULTS.md``.
- Multiway hands use a demo-only engine that the Phase 7 experiment did NOT evaluate. The
  evolved 6-max classifier reads every opponent live with the full-ring style profiles, and a
  tight-aggressive base strategy (``abc_tag``) is tilted toward the live opponents' weaknesses:
  bluff and steal more against folders, value-bet thinner and bluff less against callers.

The table tells the seat about every finished hand (``observe_hand``, public events only) and
resets it when the set of players changes, because both classifiers key their memory by seat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokerbot.baselines.heuristic import ABC_TAG
from pokerbot.engine.actions import Action
from pokerbot.engine.events import Event, HandStart, HoleCards, PlayerAction, Showdown
from pokerbot.engine.state import DecisionState
from pokerbot.evolve import hu, program
from pokerbot.evolve.api import HUHandRecord, StyleProfiles
from pokerbot.evolve.harness import FeatureTracker, redact
from pokerbot.evolve.hu_task import TEMPLATE as HU_TEMPLATE
from pokerbot.opponents.bot import StyleBot

ROOT = Path(__file__).resolve().parents[2]
HU_EXPLOITER = ROOT / "programs/exploiter_hu/champion-20261022-230930-evolve-exploiter-hu-2"
CLASSIFIER = ROOT / "programs/classifier/champion-20261021-064929-evolve-classifier-sh"
STYLES = ROOT / "profiles/styles-v1.json"
SHOWDOWNS = ROOT / "data/processed/showdown_ranges.json"

# full-ring style names (profiles/styles-v1.json, "fr") grouped by how they lose
CALLERS = ("Loose-passive", "Passive limper")
FOLDERS = ("Weak-tight", "Nit")


@lru_cache(maxsize=1)
def fr_profiles() -> StyleProfiles:
    return StyleProfiles.load(STYLES, "fr", SHOWDOWNS)


@lru_cache(maxsize=1)
def _classifier_program():
    src = CLASSIFIER.with_suffix(".py").read_text(encoding="utf-8")
    params = json.loads(CLASSIFIER.with_suffix(".json").read_text(encoding="utf-8"))["params"]
    return program.load(src)["Classifier"], params


@lru_cache(maxsize=1)
def _hu_program():
    src = HU_EXPLOITER.with_suffix(".py").read_text(encoding="utf-8")
    params = json.loads(HU_EXPLOITER.with_suffix(".json").read_text(encoding="utf-8"))["params"]
    return program.load(src, HU_TEMPLATE)["Exploiter"], params


@dataclass
class Read:
    """What PokerBot thinks of one opponent (shown over their head)."""

    style: str
    confidence: float
    hands: int
    probs: dict[str, float]


def tilted_params(callers: float, folders: float) -> dict[str, float]:
    """ABC-TAG parameters tilted by how much more the live opponents look like callers or
    folders than the population does (each in [-1, 1])."""
    c = float(np.clip(callers, -0.5, 0.5))
    f = float(np.clip(folders, -0.5, 0.5))
    p = dict(ABC_TAG)
    p["bluff"] = max(0.0, p["bluff"] * (1 + 3.0 * f - 2.0 * c))
    p["bluff_raise"] = max(0.0, p["bluff_raise"] * (1 + 3.0 * f - 2.0 * c))
    p["cbet_bonus"] = max(0.0, p["cbet_bonus"] + 0.8 * f - 0.6 * c)
    p["value_eq"] = float(np.clip(p["value_eq"] - 0.15 * c + 0.05 * f, 0.5, 0.8))
    p["bet_size"] = p["bet_size"] * (1 + 0.5 * c)
    p["open_btn"] = float(np.clip(p["open_btn"] + 0.3 * f, 0.2, 0.7))
    p["open_ep"] = float(np.clip(p["open_ep"] + 0.08 * f, 0.08, 0.25))
    p["three_bet"] = float(np.clip(p["three_bet"] + 0.08 * f, 0.03, 0.15))
    return p


class EngineSeat:
    """One PokerBot seat. ``act`` is called with hand-local seat indices; the table maps them."""

    buyin_bb = 100.0

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mode = "multiway"
        self.reset(n_players=0)

    # Saving a game (``Table`` is pickled to resume it later): the classifiers and the exploiter
    # are sandbox-loaded programs, which don't pickle, so the state is the hands seen since the
    # last reset and the RNG; loading rebuilds the programs and replays those hands.

    def __getstate__(self) -> dict:
        return {
            "seed": self.seed,
            "n": self.n,
            "history": self.history,
            "mode": self.mode,
            "rng": self.rng.bit_generator.state,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["seed"])
        self.reset(state["n"])
        for events, me in state["history"]:
            self.observe_hand(events, me)
        self.rng.bit_generator.state = state["rng"]
        self.mode = state["mode"]

    # ------------------------------------------------------------- memory

    def reset(self, n_players: int) -> None:
        """New table composition: forget everyone (classifier memory is keyed by seat)."""
        self.n = n_players
        self.history: list[tuple[list[Event], int]] = []  # hands seen since this reset
        clf, params = _classifier_program()
        self.clf = clf(fr_profiles(), dict(params))
        self.tracker = FeatureTracker()
        hu_clf, hu_params = hu._classifier()
        self.hu_clf = hu_clf(hu.profiles(), dict(hu_params))
        prog, prog_params = _hu_program()
        self.hu_seat = hu.ExploiterSeat(prog(hu.profiles(), dict(prog_params)), self.rng)

    def observe_hand(self, events: list[Event], me: int) -> None:
        """A finished hand, with hand-local seats (``me`` is this seat)."""
        self.history.append((list(events), me))
        public = redact(events)
        for e in public:
            self.clf.observe(e)
        self.tracker.add_hand(public)
        if len(next(e for e in events if isinstance(e, HandStart)).stacks) == 2:
            for e in public:
                self.hu_clf.observe(e)
            self.hu_seat.prog.observe_hand(_hu_record(events, me))

    def reads(self) -> dict[int, Read]:
        """Current read of every other hand-local seat: heads-up styles (and the blueprint) from
        the heads-up classifier at a 2-player table, full-ring styles otherwise."""
        prof, clf = (hu.profiles(), self.hu_clf) if self.n == 2 else (fr_profiles(), self.clf)
        names = prof.names
        out = {}
        for s in range(self.n):
            f = self.tracker.features(s)
            try:
                p = np.asarray(clf.posterior(s, f), dtype=float).reshape(-1)
                p = p / p.sum() if p.shape == (len(names),) and p.sum() > 0 else None
            except Exception:  # noqa: BLE001 - an odd read never stops the game
                p = None
            if p is None:
                p = prof.shares
            k = int(p.argmax())
            out[s] = Read(
                names[k], float(p[k]), f.hands, dict(zip(names, map(float, p), strict=True))
            )
        return out

    # ------------------------------------------------------------- decisions

    def act(self, state: DecisionState) -> Action:
        if state.n_players == 2:
            self.mode = "heads-up exploiter"
            opp = 1 - state.seat
            feats = self.tracker.features(opp)
            b = self.hu_clf.posterior(opp, feats)
            b, _ = hu._valid(b, hu.profiles().k)
            self.hu_seat.beliefs = b
            self.hu_seat.opp_feats = feats
            return self.hu_seat.act(state)
        self.mode = "multiway"
        reads = self.reads()
        live = [s for s in range(state.n_players) if s != state.seat and not state.folded[s]]
        shares = fr_profiles().shares
        names = fr_profiles().names
        idx_c = [names.index(n) for n in CALLERS]
        idx_f = [names.index(n) for n in FOLDERS]
        callers = folders = 0.0
        for s in live:
            p = np.array([reads[s].probs[n] for n in names])
            callers += float(p[idx_c].sum() - shares[idx_c].sum())
            folders += float(p[idx_f].sum() - shares[idx_f].sum())
        if live:
            callers, folders = callers / len(live), folders / len(live)
        bot = StyleBot(tilted_params(callers, folders), int(self.rng.integers(2**31)), "pokerbot")
        return bot.act(state)


def _hu_record(events: list[Event], me: int) -> HUHandRecord:
    start = next(e for e in events if isinstance(e, HandStart))
    hole = next(e.cards for e in events if isinstance(e, HoleCards) and e.seat == me)
    shows = {e.seat: e.cards for e in events if isinstance(e, Showdown)}
    boards = [e.board for e in events if hasattr(e, "board")]
    acts = [e for e in events if isinstance(e, PlayerAction)]
    end = next(e for e in reversed(events) if hasattr(e, "payoffs"))
    return HUHandRecord(
        start.button == me,
        hole,
        boards[-1] if boards else (),
        shows.get(1 - me),
        hu._actions(acts, me),
        end.payoffs[me] / start.big_blind,
    )
