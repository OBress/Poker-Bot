"""Per-hand stat counts from an event stream (definitions: docs/MATH.md §1–§2).

``hand_counts(events)`` returns, for every seat, a dict ``stat -> (opportunities, occurrences)``
for one hand. Each stat is counted at most once per player per hand (its first opportunity).
Works on simulated hands and on real hands replayed by ``data.replay``: both use the same events.

Stat keys:
  hands
  vpip, pfr, vpip_<POS>, pfr_<POS>, vpip_d<D>, pfr_d<D>
  rfi, rfi_<POS>, limp, steal, fold_to_steal, 3bet, fold_to_3bet, 4bet
  cbet_<st>, cbet_<st>_hu|mw, cbet_<st>_ip|oop, fold_to_cbet_<st>, donk_<st>, xr_<st>, agg_<st>
  wtsd, wsd, betsize_b<k>
with POS in EP, MP, CO, BTN, SB, BB; D a stack-depth bucket 0..3; st in flop, turn, river.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pokerbot.engine.actions import ActionType, Street
from pokerbot.engine.events import (
    UNKNOWN_STACK_BB,
    BlindPosted,
    BoardDealt,
    Event,
    HandEnd,
    HandStart,
    Muck,
    PlayerAction,
    Showdown,
)

Counts = dict[str, tuple[int, int]]

STREET_NAMES = {Street.FLOP: "flop", Street.TURN: "turn", Street.RIVER: "river"}
DEPTH_EDGES = (25, 60, 150)  # effective stack in big blinds
BETSIZE_EDGES = (0.25, 0.4, 0.6, 0.85, 1.1, 1.6, 2.5)  # bet / pot
STEAL_POSITIONS = {"CO", "BTN", "SB"}


def position_name(seat: int, button: int, n: int) -> str:
    """Position class of a seat (MATH.md §1.1)."""
    r = (seat - button) % n
    if n == 2:
        return "BTN" if r == 0 else "BB"
    if r == 0:
        return "BTN"
    if r == 1:
        return "SB"
    if r == 2:
        return "BB"
    if r == n - 1:
        return "CO"
    if r == n - 2 or (n >= 8 and r == n - 3):
        return "MP"
    return "EP"


def _bucket(x: float, edges: Sequence[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


class _Acc:
    """Records each stat's first opportunity per seat."""

    def __init__(self) -> None:
        self.counts: dict[int, Counts] = defaultdict(dict)

    def opp(self, seat: int, stat: str, hit: bool) -> None:
        c = self.counts[seat]
        if stat not in c:
            c[stat] = (1, int(hit))


def hand_counts(events: Sequence[Event]) -> dict[int, Counts]:
    start = next(e for e in events if isinstance(e, HandStart))
    n, button, bb = len(start.stacks), start.button, start.big_blind
    acc = _Acc()
    pos = {s: position_name(s, button, n) for s in range(n)}
    order = {s: (s - button - 1) % n for s in range(n)}  # postflop acting order
    actions = [e for e in events if isinstance(e, PlayerAction)]
    pre = [a for a in actions if a.street == Street.PREFLOP]
    acted_pre = {a.seat for a in pre}

    for s in range(n):
        acc.opp(s, "hands", True)
        if s in acted_pre:  # had a preflop decision (excludes walks and blind all-ins)
            others = max((st for j, st in enumerate(start.stacks) if j != s), default=0)
            effective = min(start.stacks[s], others) / bb
            known = effective < UNKNOWN_STACK_BB  # some sites didn't record stacks
            vpip = any(
                a.seat == s and (a.type == ActionType.BET_RAISE or a.amount > 0) for a in pre
            )
            pfr = any(a.seat == s and a.type == ActionType.BET_RAISE for a in pre)
            suffixes = ["", f"_{pos[s]}"]
            if known:
                suffixes.append(f"_d{_bucket(effective, DEPTH_EDGES)}")
            for suffix in suffixes:
                acc.opp(s, "vpip" + suffix, vpip)
                acc.opp(s, "pfr" + suffix, pfr)

    # ---- preflop sequence
    raises = 0
    limpers = 0
    callers_since_raise = 0
    opener: int | None = None
    opener_pos: str | None = None
    last_raiser: int | None = None
    for a in pre:
        p = a.seat
        is_raise = a.type == ActionType.BET_RAISE
        is_call = a.type == ActionType.CHECK_CALL and a.amount > 0
        is_fold = a.type == ActionType.FOLD
        if raises == 0 and pos[p] != "BB":
            acc.opp(p, "limp", is_call)
            if limpers == 0:
                acc.opp(p, "rfi", is_raise)
                acc.opp(p, f"rfi_{pos[p]}", is_raise)
                if pos[p] in STEAL_POSITIONS:
                    acc.opp(p, "steal", is_raise)
        if (
            raises == 1
            and callers_since_raise == 0
            and limpers == 0
            and pos[p] in ("SB", "BB")
            and opener_pos in STEAL_POSITIONS
            and opener != p
        ):
            acc.opp(p, "fold_to_steal", is_fold)
        if raises == 1 and p != opener:
            acc.opp(p, "3bet", is_raise)
        if raises == 2:
            if p == opener:
                acc.opp(p, "fold_to_3bet", is_fold)
            acc.opp(p, "4bet", is_raise)

        if is_raise:
            raises += 1
            callers_since_raise = 0
            last_raiser = p
            if raises == 1:
                opener, opener_pos = p, pos[p]
        elif is_call:
            if raises == 0:
                limpers += 1
            else:
                callers_since_raise += 1

    # ---- postflop
    folded = {a.seat for a in pre if a.type == ActionType.FOLD}
    saw_flop = [s for s in acted_pre | _blind_only(events, pre) if s not in folded]
    flop_dealt = any(isinstance(e, BoardDealt) and e.street == Street.FLOP for e in events)
    pot_type = "hu" if len(saw_flop) == 2 else "mw"
    aggressor = last_raiser
    if flop_dealt:
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            st = STREET_NAMES[street]
            acts = [a for a in actions if a.street == street]
            if not acts:
                if not any(isinstance(e, BoardDealt) and e.street == street for e in events):
                    break
                continue
            live = sorted({a.seat for a in acts}, key=order.__getitem__)
            last_to_act = live[-1] if live else None
            bet_seen = False
            cbet_live = False
            checked: set[int] = set()
            street_raiser: int | None = None
            for a in acts:
                p = a.seat
                is_bet = a.type == ActionType.BET_RAISE
                acc.opp(p, f"agg_{st}", is_bet)
                if not bet_seen:
                    if p == aggressor:
                        acc.opp(p, f"cbet_{st}", is_bet)
                        acc.opp(p, f"cbet_{st}_{pot_type}", is_bet)
                        ip = "ip" if p == last_to_act else "oop"
                        acc.opp(p, f"cbet_{st}_{ip}", is_bet)
                        cbet_live = is_bet
                    elif (
                        aggressor is not None and aggressor in live and order[p] < order[aggressor]
                    ):
                        acc.opp(p, f"donk_{st}", is_bet)
                    if is_bet and a.pot_before > 0:
                        size = a.to / a.pot_before  # first bet: to == bet size
                        b = _bucket(size, BETSIZE_EDGES)
                        for k in range(len(BETSIZE_EDGES) + 1):
                            acc.opp(p, f"betsize_b{k}", k == b)
                else:
                    if cbet_live and p != aggressor:
                        acc.opp(p, f"fold_to_cbet_{st}", a.type == ActionType.FOLD)
                    if p in checked:
                        acc.opp(p, f"xr_{st}", is_bet)
                if a.type == ActionType.CHECK_CALL and a.to_call == 0 and not bet_seen:
                    checked.add(p)
                if is_bet:
                    if bet_seen:
                        cbet_live = False  # a raise ends "facing the c-bet"
                    bet_seen = True
                    street_raiser = p
            aggressor = street_raiser  # c-bet on the next street needs aggression on this one

    # ---- showdown
    end = next(e for e in events if isinstance(e, HandEnd))
    at_showdown = {e.seat for e in events if isinstance(e, (Showdown, Muck))}
    all_folded = {a.seat for a in actions if a.type == ActionType.FOLD}
    if end.showdown:
        at_showdown |= {s for s in saw_flop if s not in all_folded}
    if flop_dealt:
        for s in saw_flop:
            acc.opp(s, "wtsd", s in at_showdown)
    for s in at_showdown:
        acc.opp(s, "wsd", end.payoffs[s] > 0)  # net winner of the hand
    return dict(acc.counts)


def _blind_only(events: Sequence[Event], pre: Sequence[PlayerAction]) -> set[int]:
    """Seats in the hand with no preflop decision who still reach the flop (all-in blinds)."""
    acted = {a.seat for a in pre}
    return {e.seat for e in events if isinstance(e, BlindPosted) and e.seat not in acted}
