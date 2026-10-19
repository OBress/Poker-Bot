"""A parametric rule-based player. One policy, many styles: behaviour is set by a parameter vector.

The bot sees only what any player sees (``DecisionState``: own cards, public actions), never other
players' cards. Hand strength comes from ``strength``: a preflop percentile (0 = best) and a
postflop equity estimate vs one random hand, raised to the number of live opponents.

Parameters (see ``PARAMS`` for bounds):

Preflop (widths are fractions of all hands, best first)
  open_ep, open_btn   first-in raise width at EP and BTN; MP/CO interpolated, SB = open_btn*sb_open
  limp_width          extra width limped (first in, or behind limpers) on top of the raise range
  three_bet           width re-raised facing a single raise
  call_width          extra width called facing a single raise
  defend              multiplier on 3-bet/call widths in the blinds facing a steal
  cont_vs_3bet        fraction of own opening range that continues facing a 3-bet
  four_bet            fraction of own opening range that 4-bets
  open_size           open raise size in big blinds (+1 bb per limper)
Postflop
  value_eq            equity above which a hand is played for value
  value_bet           probability of betting a value hand when checked to
  bluff               probability of betting a non-value hand when checked to
  cbet_bonus          extra betting probability as the previous street's aggressor
  donk                multiplier on leading out into the previous aggressor
  bet_size            bet size as a fraction of the pot
  call_base, call_strict  equity needed to call = call_base + pot odds * call_strict
                      (low = calling station). Equity is vs a random hand, which overrates hands
                      facing a bet, so the base requirement is substantial. late_call_base
                      replaces call_base on the turn and river (players who called the flop
                      are stickier later: real fold-to-c-bet ~50% flop vs ~35% turn).
  respect             extra equity needed to call a c-bet (the first bet of a street, made by the
                      previous street's aggressor). Real players fold more to a raiser's c-bet
                      than to a stab, which one call threshold can't express.
  commit              pot commitment: if calling would leave at most commit x (pot after the
                      call) behind, the bot calls whenever its equity beats the pot odds
                      (short stacks get it in instead of folding the turn).
  raise_eq            equity above which the bot raises facing a bet (with prob raise_value)
  raise_value         probability of raising a raise_eq hand instead of calling
  bluff_raise         probability of raising a hand that would otherwise fold
  check_raise         multiplier on raising after having checked this street
Other
  noise               probability of replacing the decision with a random legal action. In an
                      unopened preflop pot the random action is fold or raise, never a limp:
                      limping is set by limp_width alone.
  buyin               stack in big blinds the bot sits with; not calibrated, sampled per seat from
                      the style's real stack-depth distribution. At <= 25 bb raises become shoves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.events import PlayerAction
from pokerbot.engine.state import DecisionState
from pokerbot.opponents.strength import postflop_equity, preflop_percentile
from pokerbot.stats.hand_stats import position_name

# name: (low, high, default)
PARAMS: dict[str, tuple[float, float, float]] = {
    "open_ep": (0.01, 0.6, 0.12),
    "open_btn": (0.02, 0.9, 0.40),
    "sb_open": (0.2, 1.2, 0.8),
    "limp_width": (0.0, 0.6, 0.02),
    "three_bet": (0.0, 0.25, 0.05),
    "call_width": (0.0, 0.5, 0.08),
    "defend": (0.5, 10.0, 1.3),
    "cont_vs_3bet": (0.05, 1.0, 0.45),
    "four_bet": (0.0, 0.5, 0.12),
    "open_size": (2.0, 5.0, 3.0),
    "value_eq": (0.35, 0.9, 0.6),
    "value_bet": (0.1, 1.0, 0.75),
    "bluff": (0.0, 0.8, 0.2),
    "cbet_bonus": (0.0, 1.0, 0.4),
    "donk": (0.0, 1.5, 0.5),
    "bet_size": (0.25, 1.5, 0.66),
    "call_base": (0.0, 0.7, 0.35),
    "late_call_base": (0.0, 0.7, 0.35),
    "call_strict": (0.0, 2.5, 1.0),
    "respect": (0.0, 0.3, 0.03),
    "commit": (0.0, 1.5, 0.3),
    "raise_eq": (0.5, 0.98, 0.8),
    "raise_value": (0.0, 1.0, 0.5),
    "bluff_raise": (0.0, 0.3, 0.04),
    "check_raise": (0.0, 3.0, 1.0),
    "noise": (0.0, 0.2, 0.02),
    "buyin": (10.0, 150.0, 100.0),
}

POSITION_INDEX = {"EP": 0, "MP": 1, "CO": 2, "BTN": 3}
# Rule-set version. Bots files record the structure they were calibrated with, so a frozen file
# keeps its behaviour: 2 = bots-v1/v2 (no respect/commit, noise may limp), 3 = current.
STRUCTURE = 3
STRUCTURE_PARAMS = {2: tuple(k for k in PARAMS if k not in ("respect", "commit", "buyin"))}
SHOVE_BB = 25


def default_params() -> dict[str, float]:
    return {k: v[2] for k, v in PARAMS.items()}


@dataclass
class _Preflop:
    raises: int  # raises before this decision (blinds don't count)
    limpers: int
    opener: int | None
    opener_pos: str | None
    my_raises: int


def _preflop_context(state: DecisionState) -> _Preflop:
    raises = limpers = my_raises = 0
    opener = opener_pos = None
    n = state.n_players
    for a in state.actions:
        if a.street != Street.PREFLOP:
            break
        if a.type == ActionType.BET_RAISE:
            raises += 1
            if raises == 1:
                opener = a.seat
                opener_pos = position_name(a.seat, state.button, n)
            if a.seat == state.seat:
                my_raises += 1
        elif a.type == ActionType.CHECK_CALL and a.amount > 0 and raises == 0:
            limpers += 1
    return _Preflop(raises, limpers, opener, opener_pos, my_raises)


def _previous_aggressor(state: DecisionState) -> int | None:
    """Last bettor/raiser on the previous street."""
    prev = Street(state.street - 1)
    last = None
    for a in state.actions:
        if a.street == prev and a.type == ActionType.BET_RAISE:
            last = a.seat
    return last


def _acted_this_street(state: DecisionState) -> list[PlayerAction]:
    return [a for a in state.actions if a.street == state.street]


class StyleBot:
    """Rule-based player with style parameters. Deterministic given its RNG seed."""

    def __init__(
        self,
        params: dict[str, float],
        seed: int | np.random.Generator,
        name: str = "",
        structure: int = STRUCTURE,
    ) -> None:
        self.p = {**default_params(), **params}
        self.structure = structure
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.name = name

    @property
    def buyin_bb(self) -> float:
        return self.p["buyin"]

    # ------------------------------------------------------------------ entry

    def act(self, state: DecisionState) -> Action:
        if self.rng.random() < self.p["noise"]:
            return self._random_action(state)
        if state.street == Street.PREFLOP:
            return self._preflop(state)
        return self._postflop(state)

    # ------------------------------------------------------------------ helpers

    def _raise(self, state: DecisionState, to: float) -> Action:
        legal = state.legal
        if not legal.can_raise:
            return Action.check_call()
        stack_total = state.stacks[state.seat] + state.bets[state.seat]
        if stack_total <= SHOVE_BB * state.big_blind or to >= 0.4 * stack_total:
            return Action.raise_to(legal.max_raise_to)  # short or committed: shove
        return Action.raise_to(int(min(max(to, legal.min_raise_to), legal.max_raise_to)))

    def _fold_or_check(self, state: DecisionState) -> Action:
        return Action.check_call() if state.legal.to_call == 0 else Action.fold()

    def _random_action(self, state: DecisionState) -> Action:
        u = self.rng.random()
        unopened = state.street == Street.PREFLOP and _preflop_context(state).raises == 0
        no_limp = self.structure >= 3 and unopened and state.legal.to_call > 0
        if u < 0.33 or (u < 0.66 and no_limp):
            return self._fold_or_check(state)  # no random limps
        if u < 0.66 or not state.legal.can_raise:
            return Action.check_call()
        return self._raise(state, state.legal.min_raise_to)

    # ------------------------------------------------------------------ preflop

    def _open_width(self, pos: str) -> float:
        p = self.p
        if pos == "SB":
            return p["open_btn"] * p["sb_open"]
        i = POSITION_INDEX.get(pos)
        if i is None:  # BB never opens
            return p["open_btn"]
        return p["open_ep"] + (p["open_btn"] - p["open_ep"]) * i / 3

    def _preflop(self, state: DecisionState) -> Action:
        p = self.p
        bb = state.big_blind
        pct = preflop_percentile(state.hole)
        pos = position_name(state.seat, state.button, state.n_players)
        ctx = _preflop_context(state)
        current = max(state.bets)
        open_w = self._open_width(pos)

        if ctx.raises == 0:
            if pos == "BB":  # option: check or raise the limpers
                if pct < open_w * 0.5:
                    return self._raise(state, (p["open_size"] + ctx.limpers) * bb)
                return Action.check_call()
            # Over limpers the raising range tightens (isolation); limping is only ever the
            # explicit limp band just below the raising range, so tight styles raise or fold.
            raise_w = open_w if ctx.limpers == 0 else open_w * 0.7
            if pct < raise_w:
                return self._raise(state, (p["open_size"] + ctx.limpers) * bb)
            if pct < raise_w + p["limp_width"]:
                return Action.check_call()
            return self._fold_or_check(state)

        if ctx.raises == 1 and ctx.opener != state.seat:
            mult = 1.0
            if pos in ("SB", "BB") and ctx.opener_pos in ("CO", "BTN", "SB"):
                mult = p["defend"]
            if pct < p["three_bet"] * mult:
                return self._raise(state, 3 * current)
            if pct < (p["three_bet"] + p["call_width"]) * mult:
                return Action.check_call()
            return self._fold_or_check(state)

        if ctx.raises == 2 and ctx.opener == state.seat:
            if pct < open_w * p["four_bet"]:
                return self._raise(state, 2.3 * current)
            if pct < open_w * p["cont_vs_3bet"]:
                return Action.check_call()
            return self._fold_or_check(state)

        if ctx.raises == 2:  # cold facing a 3-bet
            if pct < 0.02:
                return self._raise(state, 2.3 * current)
            if pct < p["three_bet"] * 0.5:
                return Action.check_call()
            return self._fold_or_check(state)

        # 4-bet or more
        if pct < 0.025:
            return self._raise(state, state.legal.max_raise_to)
        if pct < 0.05:
            return Action.check_call()
        return self._fold_or_check(state)

    # ------------------------------------------------------------------ postflop

    def _postflop(self, state: DecisionState) -> Action:
        p = self.p
        legal = state.legal
        live_opps = sum(
            1 for s in range(state.n_players) if s != state.seat and not state.folded[s]
        )
        eq = postflop_equity(state.hole, state.board) ** max(1, live_opps)
        this_street = _acted_this_street(state)
        checked_before = any(
            a.seat == state.seat and a.type == ActionType.CHECK_CALL and a.to_call == 0
            for a in this_street
        )
        u = self.rng.random()

        if legal.to_call == 0:
            prob = p["value_bet"] if eq >= p["value_eq"] else p["bluff"] * (1 - eq)
            aggressor = _previous_aggressor(state)
            if aggressor == state.seat:
                prob += p["cbet_bonus"] * (1 - prob)
            elif aggressor is not None and not state.folded[aggressor]:
                order = lambda s: (s - state.button - 1) % state.n_players  # noqa: E731
                if order(state.seat) < order(aggressor):
                    prob *= p["donk"]
            if u < min(prob, 1.0):
                size = p["bet_size"] * float(np.exp(self.rng.normal(0, 0.2)))
                return self._raise(state, size * state.pot)
            return Action.check_call()

        pot_after_call = state.pot + legal.to_call
        pot_odds = legal.to_call / pot_after_call
        base = p["call_base"] if state.street == Street.FLOP else p["late_call_base"]
        if self.structure >= 3:
            aggressor = _previous_aggressor(state)
            bets = [a for a in this_street if a.type == ActionType.BET_RAISE]
            if len(bets) == 1 and bets[0].seat == aggressor:  # facing a c-bet
                base += p["respect"]
        need = min(0.95, base + pot_odds * p["call_strict"])
        behind = state.stacks[state.seat] - legal.to_call
        if self.structure >= 3 and behind <= p["commit"] * pot_after_call:
            need = min(need, pot_odds)  # pot-committed: pot odds alone decide
        raise_mult = p["check_raise"] if checked_before else 1.0
        current = max(state.bets)
        raise_to = current + p["bet_size"] * (state.pot + legal.to_call)
        if eq >= p["raise_eq"]:
            if u < min(1.0, p["raise_value"] * raise_mult):
                return self._raise(state, raise_to)
            return Action.check_call()
        if eq >= need:
            return Action.check_call()
        if u < min(1.0, p["bluff_raise"] * raise_mult):
            return self._raise(state, raise_to)
        return Action.fold()
