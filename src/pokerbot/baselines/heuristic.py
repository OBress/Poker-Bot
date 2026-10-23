"""Simple hold'em baselines (ARCHITECTURE §4.6): cheap, transparent, useful for debugging and as
reference points for the CFR blueprint.

- ``abc_tag``: a solid, noise-free tight-aggressive player built from ``StyleBot`` with fixed,
  hand-chosen parameters (not calibrated to any data): opens ~12% early / 40% on the button,
  3-bets ~6%, c-bets often, value-bets strong hands and folds marginal ones to pressure.
- ``HandStrengthBot``: decisions from hand strength alone. Preflop by percentile, postflop by
  equity vs the live opponents compared with fixed thresholds and the pot odds. Deterministic.
"""

from __future__ import annotations

from pokerbot.engine.actions import Action, ActionType, Street
from pokerbot.engine.state import DecisionState
from pokerbot.opponents.bot import StyleBot
from pokerbot.opponents.strength import postflop_equity, preflop_percentile

ABC_TAG: dict[str, float] = {
    "open_ep": 0.12, "open_btn": 0.40, "sb_open": 0.8, "limp_width": 0.0, "three_bet": 0.06,
    "call_width": 0.06, "defend": 1.5, "cont_vs_3bet": 0.4, "four_bet": 0.3, "open_size": 2.5,
    "value_eq": 0.65, "value_bet": 0.9, "bluff": 0.2, "bluff_eq": 0.35, "cbet_bonus": 0.5,
    "donk": 0.3, "bet_size": 0.66, "call_base": 0.3, "late_call_base": 0.35, "call_strict": 1.0,
    "respect": 0.05, "commit": 0.5, "raise_eq": 0.85, "raise_value": 0.7, "bluff_raise": 0.02,
    "check_raise": 1.0, "noise": 0.0, "buyin": 100.0,
}  # fmt: skip


def abc_tag(seed: int) -> StyleBot:
    return StyleBot(ABC_TAG, seed, "abc-tag")


class HandStrengthBot:
    """Raise strong, call medium, fold weak; sizes are fixed fractions of the pot."""

    buyin_bb = 100.0

    def __init__(
        self, raise_pct: float = 0.15, call_pct: float = 0.30, bet_eq: float = 0.6,
        raise_eq: float = 0.85, margin: float = 0.1,
    ) -> None:  # fmt: skip
        self.raise_pct, self.call_pct = raise_pct, call_pct
        self.bet_eq, self.raise_eq, self.margin = bet_eq, raise_eq, margin

    def _raise(self, state: DecisionState, to: float) -> Action:
        legal = state.legal
        if not legal.can_raise:
            return Action.check_call()
        return Action.raise_to(int(min(max(to, legal.min_raise_to), legal.max_raise_to)))

    def act(self, state: DecisionState) -> Action:
        legal = state.legal
        current = max(state.bets)
        if state.street == Street.PREFLOP:
            pct = preflop_percentile(state.hole)
            raises = sum(
                1
                for a in state.actions
                if a.street == Street.PREFLOP and a.type == ActionType.BET_RAISE
            )
            if pct < self.raise_pct / (1 + 2 * raises):  # tighter for each raise faced
                return self._raise(state, 3 * max(current, state.big_blind))
            if pct < self.call_pct / (1 + raises) or legal.to_call == 0:
                return Action.check_call()
            return Action.fold()
        live = sum(1 for s in range(state.n_players) if s != state.seat and not state.folded[s])
        eq = postflop_equity(state.hole, state.board) ** max(1, live)
        if legal.to_call == 0:
            if eq >= self.bet_eq:
                return self._raise(state, 0.66 * state.pot)
            return Action.check_call()
        pot_odds = legal.to_call / (state.pot + legal.to_call)
        if eq >= self.raise_eq:
            return self._raise(state, current + 0.75 * (state.pot + legal.to_call))
        if eq >= pot_odds + self.margin:
            return Action.check_call()
        return Action.fold()
