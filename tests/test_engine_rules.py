"""Targeted rule tests with stacked decks and scripted players."""

from __future__ import annotations

import numpy as np

from pokerbot.engine import Action, play_hand, play_session
from pokerbot.engine.cards import DECK_SIZE, cards
from pokerbot.engine.events import Event, HoleCards, Muck, Showdown
from pokerbot.engine.policies import RandomPolicy
from pokerbot.engine.state import DecisionState, HandState

C = Action.check_call
F = Action.fold
R = Action.raise_to


def stacked_deck(holes: list[str], board: str) -> list[int]:
    """Deck whose seat i gets holes[i] and whose runout is board."""
    top = [c for h in holes for c in cards(h)] + list(cards(board))
    return top + [c for c in range(DECK_SIZE) if c not in top]


def run(state: HandState, script: list[Action]) -> HandState:
    for action in script:
        state.apply(action)
    assert state.finished
    return state


def reveals(state: HandState) -> list[tuple[str, int]]:
    out = []
    for e in state.events:
        if isinstance(e, Showdown):
            out.append(("show", e.seat))
        elif isinstance(e, Muck):
            out.append(("muck", e.seat))
    return out


# Heads-up, button 0. Preflop: button limps, BB checks. Postflop seat 1 acts first.
CHECK_TO_RIVER = [C(), C(), C(), C(), C(), C()]


def test_river_bettor_shows_first_and_better_hand_shows() -> None:
    deck = stacked_deck(["AhAd", "7c2s"], "Ks9d5h3cJs")
    state = run(HandState([200, 200], 0, deck), [*CHECK_TO_RIVER, R(10), C()])
    assert reveals(state) == [("show", 1), ("show", 0)]
    assert state.payoffs == (12, -12)


def test_caller_with_worse_hand_mucks() -> None:
    deck = stacked_deck(["7c2s", "AhAd"], "Ks9d5h3cJs")
    state = run(HandState([200, 200], 0, deck), [*CHECK_TO_RIVER, R(10), C()])
    assert reveals(state) == [("show", 1), ("muck", 0)]
    assert state.payoffs == (-12, 12)


def test_checked_river_first_left_of_button_shows_first() -> None:
    # 3-handed, button 0: seat 1 is first left of the button.
    deck = stacked_deck(["AhAd", "7c2s", "KhKd"], "Qs9d5h3cJs")
    script = [C(), C(), C()] + [C()] * 9  # preflop limp/complete/check, then check down
    state = run(HandState([200, 200, 200], 0, deck), script)
    assert reveals(state) == [("show", 1), ("show", 2), ("show", 0)]


def test_tie_is_shown_not_mucked() -> None:
    deck = stacked_deck(["Ah2c", "Ad3c"], "KsKdKhQsQd")  # both play the board
    state = run(HandState([200, 200], 0, deck), [*CHECK_TO_RIVER, R(10), C()])
    assert reveals(state) == [("show", 1), ("show", 0)]
    assert state.payoffs == (0, 0)


def test_all_in_tables_every_hand() -> None:
    deck = stacked_deck(["7c2s", "AhAd", "8c3d"], "Ks9d5h4cJs")
    # Button 0 shoves, SB and BB call; the board runs out with no further action.
    state = run(HandState([200, 200, 200], 0, deck), [R(200), C(), C()])
    assert sorted(seat for kind, seat in reveals(state) if kind == "show") == [0, 1, 2]
    assert state.payoffs == (-200, 400, -200)


def test_uncalled_bet_is_returned() -> None:
    deck = stacked_deck(["AhAd", "7c2s"], "Ks9d5h3cJs")
    state = run(HandState([200, 200], 0, deck), [R(200), F()])
    assert state.payoffs == (2, -2)
    assert not state.showdown


def test_big_blind_gets_option_after_limps() -> None:
    state = HandState([200, 200, 200], 0, stacked_deck(["AhAd", "7c2s", "KhKd"], "Qs9d5h3cJs"))
    state.apply(C())  # button (UTG 3-handed) limps
    state.apply(C())  # SB completes
    assert state.actor == 2
    legal = state.legal()
    assert legal.to_call == 0 and legal.can_raise


def test_short_all_in_raise_does_not_reopen_betting() -> None:
    # Button 0 opens to 20 (full raise of 18). SB (25 chips) shoves to 25: only +5.
    state = HandState([200, 25, 200], 0, stacked_deck(["AhAd", "7c2s", "KhKd"], "Qs9d5h3cJs"))
    state.apply(R(20))
    state.apply(R(25))
    assert state.actor == 2 and state.legal().can_raise  # BB hasn't acted: may raise
    state.apply(C())
    assert state.actor == 0
    legal = state.legal()
    assert legal.to_call == 5 and not legal.can_raise  # already acted: call or fold only


def test_min_raise_tracks_last_full_raise() -> None:
    state = HandState([500, 500], 0, stacked_deck(["AhAd", "7c2s"], "Qs9d5h3cJs"))
    state.apply(R(10))  # raise of 8
    assert state.legal().min_raise_to == 18
    state.apply(R(40))  # raise of 30
    assert state.legal().min_raise_to == 70


class Recorder:
    """Policy that records what it is shown."""

    def __init__(self, seed: int) -> None:
        self.inner = RandomPolicy(seed)
        self.events: list[Event] = []
        self.holes: set[tuple[int, int]] = set()

    def act(self, state: DecisionState) -> Action:
        self.holes.add(state.hole)
        return self.inner.act(state)

    def observe(self, event: Event) -> None:
        self.events.append(event)


class Spectator:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def observe(self, event: Event) -> None:
        self.events.append(event)


def test_players_only_see_their_own_hole_cards() -> None:
    for seed in range(200):
        players = [Recorder(seed * 10 + k) for k in range(4)]
        spectator = Spectator()
        result = play_hand(players, [200] * 4, seed % 4, seed, observers=[spectator])
        dealt = {e.seat: e.cards for e in result.events if isinstance(e, HoleCards)}
        for seat, p in enumerate(players):
            private = [e for e in p.events if isinstance(e, HoleCards)]
            assert [e.seat for e in private] == [seat]
            assert p.holes <= {dealt[seat]}
        assert not any(isinstance(e, HoleCards) for e in spectator.events)


def test_sessions_are_deterministic() -> None:
    def session() -> np.ndarray:
        players = [RandomPolicy(k) for k in range(6)]
        return play_session(players, 300, seed=7).per_hand

    np.testing.assert_array_equal(session(), session())


def test_chips_are_conserved() -> None:
    result = play_session([RandomPolicy(k) for k in range(6)], 2000, seed=3)
    assert result.per_hand.sum(axis=1).tolist() == [0] * 2000
