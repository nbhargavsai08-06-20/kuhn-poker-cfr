"""
test_cfr.py
===========
pytest unit tests for the Kuhn Poker CFR solver.

Run:
    pytest test_cfr.py -v
"""

from __future__ import annotations

import pytest
from kuhn_cfr import KuhnCFRTrainer
from main import exact_ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_trainer() -> KuhnCFRTrainer:
    """Train a single CFR instance shared across tests (module-scoped)."""
    trainer = KuhnCFRTrainer()
    trainer.train(iterations=20_000, seed=0)
    return trainer


@pytest.fixture(scope="module")
def strategy(trained_trainer: KuhnCFRTrainer):
    return trained_trainer.get_average_strategy()


# ---------------------------------------------------------------------------
# Test 1: All strategy vectors sum to 1.0
# ---------------------------------------------------------------------------

class TestStrategyNormalization:
    """Ensure the average strategy is a valid probability distribution."""

    def test_all_info_sets_sum_to_one(self, strategy) -> None:
        """Every info set's action probabilities must sum to 1.0."""
        for key, probs in strategy.items():
            total = sum(probs)
            assert abs(total - 1.0) < 1e-9, (
                f"Info set '{key}': probabilities sum to {total:.10f}, expected 1.0"
            )

    def test_no_negative_probabilities(self, strategy) -> None:
        """No probability should be negative."""
        for key, probs in strategy.items():
            for p in probs:
                assert p >= -1e-12, (
                    f"Info set '{key}': negative probability {p}"
                )


# ---------------------------------------------------------------------------
# Test 2: Dominated strategy — P2 with Queen should never call
# ---------------------------------------------------------------------------

class TestDominatedStrategy:
    """
    Player 2 holding a Jack (the weakest card) faces a strictly dominated
    action when deciding to call a bet: Jack loses to both Queen and King,
    so calling can never be profitable at equilibrium. P2 with Jack should
    always fold (pass) when facing a bet.

    Info set key: 'Jb'  (P2 holds J, P1 bet)
    Action index: 1 = call/bet
    Threshold:    < 0.01  (essentially never calls)

    Note: P2 with Queen calling ~1/3 of the time IS correct equilibrium
    play (mixed strategy to prevent P1 from always bluffing with J).
    """

    def test_p2_jack_call_freq_near_zero(self, strategy) -> None:
        """P2 should never call a bet when holding a Jack (strictly dominated)."""
        jb_key = "Jb"
        assert jb_key in strategy, (
            f"Expected info set '{jb_key}' not found in strategy. "
            f"Available keys: {sorted(strategy.keys())}"
        )
        p_call = strategy[jb_key][1]   # P(bet/call | J, facing bet)
        assert p_call < 0.01, (
            f"P2 call probability with Jack = {p_call:.6f}; expected < 0.01 "
            f"(Jack loses to all other cards -- calling is strictly dominated)"
        )


# ---------------------------------------------------------------------------
# Test 3: Expected value convergence
# ---------------------------------------------------------------------------

class TestExpectedValueConvergence:
    """
    Theoretical P1 EV for Kuhn Poker is -1/18 ≈ -0.05556.
    After 20,000 iterations the empirical EV should be within ±0.01.
    """

    THEORETICAL_EV: float = -1.0 / 18.0
    TOLERANCE: float = 0.01

    def test_ev_within_tolerance(self, trained_trainer: KuhnCFRTrainer) -> None:
        """P1 EV after 20k iterations must be within +-0.01 of -1/18."""
        # Use exact tree-based EV for noise-free measurement
        strategy = trained_trainer.get_average_strategy()
        ev = exact_ev(strategy)
        error = abs(ev - self.THEORETICAL_EV)
        assert error < self.TOLERANCE, (
            f"P1 EV = {ev:.6f}, theoretical = {self.THEORETICAL_EV:.6f}, "
            f"error = {error:.6f} exceeds tolerance {self.TOLERANCE}"
        )


# ---------------------------------------------------------------------------
# Test 4: Sanity — expected info sets exist
# ---------------------------------------------------------------------------

class TestInfoSetCoverage:
    """Verify that all 12 expected Kuhn Poker information sets are visited."""

    EXPECTED_INFO_SETS = {
        # P1's initial action
        "J", "Q", "K",
        # P2's response to P1 check
        "Jp", "Qp", "Kp",
        # P2 facing P1 bet
        "Jb", "Qb", "Kb",
        # P1's response to P2 bet (after P1 checked)
        "Jpb", "Qpb", "Kpb",
    }

    def test_all_info_sets_visited(self, strategy) -> None:
        """All 12 Kuhn Poker info sets should appear in the strategy."""
        missing = self.EXPECTED_INFO_SETS - set(strategy.keys())
        assert not missing, f"Missing info sets: {missing}"
