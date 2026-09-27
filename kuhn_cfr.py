"""
kuhn_cfr.py
===========
Vanilla Counterfactual Regret Minimization (CFR) solver for Kuhn Poker.

Kuhn Poker rules
----------------
- Deck: {J=0, Q=1, K=2}
- Both players ante 1 chip.
- Player 1 acts first: check (p) or bet (b).
- If P1 checks, P2 may check (terminal) or bet.
  - If P2 bets after P1 checks, P1 may fold (p) or call (b).
- If P1 bets, P2 may fold (p) or call (b).
- Showdown: higher card wins the pot.

Information sets are keyed by "<private_card><action_history>",
where actions are 'p' (pass/check/fold) and 'b' (bet/call).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARDS: List[int] = [0, 1, 2]   # J=0, Q=1, K=2
CARD_NAMES: Dict[int, str] = {0: "J", 1: "Q", 2: "K"}

# Terminal history set
TERMINAL_HISTORIES: frozenset = frozenset(["pp", "pbp", "pbb", "bp", "bb"])


# ---------------------------------------------------------------------------
# Payoff helper
# ---------------------------------------------------------------------------

def terminal_payoff(history: str, p1_card: int, p2_card: int) -> float:
    """
    Return the payoff to Player 1 (index 0) at a terminal history.

    Parameters
    ----------
    history:  action string, e.g. 'pp', 'pbp', 'pbb', 'bp', 'bb'
    p1_card:  Player 1's private card (0=J, 1=Q, 2=K)
    p2_card:  Player 2's private card (0=J, 1=Q, 2=K)

    Returns
    -------
    float: chips gained (+) or lost (-) by Player 1.
    """
    p1_wins = int(p1_card > p2_card)  # 1 if P1 wins showdown, 0 otherwise

    if history == "pp":
        # Pot = 2; winner takes 1 from loser
        return 1.0 if p1_wins else -1.0

    if history == "pbp":
        # P1 folded; P1 loses their ante (net -1)
        return -1.0

    if history == "pbb":
        # Pot = 4; winner takes 2 from loser
        return 2.0 if p1_wins else -2.0

    if history == "bp":
        # P2 folded; P1 wins P2's ante (net +1)
        return 1.0

    if history == "bb":
        # Pot = 4; winner takes 2 from loser
        return 2.0 if p1_wins else -2.0

    raise ValueError(f"Unknown terminal history: {history!r}")


# ---------------------------------------------------------------------------
# Information Set
# ---------------------------------------------------------------------------

class InfoSet:
    """
    Represents a single information set in the CFR tree.

    Attributes
    ----------
    key:          string identifier, e.g. 'Jpb'
    regret_sum:   cumulative counterfactual regrets [pass, bet]
    strategy_sum: cumulative weighted strategy [pass, bet]
    """

    NUM_ACTIONS: int = 2   # 0 = pass/check/fold, 1 = bet/call

    def __init__(self, key: str) -> None:
        self.key: str = key
        self.regret_sum: List[float] = [0.0, 0.0]
        self.strategy_sum: List[float] = [0.0, 0.0]

    # ------------------------------------------------------------------
    # Regret matching
    # ------------------------------------------------------------------

    def current_strategy(self, reach_prob: float) -> List[float]:
        """
        Compute the current mixed strategy via regret matching and update
        the cumulative strategy profile weighted by ``reach_prob``.

        Parameters
        ----------
        reach_prob: the reaching probability of the current player
                    at this information set (used to weight strategy_sum).

        Returns
        -------
        List[float]: mixed strategy [p(pass), p(bet)] summing to 1.
        """
        positive_regrets = [max(r, 0.0) for r in self.regret_sum]
        total = sum(positive_regrets)

        if total > 0.0:
            strategy = [r / total for r in positive_regrets]
        else:
            strategy = [1.0 / self.NUM_ACTIONS] * self.NUM_ACTIONS

        # Accumulate weighted strategy for average profile
        for a in range(self.NUM_ACTIONS):
            self.strategy_sum[a] += reach_prob * strategy[a]

        return strategy

    def average_strategy(self) -> List[float]:
        """
        Return the normalized average strategy (Nash approximation).

        Returns
        -------
        List[float]: average mixed strategy [p(pass), p(bet)] summing to 1.
        """
        total = sum(self.strategy_sum)
        if total > 0.0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / self.NUM_ACTIONS] * self.NUM_ACTIONS


# ---------------------------------------------------------------------------
# CFR Trainer
# ---------------------------------------------------------------------------

class KuhnCFRTrainer:
    """
    Vanilla CFR trainer for Kuhn Poker.

    Usage
    -----
    >>> trainer = KuhnCFRTrainer()
    >>> ev = trainer.train(iterations=10_000)
    >>> strategy = trainer.get_average_strategy()
    """

    def __init__(self) -> None:
        self.info_sets: Dict[str, InfoSet] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, iterations: int, seed: int = 42) -> float:
        """
        Run ``iterations`` rounds of vanilla CFR.

        Each iteration enumerates all 6 distinct card deals (|S3|=6 orderings
        of 3 cards taken 2 at a time), weighted by their probability, then
        recurses through the game tree with reach-probability weights.

        Parameters
        ----------
        iterations: number of CFR iterations
        seed:       random seed for reproducibility

        Returns
        -------
        float: average P1 expected value over all iterations.
        """
        total_ev: float = 0.0

        for _ in range(iterations):
            # Each deal is equally likely: 1/6 each
            for p1_card in CARDS:
                for p2_card in CARDS:
                    if p1_card == p2_card:
                        continue
                    # Weight by deal probability = 1 / (3*2) = 1/6
                    total_ev += (1.0 / 6.0) * self._cfr(
                        [p1_card, p2_card], history="", p0=1.0, p1=1.0
                    )

        return total_ev / iterations

    def get_average_strategy(self) -> Dict[str, List[float]]:
        """
        Return the average strategy profile for all visited information sets.

        Returns
        -------
        Dict mapping info-set key -> [p(pass), p(bet)].
        """
        return {key: iset.average_strategy()
                for key, iset in sorted(self.info_sets.items())}

    # ------------------------------------------------------------------
    # Internal: recursive CFR
    # ------------------------------------------------------------------

    def _get_info_set(self, card: int, history: str) -> InfoSet:
        """Retrieve or create the InfoSet for (card, history)."""
        key = CARD_NAMES[card] + history
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet(key)
        return self.info_sets[key]

    def _cfr(
        self,
        cards: List[int],
        history: str,
        p0: float,
        p1: float,
    ) -> float:
        """
        Recursive vanilla CFR.

        Returns the **expected payoff to Player 0** from this node,
        given the current strategy profile sigma.

        Parameters
        ----------
        cards:   [p1_card, p2_card]  (Player 0 = P1, Player 1 = P2)
        history: action string so far (e.g., '', 'p', 'pb', ...)
        p0:      reach probability of Player 0
        p1:      reach probability of Player 1

        Returns
        -------
        float: expected payoff to Player 0 from this node.
        """
        # -- Terminal node -----------------------------------------------
        if history in TERMINAL_HISTORIES:
            return terminal_payoff(history, cards[0], cards[1])

        # -- Whose turn? -------------------------------------------------
        # In Kuhn Poker:
        #   history ''   -> Player 0 acts (len 0, even)
        #   history 'p'  -> Player 1 acts (len 1, odd)
        #   history 'b'  -> Player 1 acts (len 1, odd)
        #   history 'pb' -> Player 0 acts (len 2, even)
        player = len(history) % 2

        info_set = self._get_info_set(cards[player], history)
        reach = p0 if player == 0 else p1
        strategy = info_set.current_strategy(reach_prob=reach)

        # -- Recurse over actions ----------------------------------------
        action_utils: List[float] = [0.0, 0.0]

        for a, action in enumerate(["p", "b"]):
            next_history = history + action
            if player == 0:
                action_utils[a] = self._cfr(
                    cards, next_history, p0 * strategy[a], p1
                )
            else:
                action_utils[a] = self._cfr(
                    cards, next_history, p0, p1 * strategy[a]
                )

        # -- Node value (Player 0's perspective) -------------------------
        node_value = sum(strategy[a] * action_utils[a] for a in range(2))

        # -- Update regrets ----------------------------------------------
        # Counterfactual regret for player i at infoset I, action a:
        #   r(I, a) = (opponent reach) * (action_util(a) - node_value)
        #
        # action_utils are always from Player 0's perspective.
        # Player 1 wants to MINIMIZE Player 0's value, so we negate.
        opponent_reach = p1 if player == 0 else p0
        for a in range(2):
            if player == 0:
                info_set.regret_sum[a] += opponent_reach * (
                    action_utils[a] - node_value
                )
            else:
                # From Player 1's perspective, negate (they minimize P0's util)
                info_set.regret_sum[a] += opponent_reach * (
                    -action_utils[a] + node_value
                )

        return node_value
