"""
main.py
=======
Train the Kuhn Poker CFR solver, display the equilibrium strategy table,
log convergence milestones, and save convergence_plot.png.

Run:
    python main.py
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from kuhn_cfr import KuhnCFRTrainer, terminal_payoff, CARDS, CARD_NAMES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THEORETICAL_EV: float = -1.0 / 18.0          # ~-0.05556

# Log-spaced milestones at which to snapshot the strategy / EV
MILESTONES: List[int] = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

# Terminal history set (used in exact EV computation)
_TERMINAL = frozenset(["pp", "pbp", "pbb", "bp", "bb"])


# ---------------------------------------------------------------------------
# Exact EV computation via tree traversal
# ---------------------------------------------------------------------------

def exact_ev(strategy: Dict[str, List[float]]) -> float:
    """
    Compute the exact expected payoff to Player 1 by traversing the full
    Kuhn Poker game tree using the *current average strategy*.

    This gives a clean, noise-free EV measurement for convergence logging.

    Parameters
    ----------
    strategy: average strategy dict (info_set_key -> [p_pass, p_bet])

    Returns
    -------
    float: P1 expected value.
    """
    total = 0.0
    # Enumerate all 6 equally likely deals
    for p1_card in CARDS:
        for p2_card in CARDS:
            if p1_card == p2_card:
                continue

            def recurse(hist: str, prob: float) -> float:
                if hist in _TERMINAL:
                    return prob * terminal_payoff(hist, p1_card, p2_card)
                player = len(hist) % 2
                card = p1_card if player == 0 else p2_card
                key = CARD_NAMES[card] + hist
                s = strategy.get(key, [0.5, 0.5])
                return recurse(hist + "p", prob * s[0]) + recurse(hist + "b", prob * s[1])

            total += (1.0 / 6.0) * recurse("", 1.0)
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bluff_freq(strategy: Dict[str, List[float]]) -> float:
    """P1's initial bluffing freq = P(bet | Jack, no prior action)."""
    return strategy.get("J", [0.5, 0.5])[1]


def print_strategy_table(strategy: Dict[str, List[float]]) -> None:
    """Pretty-print the converged strategy as a tabular summary."""
    header = f"{'Info Set':>10} | {'P(check/fold)':>14} | {'P(bet/call)':>12}"
    sep = "-" * len(header)
    print()
    print("  Nash Equilibrium Approximation -- Kuhn Poker")
    print(sep)
    print(header)
    print(sep)
    info_set_order = [
        "J", "Q", "K",
        "Jp", "Qp", "Kp",
        "Jpb", "Qpb", "Kpb",
        "Jb", "Qb", "Kb",
    ]
    for key in info_set_order:
        if key in strategy:
            p_pass, p_bet = strategy[key]
            print(f"{key:>10} | {p_pass:>14.4f} | {p_bet:>12.4f}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    trainer = KuhnCFRTrainer()

    ev_log: List[float] = []
    bluff_log: List[float] = []
    iter_log: List[int] = []

    prev = 0
    print(f"{'Iterations':>12} | {'P1 EV':>10} | {'EV error':>10} | {'Bluff(J)':>10}")
    print("-" * 52)

    for milestone in MILESTONES:
        # Run (milestone - prev) additional CFR iterations
        incremental = milestone - prev
        trainer.train(iterations=incremental)
        prev = milestone

        # Measure exact EV from the current average strategy (no noise)
        strategy = trainer.get_average_strategy()
        ev = exact_ev(strategy)
        bf = bluff_freq(strategy)
        err = ev - THEORETICAL_EV

        iter_log.append(milestone)
        ev_log.append(ev)
        bluff_log.append(bf)

        print(f"{milestone:>12,} | {ev:>10.6f} | {err:>+10.6f} | {bf:>10.4f}")

    # Final strategy table
    final_strategy = trainer.get_average_strategy()
    print_strategy_table(final_strategy)

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Kuhn Poker CFR Convergence", fontsize=14, fontweight="bold")

    x = np.array(iter_log)

    # --- Subplot 1: Expected Value ---
    ax0 = axes[0]
    ax0.semilogx(x, ev_log, color="#2563eb", linewidth=2, marker="o",
                 markersize=5, label="P1 EV (CFR)")
    ax0.axhline(THEORETICAL_EV, color="#dc2626", linewidth=1.5,
                linestyle="--",
                label=f"Theoretical EV = -1/18 = {THEORETICAL_EV:.4f}")
    ax0.set_ylabel("Expected Value (chips)", fontsize=11)
    ax0.set_title("Player 1 Expected Value vs. Iterations", fontsize=11)
    ax0.legend(fontsize=9)
    ax0.grid(True, which="both", alpha=0.3)
    ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # --- Subplot 2: Bluffing frequency ---
    ax1 = axes[1]
    ax1.semilogx(x, bluff_log, color="#16a34a", linewidth=2, marker="s",
                 markersize=5, label="P(bet | J, initial)")
    ax1.set_ylabel("Bluffing Frequency", fontsize=11)
    ax1.set_xlabel("Iterations (log scale)", fontsize=11)
    ax1.set_title("Player 1 Bluffing Frequency with Jack vs. Iterations",
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plot_path = "convergence_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Convergence plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
