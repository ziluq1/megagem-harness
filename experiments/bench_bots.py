"""Paired-seed Megagem bench for the analytic bots.

Run from the repository root:

    python3 experiments/bench_bots.py
    python3 experiments/bench_bots.py --seeds 20 --sweep

Seats rotate on the same deal. Charts cycle A–E with the seed. A win is any
share of the high score, so tied games count for every tied player. Confidence
intervals are across seeds: rotations of one deal are averaged first.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contextlib import ExitStack
from random import Random

from bot_definitions.baselines import FractionValueBot, MyopicEvBot
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bot_runner import BotRunner
from bots import Bot
from custom_bots import SmallBidBot
from decks import VALUE_CHARTS
from harness import play_game

# Labels stay with the runner so a rotated seat is still the same entrant.
DEFAULT_TABLES: tuple[tuple[str, tuple[tuple[str, type[Bot]], ...]], ...] = (
    (
        "3-player mixed",
        (
            ("ValueLiquidity", ValueLiquidityBot),
            ("MyopicEV", MyopicEvBot),
            ("FractionValue", FractionValueBot),
        ),
    ),
    (
        "3-player vs small bids",
        (
            ("ValueLiquidity", ValueLiquidityBot),
            ("SmallBid", SmallBidBot),
            ("Default", Bot),
        ),
    ),
)


def _rotate(indexes: list[int], shift: int) -> list[int]:
    shift %= len(indexes)
    return indexes[shift:] + indexes[:shift]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Normal 95% interval for the mean. ``values`` should be one per seed."""
    count = len(values)
    mean = _mean(values)
    if count < 2:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    margin = 1.96 * (variance / count) ** 0.5
    return mean, mean - margin, mean + margin


def evaluate(
    entries: tuple[tuple[str, type[Bot]], ...],
    seeds: range,
    *,
    timeout: float,
) -> dict[str, object]:
    """Play every seed at every seat. Runners are reused across the match."""
    labels = [label for label, _class in entries]
    per_seed: dict[str, dict[int, list[float]]] = {
        label: {seed: [] for seed in seeds} for label in labels
    }
    per_chart: dict[str, dict[str, list[float]]] = {
        chart.name: {label: [] for label in labels} for chart in VALUE_CHARTS
    }
    wins = dict.fromkeys(labels, 0)
    games = 0
    with ExitStack() as stack:
        runners = [
            stack.enter_context(
                BotRunner(bot_class, timeout=timeout, seed=1000 + index)
            )
            for index, (_label, bot_class) in enumerate(entries)
        ]
        order_base = list(range(len(entries)))
        for seed in seeds:
            chart = VALUE_CHARTS[seed % len(VALUE_CHARTS)]
            for shift in range(len(entries)):
                order = _rotate(order_base, shift)
                result = play_game(
                    len(entries),
                    bots=[runners[index] for index in order],
                    rng=Random(seed),
                    value_chart=chart,
                )
                games += 1
                for seat, index in enumerate(order):
                    label = labels[index]
                    score = float(result.scores[seat])
                    per_seed[label][seed].append(score)
                    per_chart[chart.name][label].append(score)
                    if seat in result.winner_ids:
                        wins[label] += 1
    seed_means = {
        label: [_mean(per_seed[label][seed]) for seed in seeds] for label in labels
    }
    return {
        "labels": labels,
        "games": games,
        "wins": wins,
        "seed_means": seed_means,
        "per_chart": per_chart,
        "seeds": len(list(seeds)),
    }


def _print_table(title: str, report: dict[str, object]) -> None:
    labels = report["labels"]
    assert isinstance(labels, list)
    games = report["games"]
    wins = report["wins"]
    seed_means = report["seed_means"]
    assert isinstance(wins, dict)
    assert isinstance(seed_means, dict)
    print(f"\n{title}: {report['seeds']} seeds, {games} games")
    print(f"{'Bot':<20} {'mean':>8} {'95% CI':>22} {'win rate':>10}")
    focus = None
    focus_mean = None
    for label in labels:
        means = seed_means[label]
        assert isinstance(means, list)
        mean, low, high = _ci95(means)
        rate = wins[label] / games if isinstance(games, int) and games else 0.0
        print(f"{label:<20} {mean:8.2f} [{low:7.2f}, {high:7.2f}] {rate:10.1%}")
        if focus is None:
            focus = label
            focus_mean = means
    if focus is None or focus_mean is None:
        return
    print("Paired mean score minus the other bots (positive favors", focus + "):")
    for label in labels[1:]:
        other = seed_means[label]
        assert isinstance(other, list)
        diffs = [ours - theirs for ours, theirs in zip(focus_mean, other)]
        mean, low, high = _ci95(diffs)
        print(f"  vs {label:<16} {mean:+7.2f}  [{low:+7.2f}, {high:+7.2f}]")
    per_chart = report["per_chart"]
    assert isinstance(per_chart, dict)
    print("Mean score by chart:")
    header = f"{'chart':<8}" + "".join(f"{label[:12]:>14}" for label in labels)
    print(header)
    for chart_name, columns in per_chart.items():
        assert isinstance(columns, dict)
        cells = "".join(f"{_mean(columns[label]):14.2f}" for label in labels)
        print(f"{chart_name:<8}{cells}")


def _sweep(seeds: range, timeout: float) -> None:
    """Retune lambda and delta against the myopic and fraction baselines."""
    grid = [
        (lam, delta)
        for lam in (0.25, 0.5, 0.75, 1.0)
        for delta in (0.0, 1.0, 2.0, 3.0)
    ]
    print("\nLambda / delta sweep, ValueLiquidity vs MyopicEV and FractionValue")
    print(f"{'lam':>6} {'delta':>6} {'mean':>8} {'vs myopic':>12} {'vs fraction':>12}")
    best = None
    for lam, delta in grid:
        os.environ["MEGAGEM_VL_LAM"] = str(lam)
        os.environ["MEGAGEM_VL_DELTA"] = str(delta)
        report = evaluate(
            (
                ("ValueLiquidity", ValueLiquidityBot),
                ("MyopicEV", MyopicEvBot),
                ("FractionValue", FractionValueBot),
            ),
            seeds,
            timeout=timeout,
        )
        seed_means = report["seed_means"]
        assert isinstance(seed_means, dict)
        ours = seed_means["ValueLiquidity"]
        myopic = seed_means["MyopicEV"]
        fraction = seed_means["FractionValue"]
        assert isinstance(ours, list) and isinstance(myopic, list)
        assert isinstance(fraction, list)
        mean = _mean(ours)
        vs_myopic = _mean([a - b for a, b in zip(ours, myopic)])
        vs_fraction = _mean([a - b for a, b in zip(ours, fraction)])
        print(f"{lam:6.2f} {delta:6.1f} {mean:8.2f} {vs_myopic:+12.2f} {vs_fraction:+12.2f}")
        key = (vs_myopic + vs_fraction, mean)
        if best is None or key > best[0]:
            best = (key, lam, delta, mean, vs_myopic, vs_fraction)
    if best is not None:
        _key, lam, delta, mean, vs_myopic, vs_fraction = best
        print(
            f"Highest combined margin in this grid: lambda={lam}, delta={delta}, "
            f"mean={mean:.2f}, vs myopic {vs_myopic:+.2f}, vs fraction {vs_fraction:+.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench Megagem bots on paired seeds")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search lambda and delta (uses MEGAGEM_VL_* for this process)",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=(4, 5),
        help="Also run one mixed table at this player count",
    )
    args = parser.parse_args()
    seeds = range(args.seeds)
    try:
        if args.sweep:
            _sweep(seeds, args.timeout)
            return
        for title, entries in DEFAULT_TABLES:
            _print_table(title, evaluate(entries, seeds, timeout=args.timeout))
        if args.players:
            extras: tuple[tuple[str, type[Bot]], ...] = (
                ("ValueLiquidity", ValueLiquidityBot),
                ("MyopicEV", MyopicEvBot),
                ("FractionValue", FractionValueBot),
                ("SmallBid", SmallBidBot),
            )
            if args.players == 5:
                extras = extras + (("Default", Bot),)
            _print_table(
                f"{args.players}-player mixed",
                evaluate(extras, seeds, timeout=args.timeout),
            )
    finally:
        os.environ.pop("MEGAGEM_VL_LAM", None)
        os.environ.pop("MEGAGEM_VL_DELTA", None)
        os.environ.pop("MEGAGEM_VL_FRACTION", None)
        os.environ.pop("MEGAGEM_VL_DENIAL", None)


if __name__ == "__main__":
    main()
