"""Paired-seed Megagem bench for the analytic bots.

Run from the repository root:

    python3 experiments/bench_bots.py
    python3 experiments/bench_bots.py --seeds 36
    python3 experiments/bench_bots.py --seeds 16 --players 4

Seats rotate on the same deal. Charts cycle A–E with the seed. A win is any
share of the high score, so tied games count for every tied player. Confidence
intervals are across seeds: rotations of one deal are averaged first.
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench Megagem bots on paired seeds")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Retired. The submission bot no longer reads MEGAGEM_VL_*",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=(4, 5),
        help="Also run one mixed table at this player count",
    )
    args = parser.parse_args()
    seeds = range(args.seeds)
    if args.sweep:
        raise SystemExit(
            "ValueLiquidityBot no longer reads MEGAGEM_VL_*. "
            "Shipped lambda is 1.0 and delta is 1.0. "
            "Edit those class constants to retune."
        )
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


if __name__ == "__main__":
    main()
