"""Paired evaluation of AdaptiveBot on fresh seeds.

In-process runners call the same harness as experiments/compare.py. Each bot
keeps its checkpoint memory for the whole match, which is what lets the
cross-game portfolio run. Seats rotate on the same deal. Charts A–E are played
in full for every seed. A win is any share of the high score.

    python3 experiments/eval_adaptive.py --probe
    python3 experiments/eval_adaptive.py --first 1000 --last 1199

The default seed range is 1000–1199, which is not the range the base bot was
tuned on.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from bot_definitions.adaptive_bot import (
    AdaptiveBot,
    AdaptiveLayer1Bot,
    AdaptiveLayer12Bot,
    AdaptiveLayer13Bot,
)
from bot_definitions.baselines import FractionValueBot, MyopicEvBot
from bot_definitions.eval_opponents import (
    MissionAllInBot,
    OverbidBot,
    RandomBidBot,
    ZeroBidBot,
)
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bot_runner import BotRunner
from bots import Bot
from custom_bots import SmallBidBot
from decks import VALUE_CHARTS
from harness import play_game

Entry = tuple[str, type[Bot]]


class InlineRunner(BotRunner):
    """BotRunner that calls the bot in this process and keeps its random state.

    ``play_game`` accepts BotRunner instances and will not wrap them. Swapping
    ``random.getstate`` around each callback gives every seat the private
    stream a worker would have had.
    """

    def __init__(self, bot_class: type[Bot], seed: int) -> None:
        super().__init__(bot_class, timeout=None, seed=seed)
        previous = random.getstate()
        random.seed(seed)
        self._rng_state = random.getstate()
        random.setstate(previous)
        self._bot = bot_class()

    def call(self, method: str, *args: object) -> int | None:
        previous = random.getstate()
        random.setstate(self._rng_state)
        try:
            result = getattr(self._bot, method)(*args)
            self._rng_state = random.getstate()
        finally:
            random.setstate(previous)
        if type(result) is int:
            return result
        return None

    def close(self) -> None:
        self._channel = None
        self._process = None


def _rotate(indexes: list[int], shift: int) -> list[int]:
    shift %= len(indexes)
    return indexes[shift:] + indexes[:shift]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> tuple[float, float, float]:
    count = len(values)
    mean = _mean(values)
    if count < 2:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    margin = 1.96 * (variance / count) ** 0.5
    return mean, mean - margin, mean + margin


def _label_entries(entries: list[Entry]) -> list[Entry]:
    """Disambiguate repeated classes without merging their seats."""
    total: dict[str, int] = defaultdict(int)
    for name, _cls in entries:
        total[name] += 1
    seen: dict[str, int] = defaultdict(int)
    labeled: list[Entry] = []
    for name, cls in entries:
        if total[name] == 1:
            labeled.append((name, cls))
            continue
        seen[name] += 1
        labeled.append((f"{name}#{seen[name]}", cls))
    return labeled


def play_rotation(
    entries: list[Entry],
    seeds: list[int],
    *,
    match_seed: int,
) -> dict[str, object]:
    """Rotate seats. Memory persists on each runner for the whole match."""
    entries = _label_entries(entries)
    labels = [label for label, _cls in entries]
    runners = [
        InlineRunner(cls, seed=match_seed + 1009 * index)
        for index, (_label, cls) in enumerate(entries)
    ]
    # scores[label][seed] -> list over charts and rotations
    scores: dict[str, dict[int, list[float]]] = {
        label: {seed: [] for seed in seeds} for label in labels
    }
    chart_scores: dict[str, dict[str, dict[int, list[float]]]] = {
        chart.name: {label: {seed: [] for seed in seeds} for label in labels}
        for chart in VALUE_CHARTS
    }
    seat_sum: dict[str, dict[int, list[float]]] = {
        label: {seat: [0.0, 0.0, 0.0] for seat in range(len(entries))}
        for label in labels
    }
    wins = dict.fromkeys(labels, 0)
    games = 0
    started = time.perf_counter()
    try:
        order_base = list(range(len(entries)))
        for seed in seeds:
            for chart in VALUE_CHARTS:
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
                        won = 1.0 if seat in result.winner_ids else 0.0
                        scores[label][seed].append(score)
                        chart_scores[chart.name][label][seed].append(score)
                        bucket = seat_sum[label][seat]
                        bucket[0] += score
                        bucket[1] += won
                        bucket[2] += 1.0
                        if won:
                            wins[label] += 1
    finally:
        for runner in runners:
            runner.close()
    seed_means = {
        label: [_mean(scores[label][seed]) for seed in seeds] for label in labels
    }
    chart_means = {
        chart_name: {
            label: [_mean(per_seed[seed]) for seed in seeds]
            for label, per_seed in columns.items()
        }
        for chart_name, columns in chart_scores.items()
    }
    return {
        "labels": labels,
        "seeds": seeds,
        "games": games,
        "wins": wins,
        "seed_means": seed_means,
        "chart_means": chart_means,
        "seat_sum": seat_sum,
        "seconds": time.perf_counter() - started,
    }


def play_hero_seats(
    hero: Entry,
    opponent: Entry,
    players: int,
    seeds: list[int],
    *,
    match_seed: int,
) -> dict[str, object]:
    """Put ``hero`` in every seat against identical copies of ``opponent``.

    A second call with another hero and the same ``match_seed`` faces the same
    deals and the same opponent random streams.
    """
    hero_label, hero_cls = hero
    opp_label, opp_cls = opponent
    hero_runner = InlineRunner(hero_cls, seed=match_seed)
    opponents = [
        InlineRunner(opp_cls, seed=match_seed + 10007 * (index + 1))
        for index in range(players - 1)
    ]
    labels = [hero_label, opp_label]
    scores: dict[str, dict[int, list[float]]] = {
        label: {seed: [] for seed in seeds} for label in labels
    }
    chart_scores: dict[str, dict[str, dict[int, list[float]]]] = {
        chart.name: {label: {seed: [] for seed in seeds} for label in labels}
        for chart in VALUE_CHARTS
    }
    seat_sum: dict[str, dict[int, list[float]]] = {
        hero_label: {seat: [0.0, 0.0, 0.0] for seat in range(players)}
    }
    wins = {hero_label: 0, opp_label: 0}
    games = 0
    started = time.perf_counter()
    try:
        for seed in seeds:
            for chart in VALUE_CHARTS:
                for seat in range(players):
                    bots: list[InlineRunner] = []
                    opp_index = 0
                    for index in range(players):
                        if index == seat:
                            bots.append(hero_runner)
                        else:
                            bots.append(opponents[opp_index])
                            opp_index += 1
                    result = play_game(
                        players,
                        bots=bots,
                        rng=Random(seed),
                        value_chart=chart,
                    )
                    games += 1
                    opp_scores: list[float] = []
                    opponent_won = False
                    for index in range(players):
                        score = float(result.scores[index])
                        won = 1.0 if index in result.winner_ids else 0.0
                        if index == seat:
                            scores[hero_label][seed].append(score)
                            chart_scores[chart.name][hero_label][seed].append(score)
                            bucket = seat_sum[hero_label][seat]
                            bucket[0] += score
                            bucket[1] += won
                            bucket[2] += 1.0
                            if won:
                                wins[hero_label] += 1
                        else:
                            opp_scores.append(score)
                            if won:
                                opponent_won = True
                    if opponent_won:
                        wins[opp_label] += 1
                    opp_mean = _mean(opp_scores)
                    scores[opp_label][seed].append(opp_mean)
                    chart_scores[chart.name][opp_label][seed].append(opp_mean)
    finally:
        hero_runner.close()
        for runner in opponents:
            runner.close()
    seed_means = {
        label: [_mean(scores[label][seed]) for seed in seeds] for label in labels
    }
    chart_means = {
        chart_name: {
            label: [_mean(per_seed[seed]) for seed in seeds]
            for label, per_seed in columns.items()
        }
        for chart_name, columns in chart_scores.items()
    }
    return {
        "labels": labels,
        "seeds": seeds,
        "games": games,
        "wins": wins,
        "seed_means": seed_means,
        "chart_means": chart_means,
        "seat_sum": seat_sum,
        "seconds": time.perf_counter() - started,
    }


def _fmt_ci(values: list[float]) -> str:
    mean, low, high = _ci95(values)
    return f"{mean:7.2f} [{low:7.2f}, {high:7.2f}]"


def _diff_line(focus: list[float], other: list[float], label: str) -> str:
    diffs = [ours - theirs for ours, theirs in zip(focus, other)]
    mean, low, high = _ci95(diffs)
    side = "CI above 0" if low > 0 else "CI below 0" if high < 0 else "CI includes 0"
    return (
        f"  vs {label:<16} {mean:+7.2f}  [{low:+7.2f}, {high:+7.2f}]  {side}"
    )


def format_report(title: str, report: dict[str, object], focus: str | None) -> str:
    labels = report["labels"]
    wins = report["wins"]
    games = report["games"]
    seed_means = report["seed_means"]
    chart_means = report["chart_means"]
    seat_sum = report["seat_sum"]
    assert isinstance(labels, list)
    assert isinstance(wins, dict)
    assert isinstance(games, int)
    assert isinstance(seed_means, dict)
    assert isinstance(chart_means, dict)
    assert isinstance(seat_sum, dict)
    lines = [
        f"\n{title}",
        f"{report.get('seconds', 0):.1f}s, {len(report['seeds'])} seeds, {games} games",
        f"{'Bot':<16} {'mean score':>28} {'win rate':>10}",
    ]
    for label in labels:
        means = seed_means[label]
        assert isinstance(means, list)
        rate = wins[label] / games if games else 0.0
        lines.append(f"{label:<16} {_fmt_ci(means)} {rate:10.1%}")
    if focus is None and labels:
        focus = str(labels[0])
    if focus is not None and focus in seed_means:
        lines.append(f"Paired mean score minus the others (positive favors {focus}):")
        ours = seed_means[focus]
        assert isinstance(ours, list)
        for label in labels:
            if label == focus:
                continue
            other = seed_means[label]
            assert isinstance(other, list)
            lines.append(_diff_line(ours, other, label))
    lines.append("Mean score by chart (seed-level 95% CI of the focus shown below):")
    header = f"{'chart':<8}" + "".join(f"{label[:12]:>14}" for label in labels)
    lines.append(header)
    for chart_name, columns in chart_means.items():
        assert isinstance(columns, dict)
        cells = "".join(f"{_mean(columns[label]):14.2f}" for label in labels)
        lines.append(f"{chart_name:<8}{cells}")
    if focus is not None and focus in (next(iter(chart_means.values())) if chart_means else {}):
        lines.append(f"Paired chart gaps for {focus}:")
        for chart_name, columns in chart_means.items():
            assert isinstance(columns, dict)
            ours = columns[focus]
            assert isinstance(ours, list)
            bits = [f"{chart_name}:"]
            for label in labels:
                if label == focus:
                    continue
                other = columns[label]
                assert isinstance(other, list)
                mean, low, high = _ci95(
                    [left - right for left, right in zip(ours, other)]
                )
                bits.append(f"{label} {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
            lines.append("  " + " | ".join(bits))
    lines.append("Seat-by-seat mean score and win rate:")
    for label in labels:
        seats = seat_sum.get(label)
        if not isinstance(seats, dict) or not seats:
            continue
        bits = []
        for seat in sorted(seats):
            total, won, count = seats[seat]
            if count <= 0:
                continue
            bits.append(f"s{seat} {total / count:.1f} ({won / count:.0%})")
        if bits:
            lines.append(f"  {label:<16} {', '.join(bits)}")
    return "\n".join(lines)


def _analytic(players: int) -> list[Entry]:
    entries: list[Entry] = [
        ("Adaptive", AdaptiveBot),
        ("VL", ValueLiquidityBot),
        ("Myopic", MyopicEvBot),
    ]
    extras: list[Entry] = [("VL", ValueLiquidityBot), ("Myopic", MyopicEvBot)]
    while len(entries) < players:
        entries.append(extras[(len(entries) - 1) % 2])
    return entries


def _standard(players: int, hero: Entry) -> list[Entry]:
    field: list[Entry] = [
        hero,
        ("Fraction", FractionValueBot),
        ("SmallBid", SmallBidBot),
        ("Bot", Bot),
    ]
    if players == 3:
        return field[:3]
    if players == 5:
        field.append(("Fraction", FractionValueBot))
    return field[:players]


def _unusual_mix(players: int, hero: Entry) -> list[Entry]:
    field: list[Entry] = [
        hero,
        ("Over", OverbidBot),
        ("Zero", ZeroBidBot),
        ("Random", RandomBidBot),
        ("Mission", MissionAllInBot),
    ]
    return field[:players]


def build_jobs(seeds: list[int]) -> list[tuple[str, dict[str, object]]]:
    """Each job is (title, spec). Specs are executed by ``run_job``."""
    heroes: list[Entry] = [
        ("L1", AdaptiveLayer1Bot),
        ("L12", AdaptiveLayer12Bot),
        ("L13", AdaptiveLayer13Bot),
        ("All", AdaptiveBot),
        ("Base", ValueLiquidityBot),
    ]
    unusual: list[Entry] = [
        ("Zero", ZeroBidBot),
        ("Over", OverbidBot),
        ("Random", RandomBidBot),
        ("Mission", MissionAllInBot),
    ]
    jobs: list[tuple[str, dict[str, object]]] = []
    for players in (3, 4, 5):
        jobs.append(
            (
                f"1. Head-to-head {players}p  Adaptive vs VL vs Myopic",
                {
                    "mode": "rotate",
                    "entries": _analytic(players),
                    "seeds": seeds,
                    "match_seed": 1000 + players,
                    "focus": "Adaptive",
                },
            )
        )
        for hero in heroes:
            jobs.append(
                (
                    f"2/4. Standard {players}p  {hero[0]}",
                    {
                        "mode": "rotate",
                        "entries": _standard(players, hero),
                        "seeds": seeds,
                        "match_seed": 2000 + players,
                        "focus": hero[0],
                    },
                )
            )
            jobs.append(
                (
                    f"4. Unusual mix {players}p  {hero[0]}",
                    {
                        "mode": "rotate",
                        "entries": _unusual_mix(players, hero),
                        "seeds": seeds,
                        "match_seed": 3000 + players,
                        "focus": hero[0],
                    },
                )
            )
        for opp_name, opp_cls in unusual:
            for hero_name, hero_cls in (
                ("Adaptive", AdaptiveBot),
                ("VL", ValueLiquidityBot),
            ):
                jobs.append(
                    (
                        f"3. vs {opp_name} {players}p  {hero_name}",
                        {
                            "mode": "hero",
                            "hero": (hero_name, hero_cls),
                            "opponent": (opp_name, opp_cls),
                            "players": players,
                            "seeds": seeds,
                            "match_seed": 4000 + players * 17 + len(opp_name),
                            "focus": hero_name,
                        },
                    )
                )
    # 3p standard above is Hero/Fraction/SmallBid (Bot does not fit).
    # Add Hero/SmallBid/Bot for the submitted bot and the base bot only.
    for hero in (("All", AdaptiveBot), ("Base", ValueLiquidityBot)):
        jobs.append(
            (
                f"2. Standard 3p SmallBid+Bot  {hero[0]}",
                {
                    "mode": "rotate",
                    "entries": [
                        (hero[0], hero[1]),
                        ("SmallBid", SmallBidBot),
                        ("Bot", Bot),
                    ],
                    "seeds": seeds,
                    "match_seed": 2503,
                    "focus": hero[0],
                },
            )
        )
    return jobs


def run_job(spec: dict[str, object]) -> dict[str, object]:
    seeds = spec["seeds"]
    assert isinstance(seeds, list)
    match_seed = spec["match_seed"]
    assert isinstance(match_seed, int)
    if spec["mode"] == "rotate":
        entries = spec["entries"]
        assert isinstance(entries, list)
        report = play_rotation(entries, seeds, match_seed=match_seed)
    else:
        hero = spec["hero"]
        opponent = spec["opponent"]
        players = spec["players"]
        assert isinstance(hero, tuple)
        assert isinstance(opponent, tuple)
        assert isinstance(players, int)
        report = play_hero_seats(
            hero, opponent, players, seeds, match_seed=match_seed
        )
    report["focus"] = spec["focus"]
    return report


def _probe() -> None:
    seeds = [1000]
    started = time.perf_counter()
    report = play_rotation(
        _analytic(5),
        seeds,
        match_seed=1,
    )
    elapsed = time.perf_counter() - started
    games = report["games"]
    assert isinstance(games, int)
    print(
        f"probe: {games} five-player games in {elapsed:.2f}s "
        f"({elapsed / games:.3f}s/game)"
    )


def _paired_block(
    title: str,
    left: dict[str, object],
    right: dict[str, object],
    left_label: str,
    right_label: str,
) -> str:
    """Difference of two replacement matches on the same seeds."""
    left_means = left["seed_means"]
    right_means = right["seed_means"]
    assert isinstance(left_means, dict) and isinstance(right_means, dict)
    ours = left_means[left_label]
    theirs = right_means[right_label]
    assert isinstance(ours, list) and isinstance(theirs, list)
    lines = [f"\n{title}", _diff_line(ours, theirs, right_label)]
    left_charts = left["chart_means"]
    right_charts = right["chart_means"]
    assert isinstance(left_charts, dict) and isinstance(right_charts, dict)
    for chart_name in left_charts:
        left_cols = left_charts[chart_name]
        right_cols = right_charts[chart_name]
        assert isinstance(left_cols, dict) and isinstance(right_cols, dict)
        left_values = left_cols[left_label]
        right_values = right_cols[right_label]
        assert isinstance(left_values, list) and isinstance(right_values, list)
        mean, low, high = _ci95(
            [a - b for a, b in zip(left_values, right_values)]
        )
        lines.append(
            f"  {chart_name}: {mean:+.2f} [{low:+.2f}, {high:+.2f}]"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveBot")
    parser.add_argument("--first", type=int, default=1000)
    parser.add_argument("--last", type=int, default=1199)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--submitted",
        action="store_true",
        help="Only head-to-head and per-type unusual tables, using AdaptiveBot as shipped",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/adaptive_results.txt"),
    )
    args = parser.parse_args()
    if args.probe:
        _probe()
        return
    seeds = list(range(args.first, args.last + 1))
    jobs = build_jobs(seeds)
    if args.submitted:
        jobs = [
            job
            for job in jobs
            if job[0].startswith("1. Head-to-head") or job[0].startswith("3. vs ")
        ]
    workers = args.workers or (get_context("fork").cpu_count() or 2)
    print(f"{len(jobs)} matches, {len(seeds)} seeds, {workers} workers", flush=True)
    started = time.perf_counter()
    reports: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
    ) as pool:
        future_map = {
            pool.submit(run_job, spec): title for title, spec in jobs
        }
        done = 0
        for future in as_completed(future_map):
            title = future_map[future]
            reports[title] = future.result()
            done += 1
            seconds = reports[title]["seconds"]
            print(f"[{done}/{len(jobs)}] {title} ({seconds:.0f}s)", flush=True)
    # future_map iteration order is not submission order. Re-sort by jobs.
    del future_map
    chunks = [
        "AdaptiveBot evaluation",
        f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), charts A–E, seats rotated.",
        "Means and 95% CIs are across seeds, after averaging charts and seats.",
        "A win is any share of the high score.",
        f"Wall clock {time.perf_counter() - started:.0f}s.",
    ]
    for title, _spec in jobs:
        report = reports[title]
        focus = report.get("focus")
        chunks.append(
            format_report(title, report, focus if isinstance(focus, str) else None)
        )
    # Paired Adaptive minus VL on each copied-opponent table.
    for players in (3, 4, 5):
        for opp_name in ("Zero", "Over", "Random", "Mission"):
            left_key = f"3. vs {opp_name} {players}p  Adaptive"
            right_key = f"3. vs {opp_name} {players}p  VL"
            if left_key not in reports or right_key not in reports:
                continue
            chunks.append(
                _paired_block(
                    f"3. Paired Adaptive − VL vs {opp_name} {players}p",
                    reports[left_key],
                    reports[right_key],
                    "Adaptive",
                    "VL",
                )
            )
        for field in ("2/4. Standard", "4. Unusual mix"):
            base_key = f"{field} {players}p  Base"
            if base_key not in reports:
                continue
            right = reports[base_key]
            for variant in ("All", "L1", "L12", "L13"):
                left_key = f"{field} {players}p  {variant}"
                if left_key not in reports:
                    continue
                chunks.append(
                    _paired_block(
                        f"Paired {variant} − Base on {field} {players}p",
                        reports[left_key],
                        right,
                        variant,
                        "Base",
                    )
                )
    small_all = "2. Standard 3p SmallBid+Bot  All"
    small_base = "2. Standard 3p SmallBid+Bot  Base"
    if small_all in reports and small_base in reports:
        chunks.append(
            _paired_block(
                "Paired All − Base on 3p SmallBid+Bot",
                reports[small_all],
                reports[small_base],
                "All",
                "Base",
            )
        )
    text = "\n".join(chunks) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
