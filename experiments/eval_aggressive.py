"""Paired evaluation of AggressiveBot on seeds 2000–2199.

Same design as experiments/eval_adaptive.py: in-process runners, every chart,
seat rotation, and a normal 95% CI across seeds after averaging charts and
seats inside each seed. A win is any share of the high score. These seeds were
not used to choose the bot's constants.

    python3 experiments/eval_aggressive.py --probe
    python3 experiments/eval_aggressive.py --first 2000 --last 2199 --workers 14
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from bot_definitions.aggressive_bot import (
    AggressiveBot,
    AggressiveNoEndgame,
    AggressiveNoLiquidity,
    AggressiveNoMissions,
    AggressiveNoReveal,
)
from bot_definitions.baselines import FractionValueBot, MyopicEvBot
from bot_definitions.eval_opponents import MissionAllInBot, OverbidBot, ZeroBidBot
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bots import Bot
from custom_bots import AdaptiveBot, SmallBidBot
from decks import VALUE_CHARTS
from eval_adaptive import (
    InlineRunner,
    _ci95,
    _diff_line,
    _label_entries,
    _mean,
    _rotate,
    format_report,
)
from harness import play_game

Entry = tuple[str, type[Bot]]


def _empty_scores(
    labels: list[str], seeds: list[int]
) -> dict[str, dict[int, list[float]]]:
    return {label: {seed: [] for seed in seeds} for label in labels}


def _tally(runners: list[InlineRunner]) -> tuple[int, int, int]:
    falls = clamps = reveals = 0
    for runner in runners:
        bot = getattr(runner, "_bot", None)
        falls += int(getattr(bot, "fallback_bids", 0))
        reveals += int(getattr(bot, "fallback_reveals", 0))
        clamps += int(getattr(bot, "clamped_bids", 0))
    return falls, reveals, clamps


def _finish(
    labels: list[str],
    seeds: list[int],
    scores: dict[str, dict[int, list[float]]],
    wins_by_seed: dict[str, dict[int, list[float]]],
    chart_scores: dict[str, dict[str, dict[int, list[float]]]],
    seat_sum: dict[str, dict[int, list[float]]],
    games: int,
    started: float,
    problems: tuple[int, int, int],
) -> dict[str, object]:
    wins = {
        label: int(sum(sum(wins_by_seed[label][seed]) for seed in seeds))
        for label in labels
    }
    return {
        "labels": labels,
        "seeds": seeds,
        "games": games,
        "wins": wins,
        "seed_means": {
            label: [_mean(scores[label][seed]) for seed in seeds] for label in labels
        },
        "seed_win_rates": {
            label: [_mean(wins_by_seed[label][seed]) for seed in seeds]
            for label in labels
        },
        "chart_means": {
            chart_name: {
                label: [_mean(per_seed[seed]) for seed in seeds]
                for label, per_seed in columns.items()
            }
            for chart_name, columns in chart_scores.items()
        },
        "seat_sum": seat_sum,
        "seconds": time.perf_counter() - started,
        "fallbacks": problems[0],
        "reveal_fallbacks": problems[1],
        "clamped": problems[2],
    }


def play_rotation(
    entries: list[Entry],
    seeds: list[int],
    *,
    match_seed: int,
) -> dict[str, object]:
    """Rotate seats. Each runner keeps its memory for the whole match."""
    entries = _label_entries(entries)
    labels = [label for label, _cls in entries]
    runners = [
        InlineRunner(cls, seed=match_seed + 1009 * index)
        for index, (_label, cls) in enumerate(entries)
    ]
    scores = _empty_scores(labels, seeds)
    wins_by_seed = _empty_scores(labels, seeds)
    chart_scores: dict[str, dict[str, dict[int, list[float]]]] = {
        chart.name: {label: {seed: [] for seed in seeds} for label in labels}
        for chart in VALUE_CHARTS
    }
    seat_sum: dict[str, dict[int, list[float]]] = {
        label: {seat: [0.0, 0.0, 0.0] for seat in range(len(entries))}
        for label in labels
    }
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
                        wins_by_seed[label][seed].append(won)
                        chart_scores[chart.name][label][seed].append(score)
                        bucket = seat_sum[label][seat]
                        bucket[0] += score
                        bucket[1] += won
                        bucket[2] += 1.0
    finally:
        problems = _tally(runners)
        for runner in runners:
            runner.close()
    return _finish(
        labels,
        seeds,
        scores,
        wins_by_seed,
        chart_scores,
        seat_sum,
        games,
        started,
        problems,
    )


def play_hero_seats(
    hero: Entry,
    opponent: Entry,
    players: int,
    seeds: list[int],
    *,
    match_seed: int,
) -> dict[str, object]:
    """Hero in every seat against copies. Opponent RNG matches on ``match_seed``."""
    hero_label, hero_cls = hero
    opp_label, opp_cls = opponent
    hero_runner = InlineRunner(hero_cls, seed=match_seed)
    opponents = [
        InlineRunner(opp_cls, seed=match_seed + 10007 * (index + 1))
        for index in range(players - 1)
    ]
    labels = [hero_label, opp_label]
    scores = _empty_scores(labels, seeds)
    wins_by_seed = _empty_scores(labels, seeds)
    chart_scores: dict[str, dict[str, dict[int, list[float]]]] = {
        chart.name: {label: {seed: [] for seed in seeds} for label in labels}
        for chart in VALUE_CHARTS
    }
    seat_sum: dict[str, dict[int, list[float]]] = {
        hero_label: {seat: [0.0, 0.0, 0.0] for seat in range(players)}
    }
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
                            wins_by_seed[hero_label][seed].append(won)
                            chart_scores[chart.name][hero_label][seed].append(score)
                            bucket = seat_sum[hero_label][seat]
                            bucket[0] += score
                            bucket[1] += won
                            bucket[2] += 1.0
                        else:
                            opp_scores.append(score)
                            if won:
                                opponent_won = True
                    wins_by_seed[opp_label][seed].append(1.0 if opponent_won else 0.0)
                    opp_mean = _mean(opp_scores)
                    scores[opp_label][seed].append(opp_mean)
                    chart_scores[chart.name][opp_label][seed].append(opp_mean)
    finally:
        problems = _tally([hero_runner, *opponents])
        hero_runner.close()
        for runner in opponents:
            runner.close()
    return _finish(
        labels,
        seeds,
        scores,
        wins_by_seed,
        chart_scores,
        seat_sum,
        games,
        started,
        problems,
    )


def _standard(players: int, hero: Entry) -> list[Entry]:
    field: list[Entry] = [
        hero,
        ("Myopic", MyopicEvBot),
        ("Fraction", FractionValueBot),
        ("SmallBid", SmallBidBot),
        ("Bot", Bot),
    ]
    if players == 3:
        return field[:3]
    return field[:players]


def build_jobs(seeds: list[int]) -> list[tuple[str, dict[str, object]]]:
    heroes: list[Entry] = [
        ("Agg", AggressiveBot),
        ("NoMiss", AggressiveNoMissions),
        ("NoLiq", AggressiveNoLiquidity),
        ("NoRev", AggressiveNoReveal),
        ("NoEnd", AggressiveNoEndgame),
        ("VL", ValueLiquidityBot),
        ("Adaptive", AdaptiveBot),
    ]
    unusual: list[Entry] = [
        ("Zero", ZeroBidBot),
        ("Over", OverbidBot),
        ("Mission", MissionAllInBot),
    ]
    jobs: list[tuple[str, dict[str, object]]] = []
    for players in (3, 4, 5):
        if players == 3:
            table: list[Entry] = [
                ("Agg", AggressiveBot),
                ("VL", ValueLiquidityBot),
                ("Adaptive", AdaptiveBot),
            ]
        elif players == 4:
            table = [
                ("Agg", AggressiveBot),
                ("VL", ValueLiquidityBot),
                ("Adaptive", AdaptiveBot),
                ("Myopic", MyopicEvBot),
            ]
        else:
            table = [
                ("Agg", AggressiveBot),
                ("VL", ValueLiquidityBot),
                ("Adaptive", AdaptiveBot),
                ("Myopic", MyopicEvBot),
                ("Fraction", FractionValueBot),
            ]
        jobs.append(
            (
                f"1. Together {players}p",
                {
                    "mode": "rotate",
                    "entries": table,
                    "seeds": seeds,
                    "match_seed": 5000 + players,
                    "focus": "Agg",
                },
            )
        )
        for hero in heroes:
            jobs.append(
                (
                    f"2. Standard {players}p  {hero[0]}",
                    {
                        "mode": "rotate",
                        "entries": _standard(players, hero),
                        "seeds": seeds,
                        "match_seed": 6000 + players,
                        "focus": hero[0],
                    },
                )
            )
        if players == 3:
            for hero in (
                ("Agg", AggressiveBot),
                ("VL", ValueLiquidityBot),
                ("Adaptive", AdaptiveBot),
            ):
                jobs.append(
                    (
                        f"2b. SmallBid+Bot 3p  {hero[0]}",
                        {
                            "mode": "rotate",
                            "entries": [
                                (hero[0], hero[1]),
                                ("SmallBid", SmallBidBot),
                                ("Bot", Bot),
                            ],
                            "seeds": seeds,
                            "match_seed": 6103,
                            "focus": hero[0],
                        },
                    )
                )
        for opp in unusual:
            for hero in (
                ("Agg", AggressiveBot),
                ("VL", ValueLiquidityBot),
                ("Adaptive", AdaptiveBot),
            ):
                jobs.append(
                    (
                        f"3. vs {opp[0]} {players}p  {hero[0]}",
                        {
                            "mode": "hero",
                            "hero": hero,
                            "opponent": opp,
                            "players": players,
                            "seeds": seeds,
                            "match_seed": 7000 + players * 19 + len(opp[0]),
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


def _paired(
    title: str,
    left: dict[str, object],
    right: dict[str, object],
    left_label: str,
    right_label: str,
) -> str:
    """Seed-paired score and win-rate gaps. Positive favors ``left_label``."""
    lines = [
        f"\n{title}",
        _diff_line(
            _as_list(left, "seed_means", left_label),
            _as_list(right, "seed_means", right_label),
            right_label,
        ),
    ]
    win_diff = [
        ours - theirs
        for ours, theirs in zip(
            _as_list(left, "seed_win_rates", left_label),
            _as_list(right, "seed_win_rates", right_label),
        )
    ]
    mean, low, high = _ci95(win_diff)
    side = "CI above 0" if low > 0 else "CI below 0" if high < 0 else "CI includes 0"
    lines.append(
        f"  win rate {mean:+7.1%}  [{low:+7.1%}, {high:+7.1%}]  {side}"
    )
    left_charts = left["chart_means"]
    right_charts = right["chart_means"]
    assert isinstance(left_charts, dict) and isinstance(right_charts, dict)
    for chart_name in left_charts:
        left_cols = left_charts[chart_name]
        right_cols = right_charts[chart_name]
        assert isinstance(left_cols, dict) and isinstance(right_cols, dict)
        mean, low, high = _ci95(
            [
                a - b
                for a, b in zip(left_cols[left_label], right_cols[right_label])
            ]
        )
        lines.append(f"  {chart_name}: {mean:+.2f} [{low:+.2f}, {high:+.2f}]")
    return "\n".join(lines)


def _as_list(report: dict[str, object], key: str, label: str) -> list[float]:
    block = report[key]
    assert isinstance(block, dict)
    values = block[label]
    assert isinstance(values, list)
    return values


def _probe() -> None:
    seeds = [2000]
    started = time.perf_counter()
    report = play_rotation(
        [
            ("Agg", AggressiveBot),
            ("VL", ValueLiquidityBot),
            ("Adaptive", AdaptiveBot),
            ("Myopic", MyopicEvBot),
            ("Fraction", FractionValueBot),
        ],
        seeds,
        match_seed=1,
    )
    games = report["games"]
    assert isinstance(games, int)
    elapsed = time.perf_counter() - started
    print(
        f"probe: {games} five-player games in {elapsed:.2f}s "
        f"({elapsed / games:.3f}s/game) "
        f"fallbacks={report['fallbacks']} clamped={report['clamped']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AggressiveBot")
    parser.add_argument("--first", type=int, default=2000)
    parser.add_argument("--last", type=int, default=2199)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/aggressive_results.txt"),
    )
    args = parser.parse_args()
    if args.probe:
        _probe()
        return
    seeds = list(range(args.first, args.last + 1))
    jobs = build_jobs(seeds)
    workers = args.workers or (get_context("fork").cpu_count() or 2)
    print(f"{len(jobs)} matches, {len(seeds)} seeds, {workers} workers", flush=True)
    started = time.perf_counter()
    reports: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("fork")
    ) as pool:
        future_map = {pool.submit(run_job, spec): title for title, spec in jobs}
        done = 0
        for future in as_completed(future_map):
            title = future_map[future]
            reports[title] = future.result()
            done += 1
            print(
                f"[{done}/{len(jobs)}] {title} ({reports[title]['seconds']:.0f}s)",
                flush=True,
            )
    chunks = [
        "AggressiveBot evaluation",
        f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}). Charts A–E. Seats rotated.",
        "A seed contributes one number: its mean over charts and seat rotations.",
        "The 95% CI uses the seed means (sample variance, n-1, 1.96).",
        "Paired gaps subtract the matching seed. Positive favors the first bot.",
        "A win is any share of the high score. Win-rate gaps use the same seeds.",
        "In-process runners, so a callback timeout is not enforced here.",
        f"Wall clock {time.perf_counter() - started:.0f}s.",
    ]
    falls = sum(int(reports[title]["fallbacks"]) for title, _spec in jobs)
    reveals = sum(int(reports[title]["reveal_fallbacks"]) for title, _spec in jobs)
    clamps = sum(int(reports[title]["clamped"]) for title, _spec in jobs)
    chunks.append(
        f"Fallback bids {falls}, fallback reveals {reveals}, clamped bids {clamps}."
    )
    for title, _spec in jobs:
        report = reports[title]
        focus = report.get("focus")
        chunks.append(
            format_report(title, report, focus if isinstance(focus, str) else None)
        )
    for players in (3, 4, 5):
        full = reports[f"2. Standard {players}p  Agg"]
        for other, label in (
            ("VL", "VL"),
            ("Adaptive", "Adaptive"),
            ("NoMiss", "NoMiss"),
            ("NoLiq", "NoLiq"),
            ("NoRev", "NoRev"),
            ("NoEnd", "NoEnd"),
        ):
            chunks.append(
                _paired(
                    f"Paired Agg − {other} on standard {players}p",
                    full,
                    reports[f"2. Standard {players}p  {other}"],
                    "Agg",
                    label,
                )
            )
        if players == 3:
            chunks.append(
                _paired(
                    "Paired Agg − VL on 3p SmallBid+Bot",
                    reports["2b. SmallBid+Bot 3p  Agg"],
                    reports["2b. SmallBid+Bot 3p  VL"],
                    "Agg",
                    "VL",
                )
            )
            chunks.append(
                _paired(
                    "Paired Agg − Adaptive on 3p SmallBid+Bot",
                    reports["2b. SmallBid+Bot 3p  Agg"],
                    reports["2b. SmallBid+Bot 3p  Adaptive"],
                    "Agg",
                    "Adaptive",
                )
            )
        for opp in ("Zero", "Over", "Mission"):
            chunks.append(
                _paired(
                    f"Paired Agg − VL vs {opp} {players}p",
                    reports[f"3. vs {opp} {players}p  Agg"],
                    reports[f"3. vs {opp} {players}p  VL"],
                    "Agg",
                    "VL",
                )
            )
            chunks.append(
                _paired(
                    f"Paired Agg − Adaptive vs {opp} {players}p",
                    reports[f"3. vs {opp} {players}p  Agg"],
                    reports[f"3. vs {opp} {players}p  Adaptive"],
                    "Agg",
                    "Adaptive",
                )
            )
    text = "\n".join(chunks) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
