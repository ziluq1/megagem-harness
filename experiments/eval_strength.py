"""Before/after check for the three bidding changes.

Dev seeds are 3000–3039. Confirmation seeds are 3400–3479. Neither range was
used for the earlier Adaptive or Aggressive evaluations.

    python3 experiments/eval_strength.py --first 3000 --last 3039 --workers 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from bot_definitions.adaptive_bot import AdaptiveBot
from bot_definitions.aggressive_bot import AggressiveBot
from bot_definitions.baselines import FractionValueBot, MyopicEvBot
from bot_definitions.eval_opponents import OverbidBot
from bot_definitions.strength_variants import OldAdaptive, OldAggressive, OldVL
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bots import Bot
from custom_bots import SmallBidBot
from eval_aggressive import _paired, format_report, play_rotation

Entry = tuple[str, type[Bot]]


def _field(hero: Entry, players: int) -> list[Entry]:
    names: list[Entry] = [
        hero,
        ("Fraction", FractionValueBot),
        ("SmallBid", SmallBidBot),
        ("Myopic", MyopicEvBot),
        ("Bot", Bot),
    ]
    return names[:players]


def build_jobs(seeds: list[int]) -> list[tuple[str, dict[str, object]]]:
    pairs: list[tuple[str, type[Bot], type[Bot]]] = [
        ("VL", ValueLiquidityBot, OldVL),
        ("Adaptive", AdaptiveBot, OldAdaptive),
        ("Agg", AggressiveBot, OldAggressive),
    ]
    jobs: list[tuple[str, dict[str, object]]] = []
    for players in (3, 4, 5):
        for name, new_cls, old_cls in pairs:
            for label, cls in ((name, new_cls), (f"Old{name}", old_cls)):
                jobs.append(
                    (
                        f"{players}p {label}",
                        {
                            "entries": _field((label, cls), players),
                            "seeds": seeds,
                            "match_seed": 8000 + players,
                            "focus": label,
                        },
                    )
                )
        fresh: list[Entry] = [
            ("VL", ValueLiquidityBot),
            ("Adaptive", AdaptiveBot),
            ("Agg", AggressiveBot),
            ("Myopic", MyopicEvBot),
            ("Fraction", FractionValueBot),
        ]
        stale: list[Entry] = [
            ("VL", OldVL),
            ("Adaptive", OldAdaptive),
            ("Agg", OldAggressive),
            ("Myopic", MyopicEvBot),
            ("Fraction", FractionValueBot),
        ]
        jobs.append(
            (
                f"{players}p together new",
                {
                    "entries": fresh[:players],
                    "seeds": seeds,
                    "match_seed": 8100 + players,
                    "focus": "VL",
                },
            )
        )
        jobs.append(
            (
                f"{players}p mirror new",
                {
                    "entries": [
                        ("Adaptive", AdaptiveBot),
                        ("VL", ValueLiquidityBot),
                        ("Myopic", MyopicEvBot),
                        ("Fraction", FractionValueBot),
                        ("Over", OverbidBot),
                    ][:players],
                    "seeds": seeds,
                    "match_seed": 8200 + players,
                    "focus": "Adaptive",
                },
            )
        )
        jobs.append(
            (
                f"{players}p mirror old",
                {
                    "entries": [
                        ("Adaptive", OldAdaptive),
                        ("VL", ValueLiquidityBot),
                        ("Myopic", MyopicEvBot),
                        ("Fraction", FractionValueBot),
                        ("Over", OverbidBot),
                    ][:players],
                    "seeds": seeds,
                    "match_seed": 8200 + players,
                    "focus": "Adaptive",
                },
            )
        )
        jobs.append(
            (
                f"{players}p over new",
                {
                    "entries": [
                        ("Adaptive", AdaptiveBot),
                        ("Over", OverbidBot),
                        ("VL", ValueLiquidityBot),
                        ("Fraction", FractionValueBot),
                        ("Myopic", MyopicEvBot),
                    ][:players],
                    "seeds": seeds,
                    "match_seed": 8300 + players,
                    "focus": "Adaptive",
                },
            )
        )
        jobs.append(
            (
                f"{players}p over old",
                {
                    "entries": [
                        ("Adaptive", OldAdaptive),
                        ("Over", OverbidBot),
                        ("VL", ValueLiquidityBot),
                        ("Fraction", FractionValueBot),
                        ("Myopic", MyopicEvBot),
                    ][:players],
                    "seeds": seeds,
                    "match_seed": 8300 + players,
                    "focus": "Adaptive",
                },
            )
        )
        jobs.append(
            (
                f"{players}p together old",
                {
                    "entries": stale[:players],
                    "seeds": seeds,
                    "match_seed": 8100 + players,
                    "focus": "VL",
                },
            )
        )
    return jobs


def run_job(spec: dict[str, object]) -> dict[str, object]:
    seeds = spec["seeds"]
    entries = spec["entries"]
    match_seed = spec["match_seed"]
    assert isinstance(seeds, list)
    assert isinstance(entries, list)
    assert isinstance(match_seed, int)
    report = play_rotation(entries, seeds, match_seed=match_seed)
    report["focus"] = spec["focus"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=3000)
    parser.add_argument("--last", type=int, default=3039)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("experiments/strength_results.txt"))
    parser.add_argument("--only", choices=("all", "loans"), default="all")
    args = parser.parse_args()
    seeds = list(range(args.first, args.last + 1))
    jobs = build_jobs(seeds)
    if args.only == "loans":
        kept = ("Agg", "OldAgg", "together new", "together old")
        jobs = [job for job in jobs if job[0].split(" ", 1)[1] in kept]
    print(f"{len(jobs)} matches, {len(seeds)} seeds", flush=True)
    reports: dict[str, dict[str, object]] = {}
    workers = args.workers or 8
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
        future_map = {pool.submit(run_job, spec): title for title, spec in jobs}
        done = 0
        for future in as_completed(future_map):
            title = future_map[future]
            reports[title] = future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {title} ({reports[title]['seconds']:.0f}s)", flush=True)
    chunks = [
        f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}). Charts A–E. Seats rotated.",
        "Paired gaps are new minus old on the same seeds. Positive favors the change.",
    ]
    for title, _spec in jobs:
        report = reports[title]
        focus = report.get("focus")
        chunks.append(format_report(title, report, focus if isinstance(focus, str) else None))
    for players in (3, 4, 5):
        for name in ("VL", "Adaptive", "Agg"):
            if f"{players}p {name}" not in reports:
                continue
            chunks.append(
                _paired(
                    f"Paired {name} − old on {players}p standard",
                    reports[f"{players}p {name}"],
                    reports[f"{players}p Old{name}"],
                    name,
                    f"Old{name}",
                )
            )
        if f"{players}p together new" not in reports:
            continue
        # Same-table comparison is not a hero swap: report each seat's mean
        # from the two matches, paired by seed.
        for name in ("VL", "Adaptive", "Agg"):
            chunks.append(
                _paired(
                    f"Paired together {name} {players}p",
                    reports[f"{players}p together new"],
                    reports[f"{players}p together old"],
                    name,
                    name,
                )
            )
        if f"{players}p mirror new" not in reports:
            continue
        chunks.append(
            _paired(
                f"Paired Adaptive vs VL {players}p",
                reports[f"{players}p mirror new"],
                reports[f"{players}p mirror old"],
                "Adaptive",
                "Adaptive",
            )
        )
        chunks.append(
            _paired(
                f"Paired Adaptive vs Over {players}p",
                reports[f"{players}p over new"],
                reports[f"{players}p over old"],
                "Adaptive",
                "Adaptive",
            )
        )
    text = "\n".join(chunks) + "\n"
    args.out.write_text(text)
    print(text)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
