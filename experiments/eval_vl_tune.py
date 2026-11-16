"""Paired evaluation for ValueLiquidityBot tuning.

Protocol, for every config:
  * Tuning seeds are chosen by --first/--last. Held-out work uses 2000–2199
    and is never used to pick a constant.
  * Every seed is played on charts A–E, at 3, 4, and 5 players, with seats
    rotated. Extra seats repeat the last opponent so FrozenVL appears once.
  * Tables: frozen = [Candidate, FrozenVL, AdaptiveBot],
    myopic = [Candidate, MyopicEvBot, FractionValueBot],
    small = [Candidate, SmallBidBot, Bot].
  * Every table is paired by substitution. The deal is played twice with fresh
    bots: once with the candidate in the hero seat, once with FrozenVL there
    and the same opponents beside it. A cyclic seat order would leave the
    candidate always on the same side of FrozenVL, so same-game score gaps are
    not the comparison.
  * A seed contributes one number: its mean paired difference. The 95% CI is
    the normal approximation on those seed means (sample variance, n-1, 1.96).

    PYTHONPATH=src python3 experiments/eval_vl_tune.py --probe
    PYTHONPATH=src python3 experiments/eval_vl_tune.py --first 0 --last 39 \\
        --tables frozen --set USE_COVERING=1 --label covering
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from random import Random

from bot_definitions.adaptive_bot import AdaptiveBot
from bot_definitions.baselines import FractionValueBot, MyopicEvBot
from bot_definitions.frozen_vl import FrozenVL
from bot_definitions.small_bid_bot import SmallBidBot
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bot_runner import BotRunner
from bots import Bot
from constants import STANDARD_GEMS, AuctionType
from decks import VALUE_CHARTS
from harness import create_game, finish_game, play_round

import bot_definitions.value_liquidity_bot as vl

SWITCHES = (
    "USE_PHASE_LAMBDA",
    "PHASE_EARLY",
    "PHASE_MID",
    "PHASE_LATE",
    "PHASE_END",
    "LAM_P3",
    "LAM_P4",
    "LAM_P5",
    "INVEST_LAM_MULT",
    "CHART_LAM_A",
    "CHART_LAM_B",
    "CHART_LAM_C",
    "CHART_LAM_D",
    "CHART_LAM_E",
    "USE_LATE_FACE",
    "LATE_FACE_GEMS",
    "USE_COVERING",
    "COVER_MAX",
    "USE_LOAN_PLAN",
    "LOAN_HORIZON",
    "LOAN_CAPTURE",
    "USE_INVEST_NET",
    "SCARCITY_MODE",
    "SCARCITY_FACE_WEIGHT",
    "USE_MISSION_PROGRESS",
    "MISSION_PROGRESS_WEIGHT",
    "MISSION_BID_CAP",
    "USE_QUIET_GATE",
    "QUIET_BID",
    "QUIET_MIN_OBS",
    "QUIET_RATE",
    "REVEAL_MODE",
)
DEFAULTS = {name: getattr(vl, name) for name in SWITCHES}
# Instance parameters are copied from these class attributes in __init__.
# The bot does not read the environment.
CLASS_PARAMS = ("lam", "delta", "bid_fraction", "denial_weight")
SAVED_PARAMS = {name: getattr(vl, name) for name in CLASS_PARAMS}

TABLES: dict[str, list[tuple[str, type[Bot]]]] = {
    "frozen": [
        ("cand", ValueLiquidityBot),
        ("frozen", FrozenVL),
        ("adaptive", AdaptiveBot),
    ],
    "myopic": [
        ("cand", ValueLiquidityBot),
        ("myopic", MyopicEvBot),
        ("fraction", FractionValueBot),
    ],
    "small": [
        ("cand", ValueLiquidityBot),
        ("small", SmallBidBot),
        ("bot", Bot),
    ],
}
COMPONENTS = (
    "treasure_spend",
    "gem_value",
    "missions",
    "loan_repay",
    "investment_returns",
    "final_coins",
)

# Set in the parent before the pool is forked.
ACTIVE: list[str] = ["frozen", "myopic", "small"]


class InlineRunner(BotRunner):
    """In-process runner. Play stays in this process and keeps each bot's memory."""

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


def _fill(roles: list[tuple[str, type[Bot]]], players: int) -> list[tuple[str, type[Bot]]]:
    """Place every named opponent once, then repeat the last to fill the table."""
    hero, rest = roles[0], roles[1:]
    seated = [hero]
    for role in rest:
        if len(seated) >= players:
            break
        seated.append(role)
    while len(seated) < players:
        seated.append(rest[-1])
    return seated


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Normal 95% interval. ``values`` is one paired difference per seed."""
    count = len(values)
    mean = _mean(values)
    if count < 2:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    margin = 1.96 * (variance / count) ** 0.5
    return mean, mean - margin, mean + margin


def _empty_components() -> dict[str, int]:
    return {name: 0 for name in COMPONENTS}


def play_once(
    runners: list[InlineRunner], seed: int, chart_name: str
) -> tuple[list[int], set[int], list[dict[str, int]]]:
    """Play one deal. Returns scores, winner ids, and per-seat score parts."""
    chart = next(item for item in VALUE_CHARTS if item.name == chart_name)
    treasure = [0 for _ in runners]
    with create_game(
        len(runners),
        value_chart=chart,
        rng=Random(seed),
        bots=runners,
        bot_timeout=None,
    ) as game:
        while game.auction_gems:
            before = [player.coins for player in game.players]
            play_round(game)
            winner = game.previous_winner_id
            card = game.auction_history[-1]
            if winner is None:
                continue
            if card.type is AuctionType.AUCTION:
                treasure[winner] += before[winner] - game.players[winner].coins
        result = finish_game(game)
        gem_values = {
            gem: game.value_chart.get_value(game.value_display[gem])
            for gem in STANDARD_GEMS
        }
        parts: list[dict[str, int]] = []
        for seat, player in enumerate(game.players):
            gem_value = sum(
                player.gem_collection[gem] * gem_values[gem] for gem in STANDARD_GEMS
            )
            missions = sum(mission.value for mission in player.mission_cards)
            loan_repay = sum(
                -card.end_game_value
                for card in player.auction_cards
                if card.type is AuctionType.LOAN
            )
            invest_bonus = sum(
                card.end_game_value
                for card in player.auction_cards
                if card.type is AuctionType.INVEST
            )
            investment_returns = player.invested_coins + invest_bonus
            score = (
                player.coins
                + investment_returns
                + gem_value
                + missions
                - loan_repay
            )
            if score != result.scores[seat]:
                raise RuntimeError(
                    f"score identity failed at seat {seat}: {score} != {result.scores[seat]}"
                )
            parts.append(
                {
                    "treasure_spend": treasure[seat],
                    "gem_value": gem_value,
                    "missions": missions,
                    "loan_repay": loan_repay,
                    "investment_returns": investment_returns,
                    "final_coins": player.coins,
                }
            )
        return list(result.scores), set(result.winner_ids), parts


def _accumulate(
    bucket: dict[str, list[float]],
    key: str,
    value: float,
) -> None:
    bucket[key].append(value)


def _seat_runners(
    filled: list[tuple[str, type[Bot]]],
    order: list[int],
    hero: type[Bot],
    seed: int,
    salt: int,
) -> list[InlineRunner]:
    """Fresh bots for one deal. The hero class sits where ``cand`` was placed."""
    runners: list[InlineRunner] = []
    for seat, index in enumerate(order):
        name, cls = filled[index]
        use = hero if name == "cand" else cls
        runners.append(InlineRunner(use, seed=salt + seed * 100 + seat))
    return runners


def _play_table(seed: int, table_name: str) -> dict[str, object]:
    """Substitution pairs: same deal and seat, candidate versus FrozenVL."""
    roles = TABLES[table_name]
    diff_all: list[float] = []
    score_all: list[float] = []
    base_all: list[float] = []
    win_all: list[float] = []
    base_win_all: list[float] = []
    by_chart: dict[str, list[float]] = {chart.name: [] for chart in VALUE_CHARTS}
    chart_score: dict[str, list[float]] = {chart.name: [] for chart in VALUE_CHARTS}
    by_players: dict[str, list[float]] = {"3": [], "4": [], "5": []}
    player_score: dict[str, list[float]] = {"3": [], "4": [], "5": []}
    by_seat: dict[str, list[float]] = defaultdict(list)
    seat_score: dict[str, list[float]] = defaultdict(list)
    components: dict[str, list[float]] = {name: [] for name in COMPONENTS}
    for players in (3, 4, 5):
        filled = _fill(roles, players)
        for chart in VALUE_CHARTS:
            for shift in range(players):
                order = _rotate(list(range(players)), shift)
                hero_seat = order.index(0)
                cand_runners = _seat_runners(
                    filled, order, ValueLiquidityBot, seed, 11_000 + players
                )
                base_runners = _seat_runners(
                    filled, order, FrozenVL, seed, 21_000 + players
                )
                try:
                    scores, winners, parts = play_once(
                        cand_runners, seed, chart.name
                    )
                    scores_b, winners_b, parts_b = play_once(
                        base_runners, seed, chart.name
                    )
                finally:
                    for runner in cand_runners + base_runners:
                        runner.close()
                hero_score = float(scores[hero_seat])
                base_score = float(scores_b[hero_seat])
                diff = hero_score - base_score
                diff_all.append(diff)
                score_all.append(hero_score)
                base_all.append(base_score)
                win_all.append(1.0 if hero_seat in winners else 0.0)
                base_win_all.append(1.0 if hero_seat in winners_b else 0.0)
                by_chart[chart.name].append(diff)
                chart_score[chart.name].append(hero_score)
                by_players[str(players)].append(diff)
                player_score[str(players)].append(hero_score)
                by_seat[str(hero_seat)].append(diff)
                seat_score[str(hero_seat)].append(hero_score)
                for name in COMPONENTS:
                    components[name].append(
                        float(parts[hero_seat][name] - parts_b[hero_seat][name])
                    )
    return {
        "score": _mean(score_all),
        "base": _mean(base_all),
        "diff": _mean(diff_all),
        "win": _mean(win_all),
        "base_win": _mean(base_win_all),
        "games": len(diff_all) * 2,
        "charts": {name: _mean(values) for name, values in by_chart.items()},
        "chart_score": {name: _mean(values) for name, values in chart_score.items()},
        "players": {name: _mean(values) for name, values in by_players.items()},
        "player_score": {name: _mean(values) for name, values in player_score.items()},
        "seats": {name: _mean(values) for name, values in by_seat.items()},
        "seat_score": {name: _mean(values) for name, values in seat_score.items()},
        "components": {name: _mean(values) for name, values in components.items()},
    }


def eval_seed(seed: int) -> dict[str, object]:
    started = time.perf_counter()
    tables = {name: _play_table(seed, name) for name in ACTIVE}
    return {"seed": seed, "seconds": time.perf_counter() - started, "tables": tables}


def _column(rows: list[dict[str, object]], table: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        tables = row["tables"]
        assert isinstance(tables, dict)
        block = tables[table]
        assert isinstance(block, dict)
        value = block[key]
        assert isinstance(value, float)
        values.append(value)
    return values


def _nested(
    rows: list[dict[str, object]], table: str, key: str, inner: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        tables = row["tables"]
        assert isinstance(tables, dict)
        block = tables[table]
        assert isinstance(block, dict)
        mapping = block[key]
        assert isinstance(mapping, dict)
        value = mapping[inner]
        assert isinstance(value, float)
        values.append(value)
    return values


def _fmt(values: list[float]) -> str:
    mean, low, high = _ci95(values)
    side = "excludes 0" if low > 0 or high < 0 else "includes 0"
    return f"{mean:+7.3f}  [{low:+7.3f}, {high:+7.3f}]  {side}"


def format_report(label: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"\n== {label} ==",
        f"seeds {rows[0]['seed']}.. (n={len(rows)})  "
        "CI: normal approx on per-seed means, 1.96, variance n-1",
    ]
    seconds = sum(float(row["seconds"]) for row in rows)
    lines.append(f"worker-seconds {seconds:.1f}")
    for table in ACTIVE:
        diffs = _column(rows, table, "diff")
        scores = _column(rows, table, "score")
        base = _column(rows, table, "base")
        wins = _column(rows, table, "win")
        base_wins = _column(rows, table, "base_win")
        games = 0
        for row in rows:
            tables = row["tables"]
            assert isinstance(tables, dict)
            block = tables[table]
            assert isinstance(block, dict)
            games += int(block["games"])
        lines.append(
            f"\n[{table}] games {games}  cand mean {_mean(scores):.2f}  "
            f"frozen mean {_mean(base):.2f}  "
            f"cand win {_mean(wins):.1%}  frozen win {_mean(base_wins):.1%}"
        )
        lines.append(f"  paired score { _fmt(diffs) }")
        lines.append(f"  paired win   { _fmt([a - b for a, b in zip(wins, base_wins)]) }")
        lines.append("  by chart")
        for chart in VALUE_CHARTS:
            lines.append(
                f"    {chart.name}: score {_mean(_nested(rows, table, 'chart_score', chart.name)):.2f}  "
                f"diff {_fmt(_nested(rows, table, 'charts', chart.name))}"
            )
        lines.append("  by players")
        for count in ("3", "4", "5"):
            lines.append(
                f"    {count}p: score {_mean(_nested(rows, table, 'player_score', count)):.2f}  "
                f"diff {_fmt(_nested(rows, table, 'players', count))}"
            )
        lines.append("  by seat (candidate's seat)")
        seat_ids = sorted(
            {
                seat
                for row in rows
                for seat in (
                    row["tables"][table]["seats"].keys()  # type: ignore[index]
                )
            },
            key=int,
        )
        for seat in seat_ids:
            lines.append(
                f"    seat {seat}: score {_mean(_nested(rows, table, 'seat_score', seat)):.2f}  "
                f"diff {_fmt(_nested(rows, table, 'seats', seat))}"
            )
        lines.append("  paired components (candidate - frozen)")
        for name in COMPONENTS:
            lines.append(
                f"    {name:<20} {_fmt(_nested(rows, table, 'components', name))}"
            )
    return "\n".join(lines)


def _coerce(current: object, raw: str) -> object:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return raw
    raise SystemExit(f"cannot set a {type(current).__name__}")


_VL_INIT = ValueLiquidityBot.__init__


def apply_flags(pairs: list[tuple[str, str]]) -> None:
    """Restore shipped constants, then apply this config.

    ``lam`` and the other instance parameters are set only on
    ``ValueLiquidityBot`` instances. Subclasses keep the class defaults, which
    is what the removed environment overrides used to do.
    """
    for name, value in DEFAULTS.items():
        setattr(vl, name, value)
    for name, value in SAVED_PARAMS.items():
        setattr(vl, name, value)
    overrides: dict[str, float] = {}
    for key, raw in pairs:
        if key in SAVED_PARAMS:
            overrides[key] = float(raw)
            continue
        if not hasattr(vl, key):
            raise SystemExit(f"unknown switch {key}")
        setattr(vl, key, _coerce(getattr(vl, key), raw))

    def init(self: ValueLiquidityBot, _overrides: dict[str, float] = overrides) -> None:
        _VL_INIT(self)
        if type(self) is ValueLiquidityBot:
            for key, value in _overrides.items():
                setattr(self, key, value)

    ValueLiquidityBot.__init__ = init  # type: ignore[assignment]


def run_config(
    label: str,
    seeds: list[int],
    pairs: list[tuple[str, str]],
    workers: int,
) -> list[dict[str, object]]:
    apply_flags(pairs)
    print(f"\nrunning {label} on seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("fork")
    ) as pool:
        rows = list(pool.map(eval_seed, seeds))
    rows.sort(key=lambda row: int(row["seed"]))
    print(format_report(label, rows), flush=True)
    print(f"wall {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def _probe() -> None:
    started = time.perf_counter()
    runners = [
        InlineRunner(ValueLiquidityBot, seed=1),
        InlineRunner(FrozenVL, seed=2),
        InlineRunner(AdaptiveBot, seed=3),
    ]
    try:
        scores, winners, parts = play_once(runners, 2000, "A")
    finally:
        for runner in runners:
            runner.close()
    elapsed = time.perf_counter() - started
    print(
        f"probe 3p chart A seed 2000: scores {scores} winners {sorted(winners)} "
        f"in {elapsed:.2f}s"
    )
    print("components", parts)
    one = eval_seed(0)
    print(format_report("probe-seed-0", [one]))


def main() -> None:
    global ACTIVE
    parser = argparse.ArgumentParser(description="Tune ValueLiquidityBot")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=39)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tables", default="frozen,myopic,small")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--batch",
        type=Path,
        help="Lines of: label key=value key=value",
    )
    args = parser.parse_args()
    ACTIVE = [name.strip() for name in args.tables.split(",") if name.strip()]
    unknown = [name for name in ACTIVE if name not in TABLES]
    if unknown:
        raise SystemExit(f"unknown tables {unknown}")
    if args.probe:
        _probe()
        return
    seeds = list(range(args.first, args.last + 1))
    workers = args.workers or min(16, os.cpu_count() or 2)
    configs: list[tuple[str, list[tuple[str, str]]]] = []
    if args.batch:
        for line in args.batch.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bits = line.split()
            pairs = []
            for item in bits[1:]:
                key, separator, raw = item.partition("=")
                if not separator:
                    raise SystemExit(f"expected key=value in {item}")
                pairs.append((key, raw))
            configs.append((bits[0], pairs))
    else:
        pairs = []
        for item in args.set:
            key, separator, raw = item.partition("=")
            if not separator:
                raise SystemExit(f"expected key=value in {item}")
            pairs.append((key, raw))
        configs.append((args.label, pairs))
    for label, pairs in configs:
        run_config(label, seeds, pairs, workers)


if __name__ == "__main__":
    main()
