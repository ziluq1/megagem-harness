"""AdaptiveBot: in-game opponent model on top of ValueLiquidityBot.

Strategy
--------
The bid still maximises the base bot's liquidity-adjusted surplus

    P(win | b) * ((gem value + missions + denial - hurdle) - (1 + lambda) * b)

with the same investment and loan accounting as ``select_bid`` in
``pricing.py``. What changes is where P(win | b) and the parameters come from.
Three layers sit on that utility. Each is a class constant so it can be
ablated without editing the rest of the bot.

Layer 1 (``use_opponent_model``), the important one. ``get_bid`` caches this
lot's value signal, each opponent's coins, and the mean bid the model is about
to assume. ``on_round_end`` stores, per opponent, the validated bid, that
cached value, their coins at bid time, the auction type, and the round.
From that log the bot keeps a bid/value ratio (median, or a trimmed mean once
five observations exist), the rate of bidding 0, the rate of bidding their
whole coin stack, and the residuals around the mean that was predicted at the
time. Every estimate is shrunk toward ValueLiquidityBot's prior:

    (n * observed + k * prior) / (n + k)

with ``k = 8``, so two or three rounds barely move the prior. The predictive
distribution is that same shrinkage applied to the histogram of past
bid/value ratios, transferred onto the current lot and hard-capped at the
opponent's coins (plus this loan's proceeds). Until any evidence arrives the
distribution is exactly the base bot's Laplace-uniform mixture, so those
bids match it when tiebreaks are deterministic. P(win | b) multiplies one
factor per opponent. A factor is P(bid below) plus P(tie) times this seat's
tiebreak. The harness gives the seat after the previous winner the best
priority and leaves the previous winner last; the opening auction draws
uniformly among the tied seats, and that coin flip is priced exactly. A table
whose treasure ratio has settled below the prior is treated as passive and
the bid is shaded down toward the cheapest bid that still captures most of
the win probability. A table that has settled above the prior is treated as
aggressive: thin lots with no mission are not chased, and lots that complete
a mission are paid for.

Layer 2 (``use_portfolio``). Four parameter presets — conservative, balanced,
aggressive, mission — are scored by score margin (our score minus the mean of
the others). ``on_game_start`` picks one with UCB1. The first game, and every
game in which no previous result was checkpointed, uses the balanced preset,
which is ValueLiquidityBot's lambda, delta, and denial weight. Preset
statistics are about those presets only. They are not keyed by seat, because
a later game may sit different people in the same chairs. Old results are
decayed so a changed field can pull the choice back. If the process is fresh
every game, ``games_completed`` stays 0 and this layer does not change a bid.

Layer 3 (``use_jitter``). When several legal bids sit within a small utility
band of the best, one of them is drawn with ``random`` (the worker stream
seeded by ``--bot-seed``). A bid that is clearly best is played as itself.
The draw never leaves the budget and never selects a negative surplus.
The constant defaults to off: on seeds 1000–1199 the draw did not improve
on layers 1 and 2, and it gave a little of that gain back against ordinary
opponents. The code stays so the layer can be switched on for another trial.

Contributors: CONTRIBUTORS
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from bot_definitions.pricing import (
    _bid_distribution,
    _invest_lambda,
    _invest_limit,
    _loan_surplus,
    _opponent_target,
    _stage_costs,
    affordable,
    color_expectations,
    lot_gems,
    mission_gain,
    rival_only_mission_value,
    select_bid,
    tie_win_probability,
)
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bots import BotInfo, GameSummary, PlayerInfo, RoundResult
from constants import AuctionType, GemType

# bid, value signal, coins at bid time, auction-type name, round, predicted mean.
Observation = tuple[int, float, int, str, int, float]

# Shrinkage strength. n=3 still puts less than 30% weight on the sample.
SHRINK_K = 8.0
MAX_OBS = 24
# Treasure bid/value prior is ValueLiquidityBot.bid_fraction (0.55).
# These two match the invest and loan means in pricing._opponent_target.
INVEST_RATIO_PRIOR = 0.75
LOAN_RATIO_PRIOR = 0.15
ALLIN_PRIOR = 0.05
ZERO_PRIOR = 0.10
# Do not relabel a table on a handful of auctions.
MIN_CLASSIFY_OBS = 6
PASSIVE_RATIO = 0.40
AGGRESSIVE_RATIO = 0.80
# Cheapest bid that still holds this much of P(win) against a passive table.
COVER_PROBABILITY = 0.75
# Utility band, in expected points, that counts as a flat maximum.
JITTER_EPSILON = 0.25
# Per-game decay on preset counts and cumulative margins.
PRESET_DECAY = 0.96
# Ignore ratios on lots with almost no value signal; they explode.
MIN_VALUE = 1.0


def shrink(observed: float, prior: float, n: int, k: float = SHRINK_K) -> float:
    """Pull ``observed`` toward ``prior`` until ``n`` is large next to ``k``."""
    weight = float(n) + k
    if weight <= 0.0:
        return prior
    return (float(n) * observed + k * prior) / weight


def robust_center(values: Sequence[float]) -> float:
    """Median, or a 20% trimmed mean once five samples exist."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("robust_center requires at least one value")
    if count < 5:
        mid = count // 2
        if count % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])
    trim = max(1, count // 5)
    core = ordered[trim : count - trim]
    if not core:
        core = ordered
    return sum(core) / len(core)


def ucb1_index(
    counts: Sequence[float],
    totals: Sequence[float],
    *,
    balanced: int,
    visits: Sequence[int] | None = None,
) -> int:
    """UCB1 arm, exploring each arm once, then the best upper bound.

    A total play count of zero returns ``balanced`` rather than an arbitrary
    unplayed arm. That is the first-game preset: there is nothing to explore
    from yet. After that, any arm that has never been visited is tried before
    the confidence bounds are compared. ``visits`` is the undecayed play
    count, so a time-decayed ``counts`` entry cannot look untried.
    """
    if len(counts) != len(totals) or not counts:
        raise ValueError("UCB1 needs one count and one total per arm")
    if not 0 <= balanced < len(counts):
        raise ValueError("balanced preset index is out of range")
    if visits is not None and len(visits) != len(counts):
        raise ValueError("UCB1 visits must align with the arms")
    total_plays = sum(counts)
    explored = visits if visits is not None else None
    if explored is None and total_plays <= 1e-9:
        return balanced
    if explored is not None and sum(explored) <= 0 and total_plays <= 1e-9:
        return balanced
    pending = explored if explored is not None else counts
    # Integer visits explore each arm once. Without them, a decayed count
    # below one is treated as still untried.
    if explored is not None and sum(explored) > 0:
        for offset in range(1, len(pending) + 1):
            index = (balanced + offset) % len(pending)
            if explored[index] < 1:
                return index
    elif explored is None:
        for offset in range(1, len(pending) + 1):
            index = (balanced + offset) % len(pending)
            if counts[index] < 1.0:
                return index
    if total_plays <= 1e-9:
        return balanced
    log_term = math.log(max(total_plays, 1e-9))
    best_index = balanced
    best_score = -1e300
    for index, count in enumerate(counts):
        if count <= 1e-9:
            score = 1e150
        else:
            mean = totals[index] / count
            bonus = math.sqrt(2.0 * log_term / count)
            score = mean + bonus
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _renorm(dist: list[float]) -> list[float]:
    total = sum(dist)
    if total <= 0.0:
        if not dist:
            return [1.0]
        share = 1.0 / len(dist)
        return [share] * len(dist)
    if abs(total - 1.0) <= 1e-12:
        return dist
    return [value / total for value in dist]


def _pad(dist: Sequence[float], size: int) -> list[float]:
    values = list(dist[:size])
    if len(values) < size:
        values.extend(0.0 for _ in range(size - len(values)))
    return _renorm(values)


def _histogram(points: Sequence[float], budget: int) -> list[float]:
    """Point masses on the integers 0..budget. Values outside are clipped."""
    if budget <= 0:
        return [1.0]
    dist = [0.0] * (budget + 1)
    if not points:
        dist[0] = 1.0
        return dist
    for point in points:
        index = int(round(point))
        if index < 0:
            index = 0
        elif index > budget:
            index = budget
        dist[index] += 1.0
    return _renorm(dist)


def _blend(
    prior: Sequence[float], learned: Sequence[float], weight: float
) -> list[float]:
    """``weight`` is the share on ``learned``. Both inputs cover the same bids."""
    mix = min(1.0, max(0.0, weight))
    return _renorm(
        [
            (1.0 - mix) * left + mix * right
            for left, right in zip(prior, learned)
        ]
    )


def _impose_point(dist: Sequence[float], index: int, mass: float) -> list[float]:
    """Put ``mass`` on ``index`` and scale the other bids to keep a distribution."""
    if not dist:
        return [1.0]
    capped = min(0.95, max(0.0, mass))
    if index < 0 or index >= len(dist):
        return _renorm(list(dist))
    kept = [value for bid, value in enumerate(dist) if bid != index]
    rest = sum(kept)
    updated = list(dist)
    if rest <= 1e-12:
        share = (1.0 - capped) / max(1, len(dist) - 1)
        updated = [share] * len(dist)
    else:
        scale = (1.0 - capped) / rest
        updated = [value * scale for value in dist]
    updated[index] = capped
    return _renorm(updated)


def _beat_probability(
    bid: int,
    dist: Sequence[float],
    my_id: int,
    opponent_id: int,
    previous_winner_id: int | None,
    num_players: int,
    *,
    exact_opening: bool,
) -> float:
    """Probability a bid strictly wins, or wins the tie, against one opponent."""
    cap = len(dist) - 1
    if bid > cap:
        return 1.0
    below = sum(dist[:bid])
    if previous_winner_id is None and not exact_opening:
        # Match pricing.tie_win_probability: the opening coin flip is not banked on.
        return below
    tie_factor = tie_win_probability(
        my_id, opponent_id, previous_winner_id, num_players
    )
    return below + dist[bid] * tie_factor


def _opening_win_probability(
    bid: int, distributions: Sequence[Sequence[float]]
) -> float:
    """Exact P(win) when the harness draws uniformly among the tied seats.

    ``probs[k]`` is the probability that every opponent bid is at most ``bid``
    and exactly ``k`` of them equal ``bid``. Winning that configuration is
    the coin flip ``1 / (k + 1)``.
    """
    probs = [1.0]
    for dist in distributions:
        cap = len(dist) - 1
        if bid > cap:
            below, tie = 1.0, 0.0
        else:
            below = sum(dist[:bid])
            tie = dist[bid]
        updated = [0.0] * (len(probs) + 1)
        for tied, probability in enumerate(probs):
            updated[tied] += probability * below
            updated[tied + 1] += probability * tie
        probs = updated
    return sum(probability / (tied + 1) for tied, probability in enumerate(probs))


def win_curve(
    limit: int,
    distributions: Sequence[tuple[Sequence[float], int]],
    my_id: int,
    previous_winner_id: int | None,
    num_players: int,
    *,
    exact_opening: bool = True,
) -> list[float]:
    """P(win) for bids ``0`` through ``limit``. Non-decreasing in the bid.

    Deterministic tiebreaks factor across opponents: we win if and only if
    every opponent is below us, or tied and behind us in priority. The opening
    auction has no priority, so the tied set shares the win uniformly.
    """
    if limit < 0:
        return []
    if previous_winner_id is None and exact_opening:
        bare = [dist for dist, _opponent_id in distributions]
        return [_opening_win_probability(bid, bare) for bid in range(limit + 1)]
    curve: list[float] = []
    for bid in range(limit + 1):
        probability = 1.0
        for dist, opponent_id in distributions:
            probability *= _beat_probability(
                bid,
                dist,
                my_id,
                opponent_id,
                previous_winner_id,
                num_players,
                exact_opening=exact_opening,
            )
            if probability <= 0.0:
                break
        curve.append(probability)
    return curve


def jitter_bid(
    utilities: Sequence[float],
    best_bid: int,
    best_utility: float,
    epsilon: float,
) -> int:
    """Draw among legal non-negative bids inside the flat band, or keep ``best_bid``.

    ``best_utility`` is the tracked maximum and is never negative: the base
    search refuses a bid whose expected surplus does not beat zero. Candidates
    have to clear that same floor, so the draw cannot spend coins on a loss.
    """
    if not utilities:
        return 0
    limit = len(utilities) - 1
    if best_bid < 0 or best_bid > limit:
        best_bid = 0
    floor = best_utility - epsilon
    options = [
        bid
        for bid in range(limit + 1)
        if utilities[bid] >= floor - 1e-12 and utilities[bid] >= -1e-9
    ]
    if (
        best_bid not in options
        and 0 <= best_bid <= limit
        and utilities[best_bid] >= -1e-9
    ):
        options.append(best_bid)
    if len(options) <= 1:
        return best_bid
    return random.choice(options)


@dataclass(frozen=True)
class Preset:
    """Parameters of the base utility. Balanced matches ValueLiquidityBot."""

    name: str
    lam: float
    delta: float
    denial_weight: float
    mission_scale: float
    loan_scale: float


PRESETS: tuple[Preset, ...] = (
    Preset("conservative", 1.75, 2.5, 0.25, 0.85, 0.55),
    Preset("balanced", 1.0, 1.0, 0.40, 1.0, 1.0),
    Preset("aggressive", 0.35, 0.0, 0.55, 1.35, 1.45),
    Preset("mission", 1.15, 0.5, 0.70, 1.60, 0.80),
)
BALANCED_PRESET = 1


@dataclass
class OpponentTrack:
    """Per-opponent log for the current game. Capped so a checkpoint stays small."""

    observations: list[Observation] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)
    zero_flags: list[int] = field(default_factory=list)
    allin_flags: list[int] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)

    def record(
        self,
        bid: int,
        value: float,
        coins: int,
        kind: str,
        round_number: int,
        predicted_mean: float,
    ) -> None:
        self.observations.append(
            (bid, value, coins, kind, round_number, predicted_mean)
        )
        if len(self.observations) > MAX_OBS:
            del self.observations[: len(self.observations) - MAX_OBS]
        # Rebuild the estimate lists from the capped log so they cannot drift
        # apart. Treasure ratios stay on their own scale; loans and invests
        # are read by kind from ``observations``.
        self.ratios.clear()
        self.zero_flags.clear()
        self.allin_flags.clear()
        self.residuals.clear()
        for (
            seen_bid,
            seen_value,
            seen_coins,
            seen_kind,
            _seen_round,
            seen_mean,
        ) in self.observations:
            if seen_kind == "AUCTION" and seen_value >= MIN_VALUE:
                self.ratios.append(min(5.0, max(0.0, seen_bid / seen_value)))
            self.zero_flags.append(1 if seen_bid == 0 else 0)
            self.allin_flags.append(
                1 if seen_coins > 0 and seen_bid >= seen_coins else 0
            )
            self.residuals.append(float(seen_bid) - seen_mean)

    def kind_observations(self, kind: str) -> list[Observation]:
        return [obs for obs in self.observations if obs[3] == kind]


@dataclass
class RoundCache:
    """What get_bid knew, so on_round_end can label bids after coins move."""

    round_number: int
    kind: str
    value: float
    coins: dict[int, int]
    means: dict[int, float]


@dataclass
class LotContext:
    """Numbers shared by the surplus loop and the opponent model."""

    kind: str
    value: float
    gem_value: float
    naive: float
    missions: int
    rival_missions: int
    denial: float
    showing_value: float
    lam_treasure: float
    lam_invest: float
    market_invest_lam: float
    hurdle: float
    limit: int


def _kind_name(card_type: AuctionType) -> str:
    return card_type.name


def _prior_ratio(kind: str, bid_fraction: float) -> float:
    if kind == "INVEST":
        return INVEST_RATIO_PRIOR
    if kind == "LOAN":
        return LOAN_RATIO_PRIOR
    return bid_fraction


class AdaptiveBot(ValueLiquidityBot):
    """ValueLiquidityBot plus an opponent model and a preset portfolio.

    ``use_jitter`` is the third layer. It defaults to off after the paired
    evaluation; ``AdaptiveLayer13Bot`` turns it back on. Turning all three
    off is the base utility with ValueLiquidityBot's published parameters.
    """

    use_opponent_model = True
    use_portfolio = True
    use_jitter = False
    shrink_k = SHRINK_K
    jitter_epsilon = JITTER_EPSILON
    time_budget_seconds = 0.25

    def __init__(self) -> None:
        super().__init__()
        self.games_completed = 0
        self.preset_index = BALANCED_PRESET
        self.preset_counts = [0.0] * len(PRESETS)
        self.preset_totals = [0.0] * len(PRESETS)
        self.preset_visits = [0] * len(PRESETS)
        self.mission_scale = 1.0
        self.loan_scale = 1.0
        self.opponents: dict[int, OpponentTrack] = {}
        self.round_cache: RoundCache | None = None
        self._apply_preset(PRESETS[BALANCED_PRESET])

    def _apply_preset(self, preset: Preset) -> None:
        self.lam = preset.lam
        self.delta = preset.delta
        self.denial_weight = preset.denial_weight
        self.mission_scale = preset.mission_scale
        self.loan_scale = preset.loan_scale

    def _reset_game(self) -> None:
        # Seat ids are only meaningful inside this game. Cross-game memory is
        # the preset counts, which are not stored here.
        self.opponents = {}
        self.round_cache = None

    def on_game_start(self) -> None:
        try:
            self._reset_game()
            if (not self.use_portfolio) or self.games_completed <= 0:
                self.preset_index = BALANCED_PRESET
            else:
                self.preset_index = ucb1_index(
                    self.preset_counts,
                    self.preset_totals,
                    balanced=BALANCED_PRESET,
                    visits=self.preset_visits,
                )
            self._apply_preset(PRESETS[self.preset_index])
        except Exception:
            self.preset_index = BALANCED_PRESET
            self._apply_preset(PRESETS[BALANCED_PRESET])
            self.opponents = {}
            self.round_cache = None

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        try:
            if not self.use_portfolio:
                return
            mine = float(result.scores[info.self_player.id])
            others = [
                float(score)
                for player_id, score in result.scores.items()
                if player_id != info.self_player.id
            ]
            margin = mine - (sum(others) / len(others) if others else 0.0)
            decay = PRESET_DECAY
            self.preset_counts = [count * decay for count in self.preset_counts]
            self.preset_totals = [total * decay for total in self.preset_totals]
            index = self.preset_index
            if not 0 <= index < len(PRESETS):
                index = BALANCED_PRESET
            self.preset_counts[index] += 1.0
            self.preset_totals[index] += margin
            self.preset_visits[index] = self.preset_visits[index] + 1
            self.games_completed += 1
        except Exception:
            return

    def get_bid(self, info: BotInfo) -> int:
        started = time.perf_counter()
        deadline = started + self.time_budget_seconds
        try:
            if info.current_auction is None:
                self.round_cache = None
                return 0
            context = self._lot_context(info)
            if context is None or context.limit <= 0:
                self._remember_round(info, context, {})
                return 0
            bid = self._choose_bid(info, context, deadline)
            if bid < 0:
                return 0
            if bid > context.limit:
                return context.limit
            return bid
        except Exception:
            self.round_cache = None
            try:
                return select_bid(
                    info,
                    lam=self.lam,
                    delta=self.delta,
                    bid_fraction=self.bid_fraction,
                    bid_scale=self.bid_scale,
                    laplace_weight=self.laplace_weight,
                    denial_weight=self.denial_weight,
                    deadline=deadline,
                )
            except Exception:
                return 0

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        try:
            if not self.use_opponent_model:
                return
            cache = self.round_cache
            # Coins in ``info`` are post-resolution. The cache is the bid-time
            # snapshot from get_bid, and it is only valid for this round.
            if cache is None or cache.round_number != info.round_number:
                return
            kind = _kind_name(result.auction_card.type)
            if kind != cache.kind:
                return
            me = info.self_player.id
            for player_id, bid in result.bids.items():
                if player_id == me or player_id not in cache.coins:
                    continue
                track = self.opponents.get(player_id)
                if track is None:
                    track = OpponentTrack()
                    self.opponents[player_id] = track
                track.record(
                    int(bid),
                    cache.value,
                    cache.coins[player_id],
                    kind,
                    cache.round_number,
                    cache.means.get(player_id, 0.0),
                )
            self.round_cache = None
        except Exception:
            self.round_cache = None

    def ratio_estimate(self, opponent_id: int, kind: str = "AUCTION") -> float:
        """Shrunk bid/value ratio. The prior when this opponent has no sample."""
        prior = _prior_ratio(kind, self.bid_fraction)
        track = self.opponents.get(opponent_id)
        if track is None:
            return prior
        ratios = [
            bid / value
            for bid, value, _coins, _obs_kind, _round, _mean in track.kind_observations(
                kind
            )
            if value >= MIN_VALUE
        ]
        if not ratios:
            return prior
        return shrink(robust_center(ratios), prior, len(ratios), self.shrink_k)

    def zero_probability(self, opponent_id: int) -> float:
        track = self.opponents.get(opponent_id)
        if track is None or not track.zero_flags:
            return ZERO_PRIOR
        observed = sum(track.zero_flags) / len(track.zero_flags)
        return shrink(observed, ZERO_PRIOR, len(track.zero_flags), self.shrink_k)

    def allin_probability(self, opponent_id: int) -> float:
        track = self.opponents.get(opponent_id)
        if track is None or not track.allin_flags:
            return ALLIN_PRIOR
        observed = sum(track.allin_flags) / len(track.allin_flags)
        return shrink(observed, ALLIN_PRIOR, len(track.allin_flags), self.shrink_k)

    def _lot_context(self, info: BotInfo) -> LotContext | None:
        card = info.current_auction
        if card is None:
            return None
        limit_budget = affordable(info)
        lam_treasure, lam_invest, hurdle = _stage_costs(
            info.gems_remaining, self.lam, self.delta
        )
        lot = lot_gems(info)
        priced: tuple[GemType, ...]
        if card.type is AuctionType.LOAN:
            priced = tuple(info.auction_gems)
        else:
            priced = lot
        unit_values = color_expectations(info, priced)
        gem_value = sum(unit_values.get(gem, 0.0) for gem in lot)
        naive = 0.0
        if lot:
            chart = info.value_chart
            naive = sum(
                chart.get_value(info.value_display.get(gem, 0)) for gem in lot
            )
        missions = mission_gain(info.self_player.won_gems, lot, info.available_missions)
        rival_missions = rival_only_mission_value(info, lot)
        denial = self.denial_weight * rival_missions
        showing_value = 0.0
        if card.type is AuctionType.LOAN and info.auction_gems:
            showing_value = sum(
                unit_values.get(gem, 0.0) for gem in info.auction_gems
            ) / len(info.auction_gems)
        market_invest_lam = lam_invest
        if card.type is AuctionType.INVEST:
            lam_invest = _invest_lambda(info, lam_invest)
            limit = min(
                limit_budget,
                _invest_limit(info.self_player.coins, info.gems_remaining),
            )
        else:
            limit = limit_budget
        if card.type is AuctionType.AUCTION:
            value = max(0.0, 0.45 * naive + 0.55 * gem_value + 0.35 * rival_missions)
        else:
            value = float(card.value)
        return LotContext(
            kind=_kind_name(card.type),
            value=value,
            gem_value=gem_value,
            naive=naive,
            missions=missions,
            rival_missions=rival_missions,
            denial=denial,
            showing_value=showing_value,
            lam_treasure=lam_treasure,
            lam_invest=lam_invest,
            market_invest_lam=market_invest_lam,
            hurdle=hurdle,
            limit=limit,
        )

    def _remember_round(
        self,
        info: BotInfo,
        context: LotContext | None,
        means: dict[int, float],
    ) -> None:
        card = info.current_auction
        if card is None or context is None:
            self.round_cache = None
            return
        self.round_cache = RoundCache(
            round_number=info.round_number,
            kind=context.kind,
            value=context.value,
            coins={opponent.id: opponent.coins for opponent in info.other_players},
            means=means,
        )

    def _distribution(
        self,
        info: BotInfo,
        opponent: PlayerInfo,
        context: LotContext,
    ) -> tuple[list[float], float]:
        """Predictive bid distribution on ``0..budget``, and its mean.

        With no usable observations this is the base bot's mixture, padded
        with zeros out to the legal budget so a rival is never given a bid
        above their coins. With observations, past bid/value ratios are
        replayed on this lot (those are the residuals around the current
        mean) and shrunk toward that prior by ``n / (n + k)``.
        """
        card = info.current_auction
        assert card is not None
        coins = opponent.coins
        loan_extra = card.value if card.type is AuctionType.LOAN else 0
        budget = max(0, coins + loan_extra)
        prior_cap, prior_mean = _opponent_target(
            opponent,
            card,
            context.naive,
            context.gem_value,
            context.rival_missions,
            self.bid_fraction,
            context.market_invest_lam,
        )
        if budget <= 0:
            return [1.0], 0.0
        prior = _pad(
            _bid_distribution(
                prior_cap, prior_mean, self.bid_scale, self.laplace_weight
            ),
            budget + 1,
        )
        if not self.use_opponent_model:
            return prior, _mean_of(prior)
        track = self.opponents.get(opponent.id)
        usable: list[float] = []
        if track is not None and context.value >= MIN_VALUE:
            for (
                bid,
                value,
                _coins,
                _kind,
                _round,
                _mean,
            ) in track.kind_observations(context.kind):
                if value >= MIN_VALUE:
                    # bid/value * this lot's value is the old bid, moved onto
                    # the current mean. That is the residual around the ratio.
                    usable.append(bid / value * context.value)
        if not usable:
            return prior, _mean_of(prior)
        learned = _histogram(usable, budget)
        weight = len(usable) / (len(usable) + self.shrink_k)
        mixed = _blend(prior, learned, weight)
        raw_allin = 0.0
        if track is not None and track.allin_flags:
            raw_allin = sum(track.allin_flags) / len(track.allin_flags)
        allin_at = coins if 0 < coins <= budget else None
        if (
            allin_at is not None
            and track is not None
            and len(track.allin_flags) >= MIN_CLASSIFY_OBS
        ):
            pall = shrink(raw_allin, ALLIN_PRIOR, len(track.allin_flags), self.shrink_k)
            if pall > 0.12:
                mixed = _impose_point(mixed, allin_at, pall)
        return mixed, _mean_of(mixed)

    def _table_stance(self, info: BotInfo) -> str:
        if not self.use_opponent_model:
            return "fair"
        labels: list[str] = []
        for opponent in info.other_players:
            track = self.opponents.get(opponent.id)
            sample = 0
            if track is not None:
                sample = sum(
                    1
                    for _bid, value, _coins, _kind, _round, _mean in (
                        track.kind_observations("AUCTION")
                    )
                    if value >= MIN_VALUE
                )
            if sample < MIN_CLASSIFY_OBS:
                labels.append("fair")
                continue
            estimate = self.ratio_estimate(opponent.id, "AUCTION")
            if estimate < PASSIVE_RATIO:
                labels.append("passive")
            elif estimate > AGGRESSIVE_RATIO:
                labels.append("aggressive")
            else:
                labels.append("fair")
        if not labels:
            return "fair"
        if "aggressive" not in labels and labels.count("passive") * 2 >= len(labels):
            return "passive"
        if labels.count("aggressive") * 2 >= len(labels):
            return "aggressive"
        return "fair"

    def _choose_bid(self, info: BotInfo, context: LotContext, deadline: float) -> int:
        card = info.current_auction
        assert card is not None
        distributions: list[tuple[list[float], int]] = []
        means: dict[int, float] = {}
        expected_price = 0.0
        for opponent in info.other_players:
            if time.perf_counter() >= deadline:
                break
            dist, mean = self._distribution(info, opponent, context)
            distributions.append((dist, opponent.id))
            means[opponent.id] = mean
            if mean > expected_price:
                expected_price = mean
        self._remember_round(info, context, means)
        if len(distributions) != len(info.other_players):
            # Deadline cut the model short. Fall back to a prior-only bid.
            return select_bid(
                info,
                lam=self.lam,
                delta=self.delta,
                bid_fraction=self.bid_fraction,
                bid_scale=self.bid_scale,
                laplace_weight=self.laplace_weight,
                denial_weight=self.denial_weight,
                deadline=deadline,
            )
        lam_treasure = context.lam_treasure
        hurdle = context.hurdle
        stance = self._table_stance(info)
        if stance == "aggressive":
            if context.missions > 0:
                lam_treasure *= 0.55
                hurdle *= 0.40
            elif context.gem_value + context.denial < expected_price:
                hurdle += 2.0
        exact_opening = self.use_opponent_model
        curve = win_curve(
            context.limit,
            distributions,
            info.self_player.id,
            info.previous_winner_id,
            info.num_players,
            exact_opening=exact_opening,
        )
        utilities = self._utilities(info, context, curve, lam_treasure, hurdle)
        best_bid, best_utility = _argmax_utility(utilities)
        if stance == "passive":
            best_bid, best_utility, cover, shaded = _shade_passive(
                utilities, curve, best_bid, best_utility
            )
            if shaded:
                # Jitter must not climb back above the covering bid.
                utilities = [
                    utility if bid <= cover else -1.0
                    for bid, utility in enumerate(utilities)
                ]
        if self.use_jitter and best_utility > 0.0:
            return jitter_bid(utilities, best_bid, best_utility, self.jitter_epsilon)
        if best_bid < 0:
            return 0
        if best_bid > context.limit:
            return context.limit
        return best_bid

    def _utilities(
        self,
        info: BotInfo,
        context: LotContext,
        curve: Sequence[float],
        lam_treasure: float,
        hurdle: float,
    ) -> list[float]:
        card = info.current_auction
        assert card is not None
        utilities: list[float] = []
        for bid, probability in enumerate(curve):
            if card.type is AuctionType.AUCTION:
                shade = hurdle if bid > 0 else 0.0
                surplus = (
                    context.gem_value
                    + self.mission_scale * context.missions
                    + context.denial
                    - shade
                    - (1.0 + lam_treasure) * bid
                )
            elif card.type is AuctionType.INVEST:
                surplus = card.value - context.lam_invest * bid
            else:
                raw = _loan_surplus(
                    info,
                    card.value,
                    bid,
                    lam_treasure,
                    context.showing_value,
                )
                if self.loan_scale != 1.0:
                    benefit = raw + float(bid)
                    raw = -float(bid) + self.loan_scale * benefit
                surplus = raw
            utilities.append(probability * surplus)
        return utilities


def _mean_of(dist: Sequence[float]) -> float:
    return sum(bid * prob for bid, prob in enumerate(dist))


def _argmax_utility(utilities: Sequence[float]) -> tuple[int, float]:
    """Lowest bid whose surplus beats zero and every cheaper bid.

    Matches ``select_bid``: a negative expected surplus never replaces a bid
    of zero, and ties keep the cheaper bid.
    """
    best_bid = 0
    best_utility = 0.0
    for bid, utility in enumerate(utilities):
        if utility > best_utility + 1e-9:
            best_utility = utility
            best_bid = bid
    return best_bid, best_utility


def _shade_passive(
    utilities: Sequence[float],
    curve: Sequence[float],
    best_bid: int,
    best_utility: float,
) -> tuple[int, float, int, bool]:
    """Drop toward the cheapest bid that still holds most of P(win).

    The bool is false when the cheap bid gives away too much expected surplus,
    in which case the original bid is kept and higher bids stay eligible.
    """
    cover = len(curve) - 1
    for bid, probability in enumerate(curve):
        if probability + 1e-12 >= COVER_PROBABILITY:
            cover = bid
            break
    local_bid = 0
    local_utility = 0.0
    for bid, utility in enumerate(utilities[: cover + 1]):
        if utility > local_utility + 1e-9:
            local_utility = utility
            local_bid = bid
    # Keep the higher bid when the cheap one actually gives the lot away.
    if (
        best_utility > local_utility + 0.75
        and curve[best_bid] > curve[local_bid] + 0.08
    ):
        return best_bid, best_utility, cover, False
    return local_bid, local_utility, cover, True


class AdaptiveLayer1Bot(AdaptiveBot):
    """Ablation: opponent model only. Explicit flags, independent of the parent."""

    use_opponent_model = True
    use_portfolio = False
    use_jitter = False


class AdaptiveLayer12Bot(AdaptiveBot):
    """Ablation: opponent model and the cross-game portfolio."""

    use_opponent_model = True
    use_portfolio = True
    use_jitter = False


class AdaptiveLayer13Bot(AdaptiveBot):
    """Ablation: opponent model and bid jitter."""

    use_opponent_model = True
    use_portfolio = False
    use_jitter = True


class AdaptiveAllOffBot(AdaptiveBot):
    """Ablation: every adaptive layer disabled. The base utility remains."""

    use_opponent_model = False
    use_portfolio = False
    use_jitter = False
