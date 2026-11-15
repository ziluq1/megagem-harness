"""Pure value estimates and bid selection for analytic Megagem bots.

Terminal gem values depend only on the final value display. Every starting-hand
gem reaches that display; gems won at auction never do. Opponents' hidden gems
are a hypergeometric draw from ``unknown_gems`` (their hands plus the undealt
deck). The deck and the exposed auction gems are not part of the display.
"""

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence

from bots import BotInfo, PlayerInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from lib import AuctionCard, MissionCard

# Binomial coefficients for the small gem deck. n is at most 30.
_COMB: dict[tuple[int, int], int] = {}


def comb(n: int, k: int) -> int:
    """Return n choose k using integer arithmetic."""
    if k < 0 or n < 0 or k > n:
        return 0
    k = min(k, n - k)
    key = (n, k)
    cached = _COMB.get(key)
    if cached is not None:
        return cached
    value = 1
    for offset in range(k):
        value = value * (n - offset) // (offset + 1)
    _COMB[key] = value
    return value


def gem_count(mapping: Mapping[GemType, int], gem: GemType) -> int:
    """Return a gem count, treating a missing key as zero."""
    value = mapping.get(gem)
    return 0 if value is None else value


def affordable(info: BotInfo) -> int:
    """Maximum legal bid. Loan proceeds can be spent on that same loan."""
    card = info.current_auction
    coins = info.self_player.coins
    if card is not None and card.type is AuctionType.LOAN:
        return coins + card.value
    return max(0, coins)


def lot_gems(info: BotInfo) -> tuple[GemType, ...]:
    """Gems this treasure auction would award, first exposed gem first."""
    card = info.current_auction
    if card is None or card.type is not AuctionType.AUCTION:
        return ()
    return tuple(info.auction_gems[: card.value])


def priority_key(player_id: int, previous_winner_id: int, num_players: int) -> int:
    """Lower key wins a tie. Priority starts at the seat after the last winner."""
    return (player_id - previous_winner_id - 1) % num_players


def tie_win_probability(
    my_id: int,
    opponent_id: int,
    previous_winner_id: int | None,
    num_players: int,
) -> float:
    """Probability of beating one opponent when the bids are equal.

    The first auction draws uniformly among tied seats, so this bot does not
    count on winning that coin flip. Later ties are deterministic.
    """
    if previous_winner_id is None:
        return 0.0
    my_key = priority_key(my_id, previous_winner_id, num_players)
    opp_key = priority_key(opponent_id, previous_winner_id, num_players)
    if my_key < opp_key:
        return 1.0
    return 0.0


def min_covering_bid(info: BotInfo) -> int:
    """Smallest bid that beats every rival even if they bid their full budget."""
    card = info.current_auction
    loan = card.value if card is not None and card.type is AuctionType.LOAN else 0
    me = info.self_player.id
    previous = info.previous_winner_id
    need = 0
    for opponent in info.other_players:
        budget = opponent.coins + loan
        # Matching their max wins only when we have strict tie priority.
        if tie_win_probability(me, opponent.id, previous, info.num_players) >= 1.0:
            need = max(need, budget)
        else:
            need = max(need, budget + 1)
    return need


def final_count_distribution(info: BotInfo, color: GemType) -> list[float]:
    """Probability mass of the final display count for one color, indexes 0–6."""
    display = gem_count(info.value_display, color)
    mine = sum(gem == color for gem in info.self_player.hidden_gems)
    lower = display + mine
    opp_hidden = sum(player.hidden_gem_count for player in info.other_players)
    unknown_color = gem_count(info.unknown_gems, color)
    unknown_total = sum(info.unknown_gems.values())
    dist = [0.0] * 7

    def point_mass() -> list[float]:
        dist[min(6, max(0, lower))] = 1.0
        return dist

    if opp_hidden <= 0 or unknown_total <= 0 or unknown_color <= 0:
        return point_mass()
    draws = min(opp_hidden, unknown_total)
    successes = min(unknown_color, unknown_total)
    denom = comb(unknown_total, draws)
    if denom <= 0:
        return point_mass()
    low = max(0, draws - (unknown_total - successes))
    high = min(draws, successes)
    for extra in range(low, high + 1):
        ways = comb(successes, extra) * comb(unknown_total - successes, draws - extra)
        final = min(6, max(0, lower + extra))
        dist[final] += ways / denom
    total = sum(dist)
    if total <= 0.0:
        return point_mass()
    if abs(total - 1.0) > 1e-9:
        return [prob / total for prob in dist]
    return dist


def color_expectations(
    info: BotInfo, colors: Sequence[GemType]
) -> dict[GemType, float]:
    """Expected chart value of one collection gem of each requested color."""
    chart = info.value_chart
    values: dict[GemType, float] = {}
    for color in colors:
        if color in values:
            continue
        distribution = final_count_distribution(info, color)
        values[color] = sum(
            prob * chart.get_value(count) for count, prob in enumerate(distribution)
        )
    return values


def _filled(counts: Mapping[GemType, int]) -> dict[GemType, int]:
    return {gem: gem_count(counts, gem) for gem in STANDARD_GEMS}


def _with_lot(
    counts: Mapping[GemType, int], lot: Sequence[GemType]
) -> dict[GemType, int]:
    updated = _filled(counts)
    for gem in lot:
        updated[gem] += 1
    return updated


def mission_gain(
    counts: Mapping[GemType, int],
    lot: Sequence[GemType],
    missions: Sequence[MissionCard],
) -> int:
    """Points from missions this lot would newly complete."""
    if not lot or not missions:
        return 0
    before = _filled(counts)
    after = _with_lot(before, lot)
    gained = 0
    for mission in missions:
        if mission.is_satisfied(after) and not mission.is_satisfied(before):
            gained += mission.value
    return gained


def rival_only_mission_value(info: BotInfo, lot: Sequence[GemType]) -> int:
    """Largest mission haul a rival would newly score and we would not."""
    if not lot or not info.available_missions:
        return 0
    mine = _filled(info.self_player.won_gems)
    mine_after = _with_lot(mine, lot)
    my_new = {
        index
        for index, mission in enumerate(info.available_missions)
        if mission.is_satisfied(mine_after) and not mission.is_satisfied(mine)
    }
    best = 0
    for opponent in info.other_players:
        before = _filled(opponent.won_gems)
        after = _with_lot(before, lot)
        gained = 0
        for index, mission in enumerate(info.available_missions):
            if index in my_new:
                continue
            if mission.is_satisfied(after) and not mission.is_satisfied(before):
                gained += mission.value
        if gained > best:
            best = gained
    return best


def _stage_costs(
    gems_remaining: int, base_lam: float, base_delta: float
) -> tuple[float, float, float]:
    """Treasure lambda, investment lambda, and the positive-bid value hurdle.

    Near the end, saved coins are worth about one point, so the treasure
    premium and the value shade both go to zero. Investments still pay a
    small premium while a treasure remains, because the bid is locked.
    """
    if gems_remaining <= 2:
        invest_lam = 0.05 if base_lam > 0.0 else 0.0
        return 0.0, invest_lam, 0.0
    if gems_remaining <= 4:
        return base_lam * 0.25, base_lam * 0.5, base_delta * 0.5
    if gems_remaining <= 7:
        return base_lam * 0.7, base_lam, base_delta
    return base_lam, base_lam, base_delta


def _invest_lambda(info: BotInfo, lam_i: float) -> float:
    """Raise the cost of locking coins when the bankroll is already short."""
    if lam_i <= 0.0 or info.gems_remaining <= 0 or info.self_player.coins <= 0:
        return lam_i
    horizon = min(info.gems_remaining, 4)
    richest = max((opponent.coins for opponent in info.other_players), default=0)
    # TODO: tune — coins we still want per upcoming gem before locking an invest.
    per_gem = 1 if richest <= 3 else 2 if richest <= 8 else 3
    if info.self_player.coins <= horizon * per_gem:
        return max(lam_i, 1.1)
    return lam_i


def _invest_limit(coins: int, gems_remaining: int) -> int:
    """Leave one coin for a later treasure bid when any gems remain."""
    if gems_remaining <= 0 or coins <= 1:
        return max(0, coins)
    return coins - 1


def _loan_surplus(
    info: BotInfo,
    loan_value: int,
    bid: int,
    lam: float,
    showing_value: float,
) -> float:
    """Score cost of the bid, plus the option value of the extra coins.

    Principal is repaid at the end, so borrowed coins are not worth face
    value. Early, they are worth about ``lam`` each. Late, they are worth
    the gem surplus they can still buy.
    """
    extra = loan_value - bid
    if extra <= 0:
        return -float(bid)
    if info.gems_remaining > 4:
        return -float(bid) + lam * extra
    richest = max((opponent.coins for opponent in info.other_players), default=0)
    typical_price = 1.0 if richest <= 2 else min(6.0, 1.0 + 0.25 * richest)
    demand = typical_price * min(info.gems_remaining, 4)
    shortfall = max(0.0, demand - info.self_player.coins)
    useful = min(float(extra), shortfall)
    per_coin = max(0.0, (showing_value / typical_price) - 1.0)
    return -float(bid) + useful * per_coin


def _bid_distribution(
    cap: int, mean: float, scale: float, laplace_weight: float
) -> list[float]:
    """Mixture of a discrete Laplace around ``mean`` and a uniform hedge."""
    if cap <= 0:
        return [1.0]
    width = scale if scale > 1e-6 else 1.0
    weights = [math.exp(-abs(amount - mean) / width) for amount in range(cap + 1)]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / (cap + 1)] * (cap + 1)
    mix = min(1.0, max(0.0, laplace_weight))
    uniform = (1.0 - mix) / (cap + 1)
    return [mix * (weight / total) + uniform for weight in weights]


def _opponent_target(
    opponent: PlayerInfo,
    card: AuctionCard,
    naive: float,
    gem_value: float,
    rival_missions: int,
    bid_fraction: float,
    invest_lam: float,
) -> tuple[int, float]:
    """Modeled (cap, mean) for one opponent, already clamped to their budget."""
    loan_extra = card.value if card.type is AuctionType.LOAN else 0
    budget = opponent.coins + loan_extra
    if budget <= 0:
        return 0, 0.0
    if card.type is AuctionType.AUCTION:
        # Face value at the current display, our fair value, and contested missions.
        signal = max(0.0, 0.45 * naive + 0.55 * gem_value + 0.35 * rival_missions)
        cap = min(budget, max(0, math.ceil(signal)))
        mean = min(float(cap), bid_fraction * signal)
        return cap, mean
    if card.type is AuctionType.INVEST:
        bonus = float(card.value)
        # The bid is returned, so a sane opponent will bid past the bonus.
        reach = bonus / max(invest_lam, 0.25)
        cap = min(budget, max(0, math.ceil(max(bonus, reach))))
        mean = min(float(cap), 0.75 * bonus)
        return cap, mean
    principal = float(card.value)
    cap = min(budget, max(0, math.ceil(0.45 * principal)))
    mean = min(float(cap), 0.15 * principal)
    return cap, mean


def _win_curve(
    info: BotInfo,
    targets: Sequence[tuple[int, float, int]],
    limit: int,
    scale: float,
    laplace_weight: float,
) -> list[float]:
    """P(win) for each bid from 0 through ``limit``, under independent rivals."""
    distributions = [
        _bid_distribution(cap, mean, scale, laplace_weight)
        for cap, mean, _opp_id in targets
    ]
    cdfs: list[list[float]] = []
    for distribution in distributions:
        running = 0.0
        cdf: list[float] = []
        for prob in distribution:
            running += prob
            cdf.append(running)
        cdfs.append(cdf)
    me = info.self_player.id
    previous = info.previous_winner_id
    num_players = info.num_players
    curve: list[float] = []
    for bid in range(limit + 1):
        probability = 1.0
        for distribution, cdf, target in zip(distributions, cdfs, targets):
            _cap, _mean, opponent_id = target
            if bid >= len(distribution):
                continue
            below = cdf[bid - 1] if bid else 0.0
            tie = distribution[bid] * tie_win_probability(
                me, opponent_id, previous, num_players
            )
            probability *= below + tie
            if probability <= 0.0:
                break
        curve.append(probability)
    return curve


def select_bid(
    info: BotInfo,
    *,
    lam: float,
    delta: float,
    bid_fraction: float,
    bid_scale: float,
    laplace_weight: float,
    denial_weight: float,
    deadline: float | None = None,
) -> int:
    """Choose a legal bid maximizing liquidity-adjusted expected surplus.

    Treasure surplus is ``(value - hurdle) - (1 + lambda) * bid``. The hurdle
    is zero at a bid of zero, so a free positive gem is taken. Investments
    return the bid, so only the liquidity premium is charged. Loans charge the
    bid and credit the option value of the coins they add.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return 0
    card = info.current_auction
    if card is None:
        return 0
    limit_budget = affordable(info)
    if limit_budget <= 0:
        return 0

    lam_treasure, lam_invest, hurdle = _stage_costs(info.gems_remaining, lam, delta)
    lot = lot_gems(info)
    if card.type is AuctionType.LOAN:
        priced_colors: tuple[GemType, ...] = tuple(info.auction_gems)
    else:
        priced_colors = lot
    unit_values = color_expectations(info, priced_colors)
    gem_value = sum(unit_values.get(gem, 0.0) for gem in lot)
    naive = 0.0
    if lot:
        chart = info.value_chart
        naive = sum(
            chart.get_value(gem_count(info.value_display, gem)) for gem in lot
        )
    missions = mission_gain(
        info.self_player.won_gems, lot, info.available_missions
    )
    rival_missions = rival_only_mission_value(info, lot)
    denial = denial_weight * rival_missions
    showing_value = 0.0
    if card.type is AuctionType.LOAN and info.auction_gems:
        showing_value = sum(
            unit_values.get(gem, 0.0) for gem in info.auction_gems
        ) / len(info.auction_gems)

    market_invest_lam = lam_invest
    if card.type is AuctionType.INVEST:
        # Our own shortage raises the cost of locking coins. Rivals keep the
        # stage lambda, so a rich rival is not modeled as if they were short.
        lam_invest = _invest_lambda(info, lam_invest)
        limit = min(
            limit_budget,
            _invest_limit(info.self_player.coins, info.gems_remaining),
        )
    else:
        limit = limit_budget

    targets = [
        (
            *_opponent_target(
                opponent,
                card,
                naive,
                gem_value,
                rival_missions,
                bid_fraction,
                market_invest_lam,
            ),
            opponent.id,
        )
        for opponent in info.other_players
    ]
    curve = _win_curve(info, targets, limit, bid_scale, laplace_weight)

    best_bid = 0
    best_utility = 0.0
    for bid in range(limit + 1):
        if deadline is not None and bid % 8 == 0 and time.perf_counter() >= deadline:
            break
        probability = curve[bid]
        if card.type is AuctionType.AUCTION:
            shade = hurdle if bid > 0 else 0.0
            surplus = gem_value + missions + denial - shade - (1.0 + lam_treasure) * bid
        elif card.type is AuctionType.INVEST:
            surplus = card.value - lam_invest * bid
        else:
            surplus = _loan_surplus(info, card.value, bid, lam_treasure, showing_value)
        utility = probability * surplus
        if utility > best_utility + 1e-9:
            best_utility = utility
            best_bid = bid
    if best_bid < 0:
        return 0
    if best_bid > limit:
        return limit
    return best_bid


def choose_reveal_index(info: BotInfo) -> int:
    """Index of the hidden gem whose reveal helps us most.

    The final display is fixed, so the reveal only moves opponents' beliefs.
    Prefer to deflate a color still sitting in the auction and to avoid
    inflating one. Chart slope is read from the public display, which is what
    other players can see.
    """
    hidden = info.self_player.hidden_gems
    if not hidden:
        return 0
    chart = info.value_chart
    exposed = Counter(info.auction_gems)
    held = Counter(hidden)
    best_index = 0
    best_score = -1e300
    for index, gem in enumerate(hidden):
        display = gem_count(info.value_display, gem)
        slope = chart.get_value(min(6, display + 1)) - chart.get_value(display)
        showing = exposed[gem]
        score = -float(slope) * float(showing)
        if showing == 0:
            # Burning a color that is not on the block leaks less about price.
            score += 1.5
        if slope > 0:
            score -= 0.35 * float(held[gem])
        else:
            score += 0.15 * float(-slope)
        score += 0.05 * float(display)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index
