"""ValueLiquidityBot: sealed-bid surplus with a fixed coin shadow price.

A bid maximizes ``P(win | b)`` times surplus. Surplus is the chart value of
the lot, plus missions it completes, plus a capped mission-progress credit,
minus a one-coin shade on any positive bid, minus ``(1 + lambda)`` times the
bid. ``lambda`` is 1.0 and ``delta`` is 1.0. Rival bids are a Laplace-uniform
mixture: mean 0.55 of a public value signal, scale 2, weight 0.8, and denial
weight 0.4. A lot that only advances a mission adds at most 1 coin, at 0.35
of the reward times the share of the requirement it fills. After 4 rival
bids, if 70% were 1 coin or less, that progress credit turns off. The reveal
is strategic. It does not change the final display.

The rival mixture is not learned, so a table of very cheap bidders can still
take gems this bot shades away from. Opening ties are priced as losses.
Phase lambda, covering, the loan plan, and the invest discount stay off.
Tuned bidding runs only when ``type(self) is ValueLiquidityBot``.

Contributors: @ziluq1
"""

import math
import random
import time
from collections.abc import Mapping, Sequence

from bot_definitions.pricing import (
    _invest_lambda,
    _invest_limit,
    _loan_surplus,
    _opponent_target,
    _stage_costs,
    _win_curve,
    affordable,
    choose_reveal_index,
    color_expectations,
    final_count_distribution,
    gem_count,
    lot_gems,
    min_covering_bid,
    mission_gain,
    rival_only_mission_value,
    select_bid,
)
from bots import Bot, BotInfo, GameSummary, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import MissionCard


# --- feature switches -------------------------------------------------------
# Identity settings stay put. Only a held-out gain turns a switch on.

# Scale the staged treasure/invest lambda. 1.0 leaves ``_stage_costs`` alone.
USE_PHASE_LAMBDA = False
PHASE_EARLY = 1.0  # gems_remaining > 7
PHASE_MID = 1.0  # 5–7 gems left
PHASE_LATE = 1.0  # 3–4 gems left
PHASE_END = 1.0  # 0–2 gems left; the staged treasure premium is already 0
LAM_P3 = 1.0
LAM_P4 = 1.0
LAM_P5 = 1.0

# Scales only the premium on locking coins in an investment. 1.0 is the stage value.
INVEST_LAM_MULT = 1.0
# Scales only the treasure premium, by chart. Scarcity charts can stay dearer.
CHART_LAM_A = 1.0
CHART_LAM_B = 1.0
CHART_LAM_C = 1.0
CHART_LAM_D = 1.0
CHART_LAM_E = 1.0

# Spend cash at face value once this many gems remain (lambda and hurdle 0).
USE_LATE_FACE = False
LATE_FACE_GEMS = 4

# Bid the minimum covering price when it is below the lot's value.
USE_COVERING = False
# Do not pay a covering price above this many coins. A wide cap overpays.
COVER_MAX = 100

# Loan bids come from a multi-lot plan instead of the shadow-price shortcut.
USE_LOAN_PLAN = False
# How many not-yet-exposed gems the plan is allowed to count on.
LOAN_HORIZON = 4
# Share of the planned gem margin we treat as capturable. 1.0 is the full plan.
LOAN_CAPTURE = 1.0

# Investment surplus is the bonus minus the treasure margin those coins lose.
USE_INVEST_NET = False

# Chart B/D adjustment. "off", "conservative" (value at ceil of expected count),
# or "face" (blend toward the printed display value).
SCARCITY_MODE = "off"
SCARCITY_FACE_WEIGHT = 0.35

# Partial credit for missions this lot advances but does not finish.
USE_MISSION_PROGRESS = True
MISSION_PROGRESS_WEIGHT = 0.35
# How many coins above the no-progress bid that credit may spend.
# A wide cap overpays when rivals are not actually bidding.
MISSION_BID_CAP = 1
# After a few auctions, drop the progress premium if rivals are bidding 0 or 1.
USE_QUIET_GATE = True
QUIET_BID = 1
QUIET_MIN_OBS = 4
QUIET_RATE = 0.7

# "strategic" is choose_reveal_index. "first" reveals index 0. "random" draws.
REVEAL_MODE = "strategic"


def _phase_multiplier(gems_remaining: int) -> float:
    """Multiplier for the staged lambda. Identity unless a phase is retuned."""
    if gems_remaining <= 2:
        return PHASE_END
    if gems_remaining <= 4:
        return PHASE_LATE
    if gems_remaining <= 7:
        return PHASE_MID
    return PHASE_EARLY


def _player_multiplier(num_players: int) -> float:
    if num_players == 3:
        return LAM_P3
    if num_players == 4:
        return LAM_P4
    if num_players == 5:
        return LAM_P5
    return 1.0


def tuned_lambda(info: BotInfo, lam: float, delta: float) -> tuple[float, float]:
    """Lambda and hurdle inputs after the phase, seat-count, and late-cash switches.

    Late face value wins over the phase multiplier: once few gems remain, a
    coin saved and a coin spent are the same point.
    """
    if USE_LATE_FACE and info.gems_remaining <= LATE_FACE_GEMS:
        return 0.0, 0.0
    if USE_PHASE_LAMBDA:
        lam *= _phase_multiplier(info.gems_remaining) * _player_multiplier(
            info.num_players
        )
    return lam, delta


def _chart_treasure_mult(info: BotInfo) -> float:
    return {
        "A": CHART_LAM_A,
        "B": CHART_LAM_B,
        "C": CHART_LAM_C,
        "D": CHART_LAM_D,
        "E": CHART_LAM_E,
    }.get(info.value_chart.name, 1.0)


def _split_lambda_active() -> bool:
    return INVEST_LAM_MULT != 1.0 or _any_chart_lambda()


def _any_chart_lambda() -> bool:
    return (
        CHART_LAM_A != 1.0
        or CHART_LAM_B != 1.0
        or CHART_LAM_C != 1.0
        or CHART_LAM_D != 1.0
        or CHART_LAM_E != 1.0
    )


def _custom_features_enabled() -> bool:
    return (
        USE_LOAN_PLAN
        or USE_INVEST_NET
        or USE_MISSION_PROGRESS
        or SCARCITY_MODE != "off"
        or _split_lambda_active()
    )


def scarcity_adjust(
    info: BotInfo,
    lot: Sequence[GemType],
    gem_value: float,
    unit_values: Mapping[GemType, float],
) -> float:
    """Chart B/D value under the active scarcity mode. Other charts are unchanged."""
    del unit_values
    if SCARCITY_MODE == "off" or not lot or info.value_chart.name not in ("B", "D"):
        return gem_value
    chart = info.value_chart
    if SCARCITY_MODE == "conservative":
        total = 0.0
        for gem in lot:
            distribution = final_count_distribution(info, gem)
            mean = sum(count * prob for count, prob in enumerate(distribution))
            count = min(6, int(math.ceil(mean - 1e-12)))
            total += float(chart.get_value(count))
        return total
    if SCARCITY_MODE == "face":
        face = sum(
            float(chart.get_value(gem_count(info.value_display, gem))) for gem in lot
        )
        weight = min(1.0, max(0.0, SCARCITY_FACE_WEIGHT))
        return (1.0 - weight) * gem_value + weight * face
    return gem_value


def _progress_fraction(
    mission: MissionCard,
    before: Mapping[GemType, int],
    after: Mapping[GemType, int],
) -> float:
    """Share of one unfinished mission this lot newly covers. At most 1."""
    required = mission.gem_requirements
    wild = required.get(GemType.WILD_CARD, 0)
    if wild <= 0:
        need = 0
        gained = 0
        for gem, count in required.items():
            if count <= 0 or gem is GemType.WILD_CARD:
                continue
            need += count
            gained += min(count, after.get(gem, 0)) - min(count, before.get(gem, 0))
        if need <= 0 or gained <= 0:
            return 0.0
        return min(1.0, gained / need)
    kind = mission.wild_card_requirement
    if kind is WildCardRequirement.SAME:
        before_max = max(before.values(), default=0)
        after_max = max(after.values(), default=0)
        gained = min(wild, after_max) - min(wild, before_max)
        return max(0.0, min(1.0, gained / wild))
    if kind is WildCardRequirement.DIFFERENT:
        before_colors = sum(count > 0 for count in before.values())
        after_colors = sum(count > 0 for count in after.values())
        gained = min(wild, after_colors) - min(wild, before_colors)
        return max(0.0, min(1.0, gained / wild))
    if kind is WildCardRequirement.PAIRS:
        before_pairs = sum(count // 2 for count in before.values())
        after_pairs = sum(count // 2 for count in after.values())
        need_pairs = max(1, wild // 2)
        gained = min(need_pairs, after_pairs) - min(need_pairs, before_pairs)
        return max(0.0, min(1.0, gained / need_pairs))
    return 0.0


def mission_progress_value(info: BotInfo, lot: Sequence[GemType]) -> float:
    """Partial mission value for progress that does not finish the card.

    A mission this lot completes is left to ``mission_gain``, so it is not
    counted twice. Each remaining mission contributes at most ``weight`` times
    its full reward.
    """
    if not USE_MISSION_PROGRESS or not lot or not info.available_missions:
        return 0.0
    weight = MISSION_PROGRESS_WEIGHT
    if weight <= 0.0:
        return 0.0
    before = {gem: gem_count(info.self_player.won_gems, gem) for gem in STANDARD_GEMS}
    after = dict(before)
    for gem in lot:
        after[gem] = after.get(gem, 0) + 1
    credit = 0.0
    for mission in info.available_missions:
        if mission.is_satisfied(before) or mission.is_satisfied(after):
            continue
        credit += weight * mission.value * _progress_fraction(mission, before, after)
    return credit


def typical_price(info: BotInfo) -> float:
    """Coin price the loan and invest plans use for one upcoming gem."""
    richest = max((opponent.coins for opponent in info.other_players), default=0)
    if richest <= 2:
        return 1.0
    return min(6.0, 1.0 + 0.25 * float(richest))


def plan_margin(
    budget: float,
    price: float,
    exposed_values: Sequence[float],
    future_value: float,
    future_slots: float,
) -> float:
    """Gem points minus coins spent, buying profitable gems in deal order.

    Exposed gems come first. A gem cheaper than ``price`` in value is skipped.
    Future slots are fractional so a one-third share of the next gem still
    counts. Unspent coins are worth the same in every plan, so they drop out.
    """
    if price <= 1e-9 or budget <= 0.0:
        return 0.0
    coins = budget
    surplus = 0.0
    for value in exposed_values:
        if coins < price:
            return surplus
        if value <= price:
            continue
        coins -= price
        surplus += value - price
    slots = future_slots
    while slots > 1e-9 and coins >= price and future_value > price:
        take = 1.0 if slots >= 1.0 else slots
        cost = price * take
        if cost > coins:
            take = coins / price
            cost = coins
        coins -= cost
        surplus += (future_value - price) * take
        slots -= take
        if take <= 1e-12:
            break
    return surplus


def lot_outlook(
    info: BotInfo, unit_values: Mapping[GemType, float]
) -> tuple[float, list[float], float, float]:
    """Price, exposed gem values, expected future gem value, and future slots.

    Future slots are the not-yet-exposed gems inside the horizon, divided by
    the table size, because one seat does not win every remaining auction.
    """
    price = typical_price(info)
    exposed_values = [unit_values.get(gem, 0.0) for gem in info.auction_gems]
    unknown_total = sum(info.unknown_gems.values())
    future_value = 0.0
    if unknown_total > 0:
        for gem in STANDARD_GEMS:
            share = gem_count(info.unknown_gems, gem) / unknown_total
            future_value += share * unit_values.get(gem, 0.0)
    remaining = max(0, info.gems_remaining - len(info.auction_gems))
    future_n = min(info.gem_deck_count, remaining, max(0, LOAN_HORIZON))
    seat_share = 1.0 / info.num_players if info.num_players else 1.0
    return price, exposed_values, future_value, future_n * seat_share


def loan_plan_surplus(
    info: BotInfo,
    principal: int,
    bid: int,
    outlook: tuple[float, list[float], float, float],
) -> float:
    """Net points from winning this loan at ``bid`` and following the plan.

    Repayment cancels the principal. What is left is the extra gem margin the
    new budget can buy, minus the bid, which is spent for good.
    """
    price, exposed_values, future_value, future_slots = outlook
    base = float(info.self_player.coins)
    boosted = base + float(principal) - float(bid)
    gain = plan_margin(
        boosted, price, exposed_values, future_value, future_slots
    ) - plan_margin(base, price, exposed_values, future_value, future_slots)
    return LOAN_CAPTURE * gain - float(bid)


def invest_plan_surplus(
    info: BotInfo,
    bonus: int,
    bid: int,
    outlook: tuple[float, list[float], float, float],
) -> float:
    """Bonus minus the treasure margin locked coins can no longer buy.

    The bid itself comes back at the end, so it is not subtracted again.
    """
    price, exposed_values, future_value, future_slots = outlook
    coins = float(info.self_player.coins)
    loss = plan_margin(
        coins, price, exposed_values, future_value, future_slots
    ) - plan_margin(coins - float(bid), price, exposed_values, future_value, future_slots)
    return float(bonus) - max(0.0, loss)


def apply_covering_bid(
    info: BotInfo,
    bid: int,
    *,
    lam: float,
    delta: float,
    denial_weight: float,
) -> int:
    """Raise a treasure bid to the minimum covering price when that price is cheap.

    The covering price is ``min_covering_bid``: each rival's coins, plus this
    loan's proceeds when the card is a loan, plus one when we do not win the
    tie. Winning moves this seat to last priority on the next auction; that
    next price is computed again from the new previous winner, not guessed here.
    The bid is used only when it is legal and the sure surplus is positive, so
    the price is below the liquidity-adjusted value of the lot.
    """
    if not USE_COVERING:
        return bid
    card = info.current_auction
    if card is None or card.type is not AuctionType.AUCTION:
        return bid
    limit = affordable(info)
    cover = min_covering_bid(info)
    if cover < 0 or cover > limit:
        return bid
    if bid > cover:
        # A higher bid cannot win more often. Covering already beats every stack.
        return cover
    lam_treasure, _lam_invest, hurdle = _stage_costs(info.gems_remaining, lam, delta)
    lot = lot_gems(info)
    unit_values = color_expectations(info, lot)
    gem_value = sum(unit_values.get(gem, 0.0) for gem in lot)
    gem_value = scarcity_adjust(info, lot, gem_value, unit_values)
    missions = float(mission_gain(info.self_player.won_gems, lot, info.available_missions))
    missions += mission_progress_value(info, lot)
    denial = denial_weight * float(rival_only_mission_value(info, lot))
    raw = gem_value + missions + denial
    shade = hurdle if cover > 0 else 0.0
    net = raw - shade - (1.0 + lam_treasure) * cover
    if cover < raw and net > 0.0 and bid < cover and cover <= COVER_MAX:
        return cover
    return bid


def search_bid(
    info: BotInfo,
    *,
    lam: float,
    delta: float,
    bid_fraction: float,
    bid_scale: float,
    laplace_weight: float,
    denial_weight: float,
    deadline: float | None = None,
    use_progress: bool = True,
) -> int:
    """Same search as ``select_bid``, with the optional value and plan hooks.

    With every feature switch off, the surplus of each bid matches
    ``select_bid``. The hooks only replace the terms they own.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return 0
    card = info.current_auction
    if card is None:
        return 0
    limit_budget = affordable(info)
    if limit_budget <= 0:
        return 0

    staged_treasure, staged_invest, hurdle = _stage_costs(
        info.gems_remaining, lam, delta
    )
    # Chart and invest multipliers do not touch each other. Opponent invest
    # targets keep the staged premium; only our own lock cost is discounted.
    lam_treasure = staged_treasure * _chart_treasure_mult(info)
    lam_invest = staged_invest
    lot = lot_gems(info)
    if card.type is AuctionType.LOAN:
        priced_colors: tuple[GemType, ...] = tuple(info.auction_gems)
    else:
        priced_colors = lot
    if USE_LOAN_PLAN or USE_INVEST_NET:
        priced_colors = tuple(dict.fromkeys((*priced_colors, *STANDARD_GEMS)))
    unit_values = color_expectations(info, priced_colors)
    gem_value = sum(unit_values.get(gem, 0.0) for gem in lot)
    our_gem = scarcity_adjust(info, lot, gem_value, unit_values)
    naive = 0.0
    if lot:
        chart = info.value_chart
        naive = sum(
            chart.get_value(gem_count(info.value_display, gem)) for gem in lot
        )
    missions = mission_gain(info.self_player.won_gems, lot, info.available_missions)
    progress_on = use_progress and USE_MISSION_PROGRESS
    our_missions = float(missions) + (
        mission_progress_value(info, lot) if progress_on else 0.0
    )
    rival_missions = rival_only_mission_value(info, lot)
    denial = denial_weight * rival_missions
    showing_value = 0.0
    if card.type is AuctionType.LOAN and info.auction_gems:
        showing_value = sum(
            unit_values.get(gem, 0.0) for gem in info.auction_gems
        ) / len(info.auction_gems)
    outlook = None
    if (USE_LOAN_PLAN and card.type is AuctionType.LOAN) or (
        USE_INVEST_NET and card.type is AuctionType.INVEST
    ):
        outlook = lot_outlook(info, unit_values)

    market_invest_lam = lam_invest
    if card.type is AuctionType.INVEST:
        lam_invest = _invest_lambda(info, lam_invest * INVEST_LAM_MULT)
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
    plain_bid = 0
    plain_utility = 0.0
    for bid in range(limit + 1):
        if deadline is not None and bid % 8 == 0 and time.perf_counter() >= deadline:
            break
        probability = curve[bid]
        if card.type is AuctionType.AUCTION:
            shade = hurdle if bid > 0 else 0.0
            cost = (1.0 + lam_treasure) * bid
            surplus = our_gem + our_missions + denial - shade - cost
            if progress_on:
                plain = our_gem + float(missions) + denial - shade - cost
                plain_utility_bid = probability * plain
                if plain_utility_bid > plain_utility + 1e-9:
                    plain_utility = plain_utility_bid
                    plain_bid = bid
        elif card.type is AuctionType.INVEST:
            if USE_INVEST_NET and outlook is not None:
                surplus = invest_plan_surplus(info, card.value, bid, outlook)
            else:
                surplus = card.value - lam_invest * bid
        elif USE_LOAN_PLAN and outlook is not None:
            surplus = loan_plan_surplus(info, card.value, bid, outlook)
        else:
            surplus = _loan_surplus(
                info,
                card.value,
                bid,
                lam_treasure,
                showing_value,
            )
        utility = probability * surplus
        if utility > best_utility + 1e-9:
            best_utility = utility
            best_bid = bid
    if (
        card.type is AuctionType.AUCTION
        and progress_on
        and best_bid > plain_bid + MISSION_BID_CAP
    ):
        best_bid = plain_bid + MISSION_BID_CAP
    if best_bid < 0:
        return 0
    if best_bid > limit:
        return limit
    return best_bid


class ValueLiquidityBot(Bot):
    """Analytic value-and-liquidity bidder. See the module docstring."""

    # Shadow price of one coin above the point it scores at the end.
    # Grid-searched against myopic and fraction bidders; 1.0 beat 0.25, 0.5, and 0.75.
    lam = 1.0
    # Shade subtracted from gem value on any positive bid. 1.0 won the same grid.
    delta = 1.0
    # Rival mean bid as a fraction of the public value signal.
    bid_fraction = 0.55
    # Laplace scale, in coins, of the rival bid distribution.
    bid_scale = 2.0
    # Weight on the Laplace; the rest is a uniform hedge.
    laplace_weight = 0.8
    # Fraction of a rival-only mission counted as denial value.
    denial_weight = 0.4
    # Stop optional work well inside the harness callback budget.
    time_budget_seconds = 0.25
    # Class defaults so a restored checkpoint can read the counters. Instance
    # copies are created only for ValueLiquidityBot.
    quiet_low = 0
    quiet_seen = 0

    def __init__(self) -> None:
        self.lam = type(self).lam
        self.delta = type(self).delta
        self.bid_fraction = type(self).bid_fraction
        self.bid_scale = type(self).bid_scale
        self.laplace_weight = type(self).laplace_weight
        self.denial_weight = type(self).denial_weight
        # Quiet counts are instance state only here, so subclass checkpoints stay
        # the dictionaries they had before this feature. Parameters stay on the
        # class: the host environment cannot change them.
        if type(self) is ValueLiquidityBot:
            self.quiet_low = 0
            self.quiet_seen = 0

    def on_game_start(self) -> None:
        try:
            if type(self) is ValueLiquidityBot:
                self.quiet_low = 0
                self.quiet_seen = 0
            return None
        except Exception:
            return None

    def _table_is_quiet(self) -> bool:
        """True once rivals have mostly bid one coin or less."""
        if not USE_QUIET_GATE:
            return False
        seen = getattr(self, "quiet_seen", 0)
        if seen < QUIET_MIN_OBS:
            return False
        return (self.quiet_low / seen) >= QUIET_RATE

    def get_bid(self, info: BotInfo) -> int:
        started = time.perf_counter()
        try:
            if info.current_auction is None:
                return 0
            # Subclasses (MyopicEvBot, and anyone else that does not replace
            # get_bid) stay on the untuned pricer. AdaptiveBot replaces get_bid.
            if type(self) is not ValueLiquidityBot:
                return select_bid(
                    info,
                    lam=self.lam,
                    delta=self.delta,
                    bid_fraction=self.bid_fraction,
                    bid_scale=self.bid_scale,
                    laplace_weight=self.laplace_weight,
                    denial_weight=self.denial_weight,
                    deadline=started + self.time_budget_seconds,
                )
            lam, delta = tuned_lambda(info, self.lam, self.delta)
            deadline = started + self.time_budget_seconds
            if _custom_features_enabled():
                bid = search_bid(
                    info,
                    lam=lam,
                    delta=delta,
                    bid_fraction=self.bid_fraction,
                    bid_scale=self.bid_scale,
                    laplace_weight=self.laplace_weight,
                    denial_weight=self.denial_weight,
                    deadline=deadline,
                    use_progress=not self._table_is_quiet(),
                )
            else:
                bid = select_bid(
                    info,
                    lam=lam,
                    delta=delta,
                    bid_fraction=self.bid_fraction,
                    bid_scale=self.bid_scale,
                    laplace_weight=self.laplace_weight,
                    denial_weight=self.denial_weight,
                    deadline=deadline,
                )
            bid = apply_covering_bid(
                info,
                bid,
                lam=lam,
                delta=delta,
                denial_weight=self.denial_weight,
            )
            limit = affordable(info)
            if bid < 0:
                return 0
            if bid > limit:
                return limit
            return int(bid)
        except Exception:
            return 0

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        del bids
        started = time.perf_counter()
        try:
            hidden = info.self_player.hidden_gems
            if not hidden:
                return 0
            if time.perf_counter() - started > self.time_budget_seconds:
                return 0
            if type(self) is not ValueLiquidityBot:
                return choose_reveal_index(info)
            if REVEAL_MODE == "first":
                return 0
            if REVEAL_MODE == "random":
                return random.randrange(len(hidden))
            return choose_reveal_index(info)
        except Exception:
            return 0

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        try:
            if type(self) is not ValueLiquidityBot or not USE_QUIET_GATE:
                return None
            me = info.self_player.id
            for player_id, bid in result.bids.items():
                if player_id == me:
                    continue
                self.quiet_seen += 1
                if bid <= QUIET_BID:
                    self.quiet_low += 1
            return None
        except Exception:
            return None

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        del info, result
        try:
            return None
        except Exception:
            return None
