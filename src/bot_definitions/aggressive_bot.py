"""AggressiveBot: mission races, cheap covers, and an exact late margin.

This is a different bidder from ValueLiquidityBot and AdaptiveBot. Those two
score a bid as own-score surplus times a liquidity premium. This one spends
to finish missions and to take lots that broke rivals would otherwise keep,
and it only borrows coins when a visible lot pays for the loan.

Verified harness rules this strategy relies on
----------------------------------------------
* Treasure (``AuctionType.AUCTION``) awards ``min(card.value, len(queue))``
  gems from the front of the exposed queue, then the winner pays the bid.
  ``harness._apply_auction``.
* A loan adds ``card.value`` coins and then subtracts the bid. At the end
  ``AuctionCard.end_game_value`` subtracts the principal, so unspent loan
  coins score zero. The bid itself is spent. ``lib.AuctionCard``,
  ``harness._score_player``.
* An investment locks the bid in ``invested_coins`` (returned at the end) and
  scores an extra ``card.value`` bonus. The bid is not a score cost. The cost
  is not being able to spend those coins before the end.
* Every auction winner may reveal one hidden starting gem, via
  ``choose_hidden_gem``. The engine does not pick the color. An empty hand
  reveals nothing and does not call the bot. ``player.reveal_gem``.
* ``finish_game`` moves every remaining hidden gem into the value display
  before scoring. The display is exactly the starting hands. Won gems never
  enter it. A reveal changes when opponents learn a color, not the final
  chart values. ``harness.finish_game``.
* The chart price of a color is ``ValueChart.get_value(display_count)``.
  Charts B and D fall as the count rises. Chart E peaks at three. Scoring
  multiplies won gems by that price. ``decks.VALUE_CHARTS``,
  ``pricing.color_expectations``.
* Four missions are public. After each auction, only the winner is checked.
  A completed mission is removed, so only one player claims it. Gems are not
  consumed. ``harness._claim_missions``.
* Legal bids are ``0 .. coins``, or ``0 .. coins + principal`` on a loan.
  Anything else is treated as 0. ``player.Player.get_bid``.
* The high bid wins. The opening tie is a uniform draw. Later, the seat after
  the previous winner wins the tie, and the winner is last in the next
  auction. ``harness._select_winner``, ``pricing.tie_win_probability``.
* Opponents' hidden colors and the deck order are not visible. ``unknown_gems``
  mixes the two. ``bot_info.build_bot_info``.

A reveal cannot complete a mission. Missions look at gems won at auction.
The reveal policy only moves the public display, which is what changes a
rival's price for colors still to be sold.

Feature switches
----------------
``USE_MISSIONS``, ``USE_LIQUIDITY``, ``USE_REVEAL``, and ``USE_ENDGAME``.
The submission bot copies them in ``__init__``. Ablation subclasses set the
matching class attribute instead, so several variants can play in one process.
Turn a switch off and that idea stops affecting bids and reveals.

Seeds 2000–2199, which were not used to pick these rules, put the all-on bot
behind ValueLiquidityBot at 3, 4, and 5 players. Missions and the liquidity
weapon account for most of that gap. The endgame margin is a few hundredths
of a point worse. The reveal is about half a point at 3 players and
indistinguishable from the first hidden gem at 4 and 5. All four default to
off. The functions stay, and the ablation classes turn them back on.

Contributors: CONTRIBUTORS
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence

from bot_definitions.pricing import (
    _bid_distribution,
    _opponent_target,
    _win_curve,
    affordable,
    color_expectations,
    gem_count,
    lot_gems,
    min_covering_bid,
    select_bid,
    tie_win_probability,
)
from bots import Bot, BotInfo, GameSummary, PlayerInfo, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import MissionCard, ValueChart

# Held-out seeds 2000–2199. Missions and liquidity lost points. Endgame and
# reveal did not earn a place. Ablation classes turn a single idea back on.
USE_MISSIONS = False
USE_LIQUIDITY = False
USE_REVEAL = False
USE_ENDGAME = False

# Last auctions scored as expected final margin. Includes the current lot.
ENDGAME_GEMS = 3
# Future completions are capped: the gems may never be dealt as a lot we win.
_FUTURE_CAP = 0.8
# Early shadow price of a coin. ValueLiquidityBot uses 1.0.
_EARLY_LAMBDA = 0.45
_MID_LAMBDA = 0.25
# Rival modeled loan bid when they are short of coins. Not their full budget.
_LOAN_FIGHT = 2
_SHORT_CACHE: dict[tuple[MissionCard, tuple[int, ...]], int] = {}
_CACHE_LIMIT = 4096


def _count_key(counts: Mapping[GemType, int]) -> tuple[int, ...]:
    return tuple(gem_count(counts, gem) for gem in STANDARD_GEMS)


def gems_short(mission: MissionCard, counts: Mapping[GemType, int]) -> int:
    """Minimum extra gems needed before ``mission`` is satisfied.

    Zero means the collection already qualifies. The search is bounded by the
    six copies of each color, and the result is cached.
    """
    start = _count_key(counts)
    key = (mission, start)
    cached = _SHORT_CACHE.get(key)
    if cached is not None:
        return cached
    if mission.is_satisfied(counts):
        _remember(key, 0)
        return 0
    wild = gem_count(mission.gem_requirements, GemType.WILD_CARD)
    if wild == 0:
        short = sum(
            max(0, gem_count(mission.gem_requirements, gem) - start[index])
            for index, gem in enumerate(STANDARD_GEMS)
        )
        _remember(key, short)
        return short
    found = _search_short(mission, start)
    _remember(key, found)
    return found


def _remember(key: tuple[MissionCard, tuple[int, ...]], short: int) -> None:
    if len(_SHORT_CACHE) >= _CACHE_LIMIT:
        _SHORT_CACHE.clear()
    _SHORT_CACHE[key] = short


def _search_short(mission: MissionCard, start: tuple[int, ...]) -> int:
    """Breadth-first search over adding one physical gem at a time."""
    seen = {start}
    queue: deque[tuple[tuple[int, ...], int]] = deque([(start, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= 6:
            continue
        for index in range(len(STANDARD_GEMS)):
            if current[index] >= 6:
                continue
            nxt = list(current)
            nxt[index] += 1
            state = tuple(nxt)
            if state in seen:
                continue
            seen.add(state)
            if mission.is_satisfied(dict(zip(STANDARD_GEMS, state))):
                return depth + 1
            queue.append((state, depth + 1))
    return 6


def _with_lot(
    counts: Mapping[GemType, int], lot: Sequence[GemType]
) -> dict[GemType, int]:
    updated = {gem: gem_count(counts, gem) for gem in STANDARD_GEMS}
    for gem in lot:
        updated[gem] = updated.get(gem, 0) + 1
    return updated


def market_supply(
    info: BotInfo, color: GemType, removed: Sequence[GemType]
) -> float:
    """Expected copies of ``color`` still auctionable after ``removed``.

    Exposed gems are certain. The deck's share of ``unknown_gems`` is the
    hypergeometric expectation. Opponent hidden gems are in ``unknown_gems``
    too, and they never return to the auction.
    """
    exposed = sum(gem == color for gem in info.auction_gems) - sum(
        gem == color for gem in removed
    )
    exposed = max(0, exposed)
    unknown_total = sum(info.unknown_gems.values())
    deck = info.gem_deck_count
    if unknown_total <= 0 or deck <= 0:
        return float(exposed)
    deck_share = gem_count(info.unknown_gems, color) * deck / unknown_total
    return exposed + deck_share


def _helpful_supply(
    info: BotInfo,
    mission: MissionCard,
    counts: Mapping[GemType, int],
    removed: Sequence[GemType],
) -> float:
    """How many future gems look useful for this mission."""
    supplies = [market_supply(info, gem, removed) for gem in STANDARD_GEMS]
    wild = gem_count(mission.gem_requirements, GemType.WILD_CARD)
    if wild == 0:
        total = 0.0
        for gem, supply in zip(STANDARD_GEMS, supplies):
            deficit = max(
                0, gem_count(mission.gem_requirements, gem) - gem_count(counts, gem)
            )
            total += min(float(deficit), supply)
        return total
    kind = mission.wild_card_requirement
    if kind is WildCardRequirement.SAME:
        return max(supplies) if supplies else 0.0
    if kind is WildCardRequirement.DIFFERENT:
        return float(sum(supply >= 0.4 for supply in supplies))
    return float(sum(supplies))


def _finish_prob(short: int, supply: float, auctions: int) -> float:
    """Chance of collecting ``short`` useful gems before the auctions run out.

    Capped below 1. A later mission is never treated as already scored.
    """
    if short <= 0:
        return 1.0
    if supply <= 0.15 or auctions <= 0:
        return 0.0
    # Needing two gems is harder than needing one, even when the supply
    # scales with the deficit. Raising to ``short`` is that extra risk.
    ratio = min(1.5, supply / short)
    window = min(1.0, auctions / short)
    one = min(_FUTURE_CAP, 0.7 * ratio * window)
    return one ** short


def _later_threat(
    mission: MissionCard,
    rivals: Sequence[PlayerInfo],
    info: BotInfo,
    removed: Sequence[GemType],
) -> float:
    """Chance some rival finishes this mission on a later auction."""
    threat = 0.0
    for rival in rivals:
        short = gems_short(mission, rival.won_gems)
        if short <= 0 or short > 3:
            continue
        supply = _helpful_supply(info, mission, rival.won_gems, removed)
        if supply < 0.4:
            continue
        if rival.coins >= short:
            afford = 0.7
        elif rival.coins >= 1:
            afford = 0.25
        else:
            afford = 0.05
        closeness = 0.5 if short == 1 else 0.3 if short == 2 else 0.15
        hit = closeness * afford * min(1.0, supply / short)
        threat = 1.0 - (1.0 - threat) * (1.0 - hit)
    return min(0.85, threat)


def _immediate_threat(
    mission: MissionCard,
    rivals: Sequence[PlayerInfo],
    lot: Sequence[GemType],
) -> float:
    """Chance a rival claims the mission by winning this lot."""
    threat = 0.0
    if not lot:
        return 0.0
    for rival in rivals:
        before = gems_short(mission, rival.won_gems)
        if before <= 0:
            continue
        after = gems_short(mission, _with_lot(rival.won_gems, lot))
        if after > 0:
            continue
        hit = 0.55 if rival.coins >= 1 else 0.2
        threat = max(threat, hit)
    return threat


def mission_increment(
    info: BotInfo,
    lot: Sequence[GemType],
    counts: Mapping[GemType, int],
    rivals: Sequence[PlayerInfo],
    *,
    lot_is_ours: bool,
) -> float:
    """Expected mission points this lot adds, each mission counted once.

    The value is ``bonus * (P(claim | take) - P(claim | pass))``. Passing does
    not get credit for the gems this auction is selling: those are gone after
    this round. A rival who is close, and who still has coins, lowers the
    value of waiting. Progress that does not complete the mission is only the
    change in that probability, never the full bonus.
    """
    if not lot or not info.available_missions:
        return 0.0
    after_counts = _with_lot(counts, lot)
    # The current lot is sold now. It is not a future source of gems.
    auctions_after = max(0, info.gems_remaining - len(lot))
    total = 0.0
    for mission in info.available_missions:
        short_before = gems_short(mission, counts)
        if short_before <= 0:
            continue
        short_after = gems_short(mission, after_counts)
        # Passing leaves this lot on the table, so it is not a future source.
        # The collection used for that chance is the one that did not get it.
        future_with = _helpful_supply(info, mission, after_counts, lot)
        future_without = _helpful_supply(info, mission, counts, lot)
        later = _later_threat(mission, rivals, info, lot)
        if lot_is_ours and short_after <= 0:
            p_with = 1.0
        else:
            p_with = _finish_prob(short_after, future_with, auctions_after) * (
                1.0 - later
            )
        # If we pass, a rival may take the completing lot on this auction.
        steal = later
        if not lot_is_ours:
            now = _immediate_threat(mission, rivals, lot)
            steal = 1.0 - (1.0 - later) * (1.0 - now)
        p_without = _finish_prob(short_before, future_without, auctions_after) * (
            1.0 - steal
        )
        total += mission.value * max(0.0, p_with - p_without)
    return total


def rival_denial(info: BotInfo, lot: Sequence[GemType]) -> float:
    """Largest expected mission gain a rival gets from winning this lot.

    This is the same increment, from their collection. It is small when they
    could finish the mission on a later auction anyway.
    """
    if not lot:
        return 0.0
    me = info.self_player
    best = 0.0
    for rival in info.other_players:
        others: list[PlayerInfo] = [
            player for player in info.other_players if player.id != rival.id
        ]
        others.append(me)
        gained = mission_increment(
            info, lot, rival.won_gems, others, lot_is_ours=True
        )
        if gained > best:
            best = gained
    return best


def cover_after_we_win(info: BotInfo) -> int:
    """Coins needed next round if we win this one.

    The winner is last in the next tiebreak, so a rival's bid of their full
    stack beats a match. Broke rivals are beaten by 1.
    """
    need = 0
    for opponent in info.other_players:
        need = max(need, max(0, opponent.coins) + 1)
    return need


def loan_contest_bid(info: BotInfo) -> int:
    """Bid that beats cautious loan bids, using the current tiebreak.

    A rival is not modeled as spending the whole loan. Short stacks fight for
    a couple of coins. A stack of eight or more is treated as uninterested,
    and a zero bid still loses to them when we lose the tie.
    """
    me = info.self_player.id
    previous = info.previous_winner_id
    players = info.num_players
    need = 0
    for opponent in info.other_players:
        if opponent.coins >= 8:
            modeled = 0
        else:
            modeled = min(max(0, opponent.coins), _LOAN_FIGHT)
        if tie_win_probability(me, opponent.id, previous, players) >= 1.0:
            need = max(need, modeled)
        else:
            need = max(need, modeled + 1)
    return need


def enabled_surplus(
    info: BotInfo,
    spendable_with: int,
    spendable_without: int,
    *,
    missions: bool,
) -> float:
    """Points from visible lots the loan can buy and our own coins cannot.

    Each exposed gem is priced at the cover we will face after winning the
    loan. Gain is gem value plus mission increment, minus the coins spent,
    because leftover coins score one point and the principal is repaid.
    """
    exposed = tuple(info.auction_gems[:2])
    if not exposed or info.gems_remaining <= 0:
        return 0.0
    price = cover_after_we_win(info)
    if price <= 0:
        return 0.0
    values = color_expectations(info, exposed)
    surplus = 0.0
    left_with = spendable_with
    left_without = spendable_without
    owned = info.self_player.won_gems
    taken: list[GemType] = []
    for gem in exposed:
        lot = tuple(taken + [gem])
        unit = values.get(gem, 0.0)
        extra = 0.0
        if missions:
            gained = mission_increment(
                info, lot, owned, info.other_players, lot_is_ours=True
            )
            if taken:
                earlier = mission_increment(
                    info, tuple(taken), owned, info.other_players, lot_is_ours=True
                )
                extra = max(0.0, gained - earlier)
            else:
                extra = gained
        gain = unit + extra - price
        can_with = left_with >= price
        can_without = left_without >= price
        if gain > 0 and can_with and not can_without:
            surplus += gain
            left_with -= price
        elif gain > 0 and can_with and can_without:
            left_with -= price
            left_without -= price
        taken.append(gem)
    return surplus


def loan_decision(info: BotInfo, *, missions: bool) -> tuple[int, float]:
    """Return ``(bid, net)``. A positive bid is returned only when ``net > 0``.

    ``net`` is the surplus of lots the extra coins unlock, minus the bid.
    The principal is not profit. Bidding zero is allowed when the loan is
    free; that bid does not spend coins if it loses.
    """
    card = info.current_auction
    if card is None or card.type is not AuctionType.LOAN:
        return 0, 0.0
    coins = info.self_player.coins
    principal = card.value
    budget = coins + principal
    contest = min(budget, loan_contest_bid(info))
    # Bid 0 secures the loan only when we already beat every modeled rival,
    # including the tiebreak. Otherwise the contest bid is the plan, and it
    # is kept only when the lots it unlocks are worth more than that bid.
    if contest <= 0:
        net = enabled_surplus(
            info, coins + principal, coins, missions=missions
        )
        if net > 0.0:
            return 0, net
        return 0, 0.0
    spendable = coins - contest + principal
    net = enabled_surplus(info, spendable, coins, missions=missions) - float(
        contest
    )
    if net > 0.0:
        return contest, net
    return 0, 0.0


def _lambda_for(gems_remaining: int, liquidity: bool) -> float:
    if not liquidity:
        return 1.0
    if gems_remaining <= ENDGAME_GEMS:
        return 0.0
    if gems_remaining <= 8:
        return _MID_LAMBDA
    return _EARLY_LAMBDA


def _broke(info: BotInfo) -> bool:
    return all(opponent.coins <= 2 for opponent in info.other_players)


def reveal_index(info: BotInfo, bids: Mapping[int, int]) -> int:
    """Index of the hidden gem whose reveal shifts rival prices our way.

    The final display is fixed, so this only changes the public count rivals
    can see on later auctions. A closed color cannot change a later lot.
    A positive chart slope makes that color look better to anyone pricing off
    the display; a negative slope (scarcity, or chart E past the peak) makes
    it look worse. Bids tell us whether anyone is still spending.
    """
    hidden = info.self_player.hidden_gems
    if not hidden:
        return 0
    me = info.self_player.id
    contested = any(
        amount > 0 for player_id, amount in bids.items() if player_id != me
    )
    chart = info.value_chart
    held: dict[GemType, int] = {}
    for gem in hidden:
        held[gem] = held.get(gem, 0) + 1
    best_index = 0
    best_score = -1e300
    for index, gem in enumerate(hidden):
        score = _reveal_score(info, gem, chart, held, contested)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _reveal_score(
    info: BotInfo,
    gem: GemType,
    chart: ValueChart,
    held: Mapping[GemType, int],
    contested: bool,
) -> float:
    display = gem_count(info.value_display, gem)
    # The public chart is what rivals recompute after they see this gem.
    slope = float(
        chart.get_value(min(6, display + 1)) - chart.get_value(display)
    )
    future = market_supply(info, gem, ())
    open_market = future > 0.35
    want = _want_color(info, gem)
    score = 0.0
    if not open_market:
        # Nothing left to auction in this color. Revealing it cannot raise a
        # later lot of the same color.
        score += 4.0
    exposed = float(sum(item == gem for item in info.auction_gems))
    weight = exposed + 0.5 * max(0.0, future - exposed)
    if open_market and slope > 0.0 and want:
        score -= 2.0 * slope * max(0.3, weight)
    if open_market and slope < 0.0 and want:
        # Scarcity charts: showing another copy makes the color look cheaper.
        score += 2.0 * (-slope) * max(0.3, weight)
    if open_market and slope > 0.0 and not want:
        rich = sum(opponent.coins >= 3 for opponent in info.other_players)
        drain = 1.2 * slope * float(rich)
        if contested:
            drain += 0.4 * slope
        score += drain
    if slope > 0.0 and held.get(gem, 0) >= 2 and want:
        # Keep a color we still want while the table is short of the truth.
        score -= 0.8 * float(held[gem])
    if slope > 0.0 and _rival_needs(info, gem):
        # They need the color for a public mission. A higher display price
        # spends their coins. It does not change their collection.
        score += 0.6 * slope
    return score


def _want_color(info: BotInfo, gem: GemType) -> bool:
    gained = mission_increment(
        info,
        (gem,),
        info.self_player.won_gems,
        info.other_players,
        lot_is_ours=True,
    )
    if gained >= 1.5:
        return True
    value = color_expectations(info, (gem,)).get(gem, 0.0)
    return value >= 8.0 and info.self_player.coins >= 2


def _rival_needs(info: BotInfo, gem: GemType) -> bool:
    for rival in info.other_players:
        for mission in info.available_missions:
            needed = gem_count(mission.gem_requirements, gem)
            owned = gem_count(rival.won_gems, gem)
            if needed <= owned:
                continue
            if gems_short(mission, rival.won_gems) <= 2:
                return True
    return False


def _targets(
    info: BotInfo,
    naive: float,
    gem_value: float,
    rival_missions: int,
    bid_fraction: float,
) -> list[tuple[int, float, int]]:
    card = info.current_auction
    if card is None:
        return []
    built: list[tuple[int, float, int]] = []
    for opponent in info.other_players:
        cap, mean = _opponent_target(
            opponent,
            card,
            naive,
            gem_value,
            rival_missions,
            bid_fraction,
            0.5,
        )
        # No rival can bid above coins, plus this loan's proceeds.
        loan = card.value if card.type is AuctionType.LOAN else 0
        cap = min(cap, max(0, opponent.coins + loan))
        mean = min(mean, float(cap))
        built.append((cap, mean, opponent.id))
    return built


def _naive(info: BotInfo, lot: Sequence[GemType]) -> float:
    chart = info.value_chart
    return float(
        sum(chart.get_value(gem_count(info.value_display, gem)) for gem in lot)
    )


def _zero_curve(
    info: BotInfo,
    targets: Sequence[tuple[int, float, int]],
    limit: int,
    zero_mass: float,
) -> list[float]:
    """P(win) with an extra point mass at bid 0.

    Used only in the endgame, so a rival who saves coins is not treated as
    bidding their maximum.
    """
    distributions: list[list[float]] = []
    for cap, mean, _opponent_id in targets:
        dist = _bid_distribution(cap, mean, 2.5, 0.7)
        if zero_mass > 0.0 and dist:
            kept = 1.0 - zero_mass
            dist = [prob * kept for prob in dist]
            dist[0] += zero_mass
        distributions.append(dist)
    cdfs: list[list[float]] = []
    for dist in distributions:
        running = 0.0
        cdf: list[float] = []
        for prob in dist:
            running += prob
            cdf.append(running)
        cdfs.append(cdf)
    me = info.self_player.id
    previous = info.previous_winner_id
    players = info.num_players
    curve: list[float] = []
    for bid in range(limit + 1):
        probability = 1.0
        for dist, cdf, target in zip(distributions, cdfs, targets):
            if bid >= len(dist):
                continue
            below = cdf[bid - 1] if bid else 0.0
            tie = dist[bid] * tie_win_probability(
                me, target[2], previous, players
            )
            probability *= below + tie
            if probability <= 0.0:
                break
        curve.append(probability)
    return curve


def _expected_rival_pay(targets: Sequence[tuple[int, float, int]]) -> float:
    if not targets:
        return 0.0
    return max(mean for _cap, mean, _id in targets)


def _next_gem_value(info: BotInfo, lot: Sequence[GemType]) -> float:
    """Rough value of one gem still exposed after this lot is taken."""
    queue = list(info.auction_gems)
    for gem in lot:
        if gem in queue:
            queue.remove(gem)
    if not queue:
        return 0.0
    values = color_expectations(info, queue[:1])
    return values.get(queue[0], 0.0)


class AggressiveBot(Bot):
    """Mission racer. See the module docstring for the four switches."""

    use_missions = USE_MISSIONS
    use_liquidity = USE_LIQUIDITY
    use_reveal = USE_REVEAL
    use_endgame = USE_ENDGAME
    # Buy a loan when the borrowed coins are worth more than the bid.
    # The liquidity switch stays off: that one also capped treasure bids.
    use_valued_loans = True
    time_budget_seconds = 0.25

    def __init__(self) -> None:
        # The submission class follows the module switches. Ablation
        # subclasses carry their own flags so one process can compare them.
        if type(self) is AggressiveBot:
            self.use_missions = USE_MISSIONS
            self.use_liquidity = USE_LIQUIDITY
            self.use_reveal = USE_REVEAL
            self.use_endgame = USE_ENDGAME
        else:
            self.use_missions = type(self).use_missions
            self.use_liquidity = type(self).use_liquidity
            self.use_reveal = type(self).use_reveal
            self.use_endgame = type(self).use_endgame
        self.fallback_bids = 0
        self.fallback_reveals = 0
        self.clamped_bids = 0

    def on_game_start(self) -> None:
        try:
            return None
        except Exception:
            return None

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        del info, result
        try:
            return None
        except Exception:
            return None

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        del info, result
        try:
            return None
        except Exception:
            return None

    def get_bid(self, info: BotInfo) -> int:
        started = time.perf_counter()
        try:
            card = info.current_auction
            if card is None:
                return 0
            budget = affordable(info)
            if budget <= 0:
                return 0
            if (
                self.use_endgame
                and info.gems_remaining <= ENDGAME_GEMS
                and card.type is not AuctionType.LOAN
            ):
                bid = self._endgame_bid(info, budget, started)
            elif card.type is AuctionType.LOAN:
                bid = self._loan_bid(info)
            elif card.type is AuctionType.INVEST:
                bid = self._invest_bid(info, budget, started)
            else:
                bid = self._treasure_bid(info, budget, started)
            if type(bid) is not int:
                self.fallback_bids += 1
                return 0
            if bid < 0 or bid > budget:
                self.clamped_bids += 1
                return min(budget, max(0, bid))
            return bid
        except Exception:
            self.fallback_bids += 1
            return 0

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        try:
            hidden = info.self_player.hidden_gems
            if not hidden:
                return 0
            if not self.use_reveal:
                return 0
            index = reveal_index(info, bids)
            if type(index) is not int or not 0 <= index < len(hidden):
                self.fallback_reveals += 1
                return 0
            return index
        except Exception:
            self.fallback_reveals += 1
            return 0

    def _my_mission(self, info: BotInfo, lot: Sequence[GemType]) -> float:
        if not self.use_missions or not lot:
            return 0.0
        return mission_increment(
            info,
            lot,
            info.self_player.won_gems,
            info.other_players,
            lot_is_ours=True,
        )

    def _denial(self, info: BotInfo, lot: Sequence[GemType]) -> float:
        if not self.use_missions or not lot:
            return 0.0
        return rival_denial(info, lot)

    def _prize(
        self, info: BotInfo, lot: Sequence[GemType], gem_value: float
    ) -> tuple[float, float]:
        """Own-score prize, and the rival mission swing used as denial.

        Denial is added in full only when the covering bid is small next to
        the points it blocks. Otherwise a fifth of it is kept, because the
        rival may still finish later and chasing is expensive.
        """
        mine = gem_value + self._my_mission(info, lot)
        denial = self._denial(info, lot)
        if not self.use_missions:
            return mine, 0.0
        cover = min_covering_bid(info)
        if cover <= max(2, 0.25 * max(denial, 1.0)):
            return mine + denial, denial
        return mine + 0.2 * denial, denial

    def _treasure_bid(self, info: BotInfo, budget: int, started: float) -> int:
        lot = lot_gems(info)
        if not lot:
            return 0
        values = color_expectations(info, lot)
        gem_value = sum(values.get(gem, 0.0) for gem in lot)
        prize, denial = self._prize(info, lot, gem_value)
        if self.use_liquidity and _broke(info) and gem_value > 0.0:
            # They would bank this lot for a coin or two. Taking it also
            # keeps that value off their score.
            prize += 0.35 * gem_value
        naive = _naive(info, lot)
        rival_signal = int(round(denial))
        targets = _targets(info, naive, gem_value, rival_signal, 0.5)
        cover = min_covering_bid(info)
        if self.use_liquidity:
            limit = min(budget, cover)
        else:
            limit = budget
        curve = _win_curve(info, targets, limit, 2.0, 0.75)
        lam = _lambda_for(info.gems_remaining, self.use_liquidity)
        delta = 0.25 if self.use_liquidity else 1.0
        return _maximize(
            curve, limit, prize, lam, delta, started, self.time_budget_seconds
        )

    def _invest_bid(self, info: BotInfo, budget: int, started: float) -> int:
        card = info.current_auction
        if card is None:
            return 0
        bonus = float(card.value)
        reserve = self._invest_reserve(info, bonus)
        limit = max(0, min(budget, info.self_player.coins - reserve))
        if self.use_liquidity:
            # Stay just under the printed bonus. The bid comes back; the
            # bonus is the only profit, and locking more than that is a
            # different bet from buying the investment.
            limit = min(limit, max(0, card.value - 1))
            lam = 0.2
        else:
            lam = 1.0
        if limit <= 0:
            return 0
        targets = _targets(info, bonus, bonus, 0, 0.5)
        curve = _win_curve(info, targets, limit, 2.0, 0.75)
        return _maximize(
            curve,
            limit,
            bonus,
            lam,
            0.0,
            started,
            self.time_budget_seconds,
            invest=True,
        )

    def _invest_reserve(self, info: BotInfo, bonus: float) -> int:
        if info.gems_remaining <= 0 or info.self_player.coins <= 1:
            return 0
        if not self.use_liquidity or not info.auction_gems:
            return 1
        gem = info.auction_gems[0]
        value = color_expectations(info, (gem,)).get(gem, 0.0)
        if value <= bonus:
            return 1
        return min(info.self_player.coins, cover_after_we_win(info))

    def _loan_bid(self, info: BotInfo) -> int:
        if self.use_liquidity:
            bid, _net = loan_decision(info, missions=self.use_missions)
            return bid
        if not self.use_valued_loans:
            return 0
        # Same loan price ValueLiquidityBot pays. Treasure bids stay on the
        # aggressive curve; only the loan decision is shared.
        return select_bid(
            info,
            lam=1.0,
            delta=1.0,
            bid_fraction=0.55,
            bid_scale=2.0,
            laplace_weight=0.8,
            denial_weight=0.4,
        )

    def _endgame_bid(self, info: BotInfo, budget: int, started: float) -> int:
        """Maximize expected final margin over the strongest rival.

        Cash left is one point. A rival's maximum bid is their stack (plus
        loan proceeds, which this path does not spend). Their expected bid is
        a shrunk value bid with mass at zero, not that maximum.
        """
        card = info.current_auction
        if card is None:
            return 0
        if card.type is AuctionType.INVEST:
            return self._endgame_invest(info, budget, started)
        lot = lot_gems(info)
        values = color_expectations(info, lot)
        gem_value = sum(values.get(gem, 0.0) for gem in lot)
        my_gain = gem_value + self._my_mission(info, lot)
        their_gain = gem_value + self._denial(info, lot)
        naive = _naive(info, lot) if lot else 0.0
        targets = _targets(info, naive, gem_value, int(round(their_gain)), 0.35)
        cover = min_covering_bid(info)
        limit = min(budget, cover) if self.use_liquidity else budget
        curve = _zero_curve(info, targets, limit, 0.15)
        their_pay = _expected_rival_pay(targets)
        next_value = _next_gem_value(info, lot)
        future_price = cover_after_we_win(info)
        # Half the next card is a treasure; the rest are loans or investments.
        future_weight = 0.5 if info.gems_remaining > len(lot) else 0.0
        best_bid = 0
        best_margin = -1e300
        for bid in range(limit + 1):
            if (
                bid % 8 == 0
                and time.perf_counter() - started > self.time_budget_seconds
            ):
                break
            probability = curve[bid] if bid < len(curve) else 1.0
            # Exclusive gem and mission: we score them or the rival does.
            win_term = my_gain - float(bid)
            lose_term = max(0.0, their_gain - their_pay)
            margin = probability * win_term - (1.0 - probability) * lose_term
            if future_weight > 0.0 and next_value > future_price:
                coins_if_win = info.self_player.coins - bid
                p_future = 1.0 if coins_if_win >= future_price else 0.0
                margin += probability * future_weight * p_future * (
                    next_value - future_price
                )
            if margin > best_margin + 1e-9:
                best_margin = margin
                best_bid = bid
        return best_bid

    def _endgame_invest(self, info: BotInfo, budget: int, started: float) -> int:
        card = info.current_auction
        if card is None:
            return 0
        bonus = float(card.value)
        reserve = self._invest_reserve(info, bonus)
        limit = max(0, min(budget, info.self_player.coins - reserve))
        if self.use_liquidity:
            limit = min(limit, max(0, card.value - 1))
        if limit <= 0:
            return 0
        targets = _targets(info, bonus, bonus, 0, 0.35)
        curve = _zero_curve(info, targets, limit, 0.15)
        # The bid is returned. The swing versus a rival who takes it is two
        # bonuses, minus any treasure the locked coins can no longer buy.
        next_value = _next_gem_value(info, ())
        future_price = cover_after_we_win(info)
        best_bid = 0
        best_margin = -1e300
        for bid in range(limit + 1):
            if (
                bid % 8 == 0
                and time.perf_counter() - started > self.time_budget_seconds
            ):
                break
            probability = curve[bid] if bid < len(curve) else 1.0
            locked_out = 0.0
            if (
                next_value > future_price
                and info.self_player.coins - bid < future_price
                <= info.self_player.coins
            ):
                locked_out = next_value - future_price
            margin = probability * (bonus - locked_out) - (1.0 - probability) * bonus
            if margin > best_margin + 1e-9:
                best_margin = margin
                best_bid = bid
        return best_bid


def _maximize(
    curve: Sequence[float],
    limit: int,
    prize: float,
    lam: float,
    delta: float,
    started: float,
    budget_seconds: float,
    *,
    invest: bool = False,
) -> int:
    """Pick the bid with the best expected surplus. Never bids above ``limit``."""
    best_bid = 0
    best_utility = -1e300
    for bid in range(limit + 1):
        if bid % 8 == 0 and time.perf_counter() - started > budget_seconds:
            break
        probability = curve[bid] if bid < len(curve) else (curve[-1] if curve else 0.0)
        if invest:
            surplus = prize - lam * bid
        else:
            shade = delta if bid > 0 else 0.0
            surplus = prize - shade - (1.0 + lam) * bid
        utility = probability * surplus
        if utility > best_utility + 1e-9:
            best_utility = utility
            best_bid = bid
    if best_utility < 0.0:
        return 0
    return best_bid


class AggressiveNoMissions(AggressiveBot):
    """Ablation: mission progress and mission denial are off."""

    use_missions = False
    use_liquidity = True
    use_reveal = True
    use_endgame = True


class AggressiveNoLiquidity(AggressiveBot):
    """Ablation: no cheap covers, no planned loans, ValueLiquidity lambda."""

    use_missions = True
    use_liquidity = False
    use_reveal = True
    use_endgame = True


class AggressiveNoReveal(AggressiveBot):
    """Ablation: reveal the first hidden gem."""

    use_missions = True
    use_liquidity = True
    use_reveal = False
    use_endgame = True


class AggressiveNoEndgame(AggressiveBot):
    """Ablation: the last auctions use the same utility as the rest."""

    use_missions = True
    use_liquidity = True
    use_reveal = True
    use_endgame = False
