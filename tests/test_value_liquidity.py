"""Pricing, legality, tiebreaks, missions, and callback timing."""

import pickle
import time
import unittest
from collections import Counter
from random import Random
from unittest.mock import patch

from bot_definitions.baselines import FractionValueBot
from bot_definitions.pricing import (
    choose_reveal_index,
    color_expectations,
    final_count_distribution,
    min_covering_bid,
    mission_gain,
    rival_only_mission_value,
    select_bid,
)
from bot_definitions.value_liquidity_bot import ValueLiquidityBot
from bot_info import build_bot_info
from bots import Bot, BotInfo, PlayerInfo, RoundResult, SelfInfo
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from decks import VALUE_CHARTS
from harness import create_game, play_round
from immutable import FrozenMap
from lib import AuctionCard, MissionCard
from player import Player

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS


def _counts(overrides: dict[GemType, int] | None = None) -> FrozenMap[GemType, int]:
    values = {gem: 0 for gem in STANDARD_GEMS}
    if overrides:
        values.update(overrides)
    return FrozenMap(values)


def _player(
    player_id: int,
    coins: int,
    *,
    hidden_count: int = 0,
    won: FrozenMap[GemType, int] | None = None,
) -> PlayerInfo:
    return PlayerInfo(
        id=player_id,
        coins=coins,
        hidden_gem_count=hidden_count,
        revealed_gems=(),
        won_gems=won if won is not None else _counts(),
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )


def make_info(
    *,
    coins: int = 35,
    hidden: tuple[GemType, ...] = (),
    others: tuple[PlayerInfo, ...] | None = None,
    auction: AuctionCard | None = None,
    auction_gems: tuple[GemType, ...] = (BLUE,),
    display: dict[GemType, int] | None = None,
    unknown: dict[GemType, int] | None = None,
    chart_index: int = 0,
    previous: int | None = 2,
    missions: tuple[MissionCard, ...] = (),
    gems_remaining: int = 15,
    self_id: int = 0,
    won: FrozenMap[GemType, int] | None = None,
) -> BotInfo:
    opponents = others if others is not None else (
        _player(1, 35),
        _player(2, 35),
    )
    return BotInfo(
        self_player=SelfInfo(
            id=self_id,
            coins=coins,
            hidden_gem_count=len(hidden),
            revealed_gems=(),
            won_gems=won if won is not None else _counts(),
            auction_cards=(),
            mission_cards=(),
            invested_coins=0,
            hidden_gems=hidden,
        ),
        other_players=opponents,
        num_players=1 + len(opponents),
        round_number=3,
        current_auction=(
            auction
            if auction is not None
            else AuctionCard(AuctionType.AUCTION, 1)
        ),
        gem_deck_count=8,
        gems_remaining=gems_remaining,
        unknown_gems=_counts(unknown),
        auction_gems=auction_gems,
        value_display=_counts(display),
        previous_winner_id=previous,
        value_chart=VALUE_CHARTS[chart_index],
        available_missions=missions,
        auction_deck_count=10,
        auction_cards_remaining=FrozenMap(),
    )


def _bid(info: BotInfo, **overrides: float) -> int:
    params = {
        "lam": 0.5,
        "delta": 2.0,
        "bid_fraction": 0.55,
        "bid_scale": 2.0,
        "laplace_weight": 0.8,
        "denial_weight": 0.4,
    }
    params.update(overrides)
    return select_bid(info, **params)  # type: ignore[arg-type]


class PricingTests(unittest.TestCase):
    def test_hypergeometric_final_counts_when_two_cards_are_unknown(self):
        # 2 blue and 2 other unknown cards; opponents hold 2 of them.
        # P(0)=1/6, P(1)=4/6, P(2)=1/6, and our hand adds nothing.
        info = make_info(
            hidden=(),
            others=(_player(1, 10, hidden_count=2),),
            unknown={BLUE: 2, GREEN: 2},
            auction_gems=(),
        )
        dist = final_count_distribution(info, BLUE)
        self.assertAlmostEqual(dist[0], 1 / 6)
        self.assertAlmostEqual(dist[1], 4 / 6)
        self.assertAlmostEqual(dist[2], 1 / 6)
        self.assertAlmostEqual(sum(dist), 1.0)

    def test_known_hand_fixes_chart_value_including_scarcity_and_chart_e(self):
        # Two hidden blue gems, no opponent cards left: final display is 2.
        abundant = make_info(hidden=(BLUE, BLUE), others=(_player(1, 0), _player(2, 0)))
        self.assertAlmostEqual(color_expectations(abundant, (BLUE,))[BLUE], 8.0)
        scarce = make_info(
            chart_index=1,
            hidden=(),
            others=(_player(1, 0),),
            display={BLUE: 0},
        )
        self.assertAlmostEqual(color_expectations(scarce, (BLUE,))[BLUE], 20.0)
        peaked = make_info(chart_index=4, display={BLUE: 3}, others=(_player(1, 0),))
        crashed = make_info(chart_index=4, display={BLUE: 4}, others=(_player(1, 0),))
        self.assertAlmostEqual(color_expectations(peaked, (BLUE,))[BLUE], 18.0)
        self.assertAlmostEqual(color_expectations(crashed, (BLUE,))[BLUE], 6.0)

    def test_mission_bonus_detects_new_completions_only(self):
        pair = MissionCard(5, {GemType.WILD_CARD: 2}, WildCardRequirement.SAME)
        triple = MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME)
        exact = MissionCard(5, {BLUE: 1, GREEN: 1})
        empty = _counts()
        self.assertEqual(mission_gain(empty, (BLUE, BLUE), (pair, triple)), 5)
        self.assertEqual(mission_gain(empty, (BLUE, GREEN), (exact,)), 5)
        self.assertEqual(mission_gain(_counts({BLUE: 1}), (BLUE,), (pair,)), 5)
        self.assertEqual(mission_gain(_counts({BLUE: 2}), (BLUE,), (pair,)), 0)

    def test_rival_mission_is_ignored_when_we_complete_it_ourselves(self):
        triple = MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME)
        rival = make_info(
            auction_gems=(BLUE,),
            won=_counts(),
            others=(_player(1, 20, won=_counts({BLUE: 2})),),
            missions=(triple,),
            hidden=(),
        )
        self.assertEqual(rival_only_mission_value(rival, (BLUE,)), 10)
        both = make_info(
            auction_gems=(BLUE,),
            won=_counts({BLUE: 2}),
            others=(_player(1, 20, won=_counts({BLUE: 2})),),
            missions=(triple,),
        )
        self.assertEqual(rival_only_mission_value(both, (BLUE,)), 0)

    def test_covering_bid_respects_tie_priority_and_opening_randomness(self):
        # Previous winner is seat 1, so seat 2 wins ties and seat 1 loses them.
        favored = make_info(
            self_id=2,
            previous=1,
            coins=35,
            others=(_player(0, 5), _player(1, 4)),
        )
        trailer = make_info(
            self_id=1,
            previous=1,
            coins=35,
            others=(_player(0, 5), _player(2, 4)),
        )
        opening = make_info(
            self_id=0,
            previous=None,
            others=(_player(1, 5), _player(2, 4)),
        )
        loan = make_info(
            self_id=0,
            previous=None,
            auction=AuctionCard(AuctionType.LOAN, 20),
            others=(_player(1, 5), _player(2, 0)),
        )
        self.assertEqual(min_covering_bid(favored), 5)
        self.assertEqual(min_covering_bid(trailer), 6)
        self.assertEqual(min_covering_bid(opening), 6)
        self.assertEqual(min_covering_bid(loan), 26)

    def test_bid_never_exceeds_cash_and_shades_a_modest_lot(self):
        modest = make_info(
            coins=35,
            hidden=(BLUE, BLUE),
            others=(_player(1, 35, hidden_count=0), _player(2, 35, hidden_count=0)),
            auction_gems=(BLUE,),
            gems_remaining=15,
        )
        bid = _bid(modest)
        self.assertIs(type(bid), int)
        self.assertGreaterEqual(bid, 0)
        # Value is 8. With delta 2 and lambda 0.5, bids above 3 have no surplus.
        self.assertLessEqual(bid, 3)

    def test_pays_up_to_the_budget_for_a_clearly_underpriced_lot(self):
        jackpot = make_info(
            chart_index=1,
            coins=10,
            hidden=(),
            auction=AuctionCard(AuctionType.AUCTION, 2),
            auction_gems=(BLUE, GREEN),
            others=(_player(1, 35), _player(2, 35)),
            gems_remaining=15,
        )
        self.assertEqual(_bid(jackpot), 10)
        self.assertEqual(ValueLiquidityBot().get_bid(jackpot), 10)

    def test_free_wins_when_we_have_tiebreak_and_rivals_are_broke(self):
        broke = (
            _player(1, 0),
            _player(2, 0),
        )
        treasure = make_info(
            coins=20,
            hidden=(BLUE, BLUE),
            others=broke,
            previous=2,
            self_id=0,
            gems_remaining=12,
        )
        self.assertEqual(_bid(treasure), 0)
        must_step = make_info(
            coins=20,
            hidden=(BLUE, BLUE),
            others=broke,
            previous=2,
            self_id=2,
            gems_remaining=12,
        )
        self.assertEqual(_bid(must_step), 1)
        worthless = make_info(
            coins=20,
            hidden=(),
            others=broke,
            previous=2,
            self_id=2,
            auction_gems=(BLUE,),
            gems_remaining=12,
        )
        self.assertEqual(_bid(worthless), 0)

    def test_invest_and_loan_are_taken_for_free_and_stay_legal(self):
        broke = (_player(1, 0), _player(2, 0))
        invest = make_info(
            coins=18,
            auction=AuctionCard(AuctionType.INVEST, 10),
            others=broke,
            previous=2,
            self_id=0,
            gems_remaining=10,
        )
        loan = make_info(
            coins=18,
            auction=AuctionCard(AuctionType.LOAN, 20),
            others=broke,
            previous=2,
            self_id=0,
            gems_remaining=10,
        )
        self.assertEqual(_bid(invest), 0)
        # Rivals can spend the loan itself, so a zero-coin rival is not a free win.
        loan_price = _bid(loan)
        self.assertGreaterEqual(loan_price, 1)
        self.assertLessEqual(loan_price, 8)
        contested = make_info(
            coins=12,
            auction=AuctionCard(AuctionType.LOAN, 20),
            others=(_player(1, 30), _player(2, 30)),
            gems_remaining=12,
        )
        loan_bid = _bid(contested)
        self.assertIs(type(loan_bid), int)
        self.assertGreaterEqual(loan_bid, 0)
        self.assertLessEqual(loan_bid, 12 + 20)
        invest_bid = _bid(
            make_info(
                coins=12,
                auction=AuctionCard(AuctionType.INVEST, 5),
                others=(_player(1, 30), _player(2, 30)),
                gems_remaining=12,
            )
        )
        self.assertLessEqual(invest_bid, 11)

    def test_reveal_deflates_scarcity_lots_and_hides_rising_ones(self):
        rising = make_info(
            chart_index=0,
            hidden=(BLUE, GREEN),
            auction_gems=(BLUE,),
            others=(_player(1, 10),),
        )
        falling = make_info(
            chart_index=1,
            hidden=(BLUE, GREEN),
            auction_gems=(BLUE,),
            others=(_player(1, 10),),
        )
        self.assertEqual(choose_reveal_index(rising), 1)
        self.assertEqual(choose_reveal_index(falling), 0)
        self.assertEqual(choose_reveal_index(make_info(hidden=())), 0)

    def test_fraction_baseline_bids_a_share_of_fair_value(self):
        info = make_info(
            hidden=(BLUE, BLUE),
            others=(_player(1, 0), _player(2, 0)),
            auction_gems=(BLUE,),
            coins=30,
        )
        self.assertEqual(FractionValueBot().get_bid(info), 4)

    def test_bot_memory_round_trips_through_the_checkpoint(self):
        bot = ValueLiquidityBot()
        bot.on_game_start()
        restored = ValueLiquidityBot.__new__(ValueLiquidityBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(restored.lam, 1.0)
        self.assertEqual(restored.delta, 1.0)

    def test_worst_case_callback_stays_under_half_a_second(self):
        missions = (
            MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),
            MissionCard(10, {GemType.WILD_CARD: 4}, WildCardRequirement.DIFFERENT),
            MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1}),
            MissionCard(5, {YELLOW: 2}),
        )
        info = make_info(
            coins=20,
            hidden=(BLUE, GREEN, PINK),
            auction=AuctionCard(AuctionType.LOAN, 20),
            auction_gems=(BLUE, YELLOW),
            missions=missions,
            others=tuple(_player(pid, 20, hidden_count=3) for pid in range(1, 5)),
            unknown={BLUE: 3, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 3},
            gems_remaining=12,
        )
        bot = ValueLiquidityBot()
        started = time.perf_counter()
        slowest = 0.0
        for _ in range(100):
            call = time.perf_counter()
            bid = bot.get_bid(info)
            slowest = max(slowest, time.perf_counter() - call)
            self.assertIs(type(bid), int)
            self.assertLessEqual(bid, 20 + 20)
        self.assertLess(slowest, 0.5)
        self.assertLess(time.perf_counter() - started, 2.0)
        reveal = bot.choose_hidden_gem(info, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
        self.assertIn(reveal, (0, 1, 2))


class LiveGameTests(unittest.TestCase):
    def test_callbacks_are_legal_on_real_deals(self):
        bot = ValueLiquidityBot()
        for count, chart_index in ((3, 1), (4, 4), (5, 2)):
            with self.subTest(players=count, chart=VALUE_CHARTS[chart_index].name):
                game = create_game(
                    count,
                    rng=Random(count * 17 + chart_index),
                    value_chart=VALUE_CHARTS[chart_index],
                    bots=[Bot] * count,
                )
                self.addCleanup(game.close)
                bot.on_game_start()

                def _check(player: Player, info: BotInfo) -> int:
                    started = time.perf_counter()
                    bid = bot.get_bid(info)
                    self.assertLess(time.perf_counter() - started, 0.5)
                    self.assertIs(type(bid), int)
                    budget = player.coins
                    card = info.current_auction
                    if card is not None and card.type is AuctionType.LOAN:
                        budget += card.value
                    self.assertGreaterEqual(bid, 0)
                    self.assertLessEqual(bid, budget)
                    index = bot.choose_hidden_gem(info, {p.id: 0 for p in game.players})
                    self.assertIs(type(index), int)
                    if info.self_player.hidden_gems:
                        self.assertIn(index, range(len(info.self_player.hidden_gems)))
                    else:
                        self.assertEqual(index, 0)
                    return 0

                with patch.object(Player, "get_bid", _check):
                    while game.auction_gems:
                        play_round(game)


class _Flags:
    """Set module switches for one test and put them back."""

    def __init__(self, **overrides: object) -> None:
        self.overrides = overrides
        self.saved: dict[str, object] = {}

    def __enter__(self) -> "_Flags":
        from bot_definitions import value_liquidity_bot as vl

        for key, value in self.overrides.items():
            self.saved[key] = getattr(vl, key)
            setattr(vl, key, value)
        return self

    def __exit__(self, *args: object) -> None:
        from bot_definitions import value_liquidity_bot as vl

        for key, value in self.saved.items():
            setattr(vl, key, value)


class BidEverything(Bot):
    """Asks for more coins than any legal budget. The harness must reject it."""

    def get_bid(self, info: BotInfo) -> int:
        del info
        return 10**9


class BidLegalMax(Bot):
    """Bids the largest amount the harness is willing to accept."""

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        coins = info.self_player.coins
        if card is not None and card.type is AuctionType.LOAN:
            return coins + card.value
        return max(0, coins)


class FlatBid(Bot):
    """Same bid from every seat, so ties are decided only by the harness."""

    def get_bid(self, info: BotInfo) -> int:
        return min(3, info.self_player.coins)


class TuningTests(unittest.TestCase):
    def test_untuned_search_matches_select_bid_on_live_deals(self):
        from bot_definitions.value_liquidity_bot import search_bid

        params = {
            "lam": 1.0,
            "delta": 1.0,
            "bid_fraction": 0.55,
            "bid_scale": 2.0,
            "laplace_weight": 0.8,
            "denial_weight": 0.4,
        }
        with _Flags(USE_MISSION_PROGRESS=False, USE_QUIET_GATE=False):
            for players, chart_index, seed in (
                (3, 0, 1),
                (4, 1, 2),
                (5, 4, 3),
                (3, 3, 9),
            ):
                game = create_game(
                    players,
                    rng=Random(seed),
                    value_chart=VALUE_CHARTS[chart_index],
                    bots=[Bot] * players,
                )
                self.addCleanup(game.close)
                seen = 0
                while game.auction_gems and seen < 5:
                    card = game.auction_deck.draw()
                    self.assertIsNotNone(card)
                    game.round_number += 1
                    game.current_auction = card
                    game.auction_history.append(card)
                    info = build_bot_info(game, game.players[0])
                    self.assertEqual(
                        search_bid(info, **params), select_bid(info, **params)
                    )
                    seen += 1

    def test_plan_margin_buys_profitable_gems_in_order(self):
        from bot_definitions.value_liquidity_bot import (
            loan_plan_surplus,
            plan_margin,
        )

        # Price 2. First exposed gem is worth 5, the second is not worth buying.
        # One future gem is worth 4. Budget 10 buys the first and the future gem.
        self.assertAlmostEqual(plan_margin(10, 2, [5, 1], 4, 1), 5)
        self.assertAlmostEqual(plan_margin(0, 2, [5], 4, 1), 0)
        self.assertAlmostEqual(plan_margin(2, 2, [8], 4, 0), 6)
        # Already able to afford the only gem: the loan's bid is a pure loss.
        broke = make_info(
            coins=0,
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(BLUE,),
            others=(_player(1, 0), _player(2, 0)),
            gems_remaining=4,
        )
        rich = make_info(
            coins=6,
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(BLUE,),
            others=(_player(1, 0), _player(2, 0)),
            gems_remaining=4,
        )
        # Rivals are broke, so the plan's price is 1. A known blue at display 0
        # with no hidden cards left is worth 4 on chart A.
        outlook = (1.0, [4.0], 0.0, 0.0)
        self.assertAlmostEqual(loan_plan_surplus(broke, 10, 1, outlook), 2.0)
        self.assertAlmostEqual(loan_plan_surplus(rich, 10, 1, outlook), -1.0)

    def test_covering_bid_adds_one_without_priority_and_stays_legal(self):
        from bot_definitions.value_liquidity_bot import apply_covering_bid

        # Previous winner is seat 2, so seat 0 wins ties and seat 1 does not.
        # Chart B at a known count of 0 is 20. Rivals hold 3 and 1 coin.
        favored = make_info(
            chart_index=1,
            coins=10,
            self_id=0,
            previous=2,
            others=(_player(1, 3), _player(2, 1)),
            auction_gems=(BLUE,),
            gems_remaining=15,
            hidden=(),
        )
        trailer = make_info(
            chart_index=1,
            coins=10,
            self_id=1,
            previous=2,
            others=(_player(0, 3), _player(2, 1)),
            auction_gems=(BLUE,),
            gems_remaining=15,
            hidden=(),
        )
        with _Flags(USE_COVERING=True):
            # Seat 0 matches the larger stack. Seat 1 must bid one more.
            self.assertEqual(min_covering_bid(favored), 3)
            self.assertEqual(min_covering_bid(trailer), 4)
            self.assertEqual(
                apply_covering_bid(
                    favored, 0, lam=1.0, delta=1.0, denial_weight=0.4
                ),
                3,
            )
            self.assertEqual(
                apply_covering_bid(
                    trailer, 0, lam=1.0, delta=1.0, denial_weight=0.4
                ),
                4,
            )
        # A tight cap refuses a cover that would spend the whole rival stack.
        with _Flags(USE_COVERING=True, COVER_MAX=2):
            self.assertEqual(
                apply_covering_bid(
                    trailer, 0, lam=1.0, delta=1.0, denial_weight=0.4
                ),
                0,
            )
            # A bid past the covering price is waste: both win for sure.
            self.assertEqual(
                apply_covering_bid(
                    favored, 9, lam=1.0, delta=1.0, denial_weight=0.4
                ),
                3,
            )
        # The switch is off, so the model bid is left alone.
        self.assertEqual(
            apply_covering_bid(trailer, 1, lam=1.0, delta=1.0, denial_weight=0.4),
            1,
        )
        short = make_info(
            chart_index=1,
            coins=2,
            self_id=1,
            previous=2,
            others=(_player(0, 3), _player(2, 1)),
            auction_gems=(BLUE,),
            gems_remaining=15,
        )
        with _Flags(USE_COVERING=True):
            self.assertEqual(
                apply_covering_bid(short, 0, lam=1.0, delta=1.0, denial_weight=0.4),
                0,
            )

    def test_covering_price_rises_after_we_win_the_tiebreak(self):
        # Seat 0 has priority while seat 2 is the previous winner.
        before = make_info(
            self_id=0,
            previous=2,
            others=(_player(1, 4), _player(2, 4)),
        )
        # After seat 0 wins, it is last, and the same stacks cost one more coin.
        after = make_info(
            self_id=0,
            previous=0,
            others=(_player(1, 4), _player(2, 4)),
        )
        self.assertEqual(min_covering_bid(before), 4)
        self.assertEqual(min_covering_bid(after), 5)

    def test_feature_switches_do_not_change_subclasses_or_the_frozen_copy(self):
        from bot_definitions.adaptive_bot import AdaptiveBot
        from bot_definitions.baselines import MyopicEvBot
        from bot_definitions.frozen_vl import FrozenVL
        from bot_definitions import value_liquidity_bot as vl

        info = make_info(
            chart_index=1,
            coins=12,
            hidden=(BLUE, GREEN),
            auction_gems=(BLUE,),
            others=(_player(1, 4), _player(2, 2)),
            gems_remaining=12,
            previous=2,
        )
        myopic_before = MyopicEvBot().get_bid(info)
        adaptive = AdaptiveBot()
        adaptive_before = adaptive.get_bid(info)
        reveal_before = adaptive.choose_hidden_gem(info, {0: 1, 1: 1, 2: 1})
        frozen_before = FrozenVL().get_bid(info)
        with _Flags(
            USE_PHASE_LAMBDA=True,
            PHASE_EARLY=2.0,
            USE_LATE_FACE=True,
            LATE_FACE_GEMS=20,
            USE_COVERING=True,
            USE_LOAN_PLAN=True,
            USE_INVEST_NET=True,
            USE_MISSION_PROGRESS=True,
            SCARCITY_MODE="face",
            REVEAL_MODE="first",
        ):
            self.assertEqual(MyopicEvBot().get_bid(info), myopic_before)
            self.assertEqual(AdaptiveBot().get_bid(info), adaptive_before)
            self.assertEqual(
                AdaptiveBot().choose_hidden_gem(info, {0: 1, 1: 1, 2: 1}),
                reveal_before,
            )
            self.assertEqual(FrozenVL().get_bid(info), frozen_before)
            self.assertEqual(ValueLiquidityBot().choose_hidden_gem(info, {}), 0)
        self.assertEqual(vl.REVEAL_MODE, "strategic")
        self.assertFalse(vl.USE_COVERING)
        self.assertFalse(vl.USE_LOAN_PLAN)
        self.assertEqual(
            select_bid(
                info,
                lam=0.0,
                delta=0.0,
                bid_fraction=0.55,
                bid_scale=2.0,
                laplace_weight=0.8,
                denial_weight=0.0,
            ),
            myopic_before,
        )

    def test_default_bot_matches_the_frozen_snapshot(self):
        from bot_definitions.frozen_vl import FrozenVL

        info = make_info(
            coins=18,
            hidden=(BLUE,),
            auction_gems=(GREEN, BLUE),
            auction=AuctionCard(AuctionType.AUCTION, 2),
            gems_remaining=9,
        )
        self.assertEqual(ValueLiquidityBot().get_bid(info), FrozenVL().get_bid(info))
        loan = make_info(
            coins=8,
            auction=AuctionCard(AuctionType.LOAN, 20),
            gems_remaining=11,
        )
        self.assertEqual(ValueLiquidityBot().get_bid(loan), FrozenVL().get_bid(loan))
        # The kept progress term is the one place the live bot leaves the snapshot.
        partial = make_info(
            chart_index=1,
            coins=25,
            hidden=(),
            auction_gems=(BLUE,),
            missions=(MissionCard(10, {BLUE: 2}),),
            others=(_player(1, 12), _player(2, 12)),
            gems_remaining=10,
            previous=2,
        )
        self.assertGreater(
            ValueLiquidityBot().get_bid(partial), FrozenVL().get_bid(partial)
        )
        from bot_definitions.baselines import MyopicEvBot

        myopic = MyopicEvBot()
        self.assertNotIn("quiet_low", myopic.__dict__)
        self.assertNotIn("quiet_seen", myopic.__dict__)

    def test_pickle_round_trip_and_all_switches_stay_under_the_budget(self):
        from bot_definitions.frozen_vl import FrozenVL
        from bot_definitions.value_liquidity_bot import search_bid

        bot = ValueLiquidityBot()
        restored = ValueLiquidityBot.__new__(ValueLiquidityBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(restored.lam, bot.lam)
        self.assertEqual(restored.bid_fraction, bot.bid_fraction)
        frozen = FrozenVL()
        frozen_restored = FrozenVL.__new__(FrozenVL)
        frozen_restored.load_state(pickle.loads(pickle.dumps(frozen.save_state())))
        self.assertEqual(frozen_restored.lam, 1.0)
        missions = (
            MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),
            MissionCard(10, {GemType.WILD_CARD: 4}, WildCardRequirement.DIFFERENT),
            MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1}),
            MissionCard(5, {YELLOW: 2}),
        )
        info = make_info(
            coins=20,
            hidden=(BLUE, GREEN, PINK),
            auction=AuctionCard(AuctionType.LOAN, 20),
            auction_gems=(BLUE, YELLOW),
            missions=missions,
            others=tuple(_player(pid, 20, hidden_count=3) for pid in range(1, 5)),
            unknown={BLUE: 3, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 3},
            gems_remaining=12,
        )
        with _Flags(
            USE_PHASE_LAMBDA=True,
            USE_LATE_FACE=True,
            USE_COVERING=True,
            USE_LOAN_PLAN=True,
            USE_INVEST_NET=True,
            USE_MISSION_PROGRESS=True,
            SCARCITY_MODE="conservative",
            REVEAL_MODE="random",
        ):
            started = time.perf_counter()
            slowest = 0.0
            for _ in range(50):
                call = time.perf_counter()
                bid = bot.get_bid(info)
                slowest = max(slowest, time.perf_counter() - call)
                self.assertIs(type(bid), int)
                self.assertGreaterEqual(bid, 0)
                self.assertLessEqual(bid, 20 + 20)
            self.assertLess(slowest, 0.3)
            self.assertLess(time.perf_counter() - started, 2.0)
            direct = search_bid(
                info,
                lam=1.0,
                delta=1.0,
                bid_fraction=0.55,
                bid_scale=2.0,
                laplace_weight=0.8,
                denial_weight=0.4,
            )
            self.assertIs(type(direct), int)
            self.assertLessEqual(direct, 40)
            empty = make_info(hidden=(), coins=0)
            self.assertEqual(bot.choose_hidden_gem(empty, {}), 0)
            self.assertEqual(bot.get_bid(make_info(coins=0, hidden=())), 0)

    def test_mission_progress_does_not_recount_a_finished_mission(self):
        from bot_definitions.value_liquidity_bot import mission_progress_value

        pair = MissionCard(5, {GemType.WILD_CARD: 2}, WildCardRequirement.SAME)
        exact = MissionCard(5, {BLUE: 1, GREEN: 1})
        finishing = make_info(
            won=_counts({BLUE: 1}),
            auction_gems=(BLUE,),
            missions=(pair, exact),
        )
        with _Flags(USE_MISSION_PROGRESS=True, MISSION_PROGRESS_WEIGHT=1.0):
            # The pair completes, so it is not counted again. The exact mission
            # already had its blue and gains no green from this lot.
            self.assertAlmostEqual(mission_progress_value(finishing, (BLUE,)), 0.0)
            self.assertEqual(mission_gain(_counts({BLUE: 1}), (BLUE,), (pair,)), 5)
            partial = make_info(
                won=_counts(),
                auction_gems=(BLUE,),
                missions=(exact,),
            )
            self.assertAlmostEqual(mission_progress_value(partial, (BLUE,)), 2.5)

    def test_invest_discount_and_chart_lambda_move_the_bid_the_right_way(self):
        invest = make_info(
            coins=18,
            auction=AuctionCard(AuctionType.INVEST, 10),
            others=(_player(1, 12), _player(2, 12)),
            gems_remaining=12,
            previous=2,
            self_id=0,
        )
        scarce = make_info(
            chart_index=3,
            coins=20,
            hidden=(),
            auction_gems=(BLUE,),
            others=(_player(1, 14), _player(2, 14)),
            gems_remaining=12,
            previous=2,
            self_id=0,
        )
        base_invest = ValueLiquidityBot().get_bid(invest)
        base_scarce = ValueLiquidityBot().get_bid(scarce)
        with _Flags(INVEST_LAM_MULT=0.25):
            cheaper_lock = ValueLiquidityBot().get_bid(invest)
        with _Flags(CHART_LAM_D=2.0):
            dearer = ValueLiquidityBot().get_bid(scarce)
        self.assertGreaterEqual(cheaper_lock, base_invest)
        self.assertLessEqual(dearer, base_scarce)
        self.assertNotEqual(cheaper_lock, base_invest)

    def test_quiet_rivals_turn_off_mission_progress(self):
        pair = MissionCard(10, {BLUE: 2})
        info = make_info(
            chart_index=1,
            coins=25,
            won=_counts(),
            hidden=(),
            auction_gems=(BLUE,),
            missions=(pair,),
            others=(_player(1, 12), _player(2, 12)),
            gems_remaining=10,
            previous=2,
        )
        result = RoundResult(
            AuctionCard(AuctionType.AUCTION, 1),
            FrozenMap({0: 3, 1: 0, 2: 1}),
            0,
            3,
            None,
        )
        bot = ValueLiquidityBot()
        bot.on_game_start()
        with _Flags(
            USE_MISSION_PROGRESS=True,
            MISSION_PROGRESS_WEIGHT=1.0,
            MISSION_BID_CAP=6,
            USE_QUIET_GATE=True,
        ):
            eager = bot.get_bid(info)
            for _ in range(3):
                bot.on_round_end(info, result)
            quiet = bot.get_bid(info)
        with _Flags(USE_MISSION_PROGRESS=False):
            plain = ValueLiquidityBot().get_bid(info)
        self.assertGreater(eager, plain)
        self.assertEqual(quiet, plain)
        restored = ValueLiquidityBot.__new__(ValueLiquidityBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(restored.quiet_seen, bot.quiet_seen)
        self.assertEqual(restored.quiet_low, bot.quiet_low)

    def test_fallback_bid_is_legal_when_pricing_raises(self):
        info = make_info()
        with patch(
            "bot_definitions.value_liquidity_bot.select_bid",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(ValueLiquidityBot().get_bid(info), 0)


class HarnessRuleTests(unittest.TestCase):
    def test_tiebreak_order_and_the_rotation_after_a_win(self):
        from harness import _select_winner

        rng = Random(0)
        # Previous winner is seat 1. Priority starts at seat 2.
        self.assertEqual(_select_winner([4, 4, 4], 1, rng), 2)
        self.assertEqual(_select_winner([5, 5, 0], 2, rng), 0)
        # After seat 0 wins, seat 0 is last and seat 1 takes the next tie.
        self.assertEqual(_select_winner([5, 5, 0], 0, rng), 1)
        opening = {_select_winner([2, 2, 2], None, Random(seed)) for seed in range(40)}
        self.assertEqual(opening, {0, 1, 2})

    def test_over_budget_bid_is_zero_and_loan_proceeds_count(self):
        for players, chart_index, bots in (
            (3, 0, [BidEverything] * 3),
            (4, 2, [BidLegalMax] * 4),
        ):
            game = create_game(
                players,
                rng=Random(11 + players + chart_index),
                value_chart=VALUE_CHARTS[chart_index],
                bots=bots,
                bot_timeout=None,
            )
            self.addCleanup(game.close)
            rounds = 0
            legal_max = bots[0] is BidLegalMax
            while game.auction_gems and rounds < 8:
                before = [player.coins for player in game.players]
                play_round(game)
                for player in game.players:
                    self.assertIsNone(player.bot.last_error)
                card = game.auction_history[-1]
                winner = game.previous_winner_id
                self.assertIsNotNone(winner)
                assert winner is not None
                coins = game.players[winner].coins
                if legal_max:
                    # coins, or coins plus the loan principal, is a legal bid.
                    self.assertEqual(coins, 0)
                elif card.type is AuctionType.LOAN:
                    # One past the cap is rejected, so the principal is not spent.
                    self.assertEqual(coins, before[winner] + card.value)
                else:
                    self.assertEqual(coins, before[winner])
                rounds += 1
            self.assertGreater(rounds, 0)

    def test_live_tie_rotates_to_the_seat_after_the_winner(self):
        game = create_game(
            3,
            rng=Random(4),
            value_chart=VALUE_CHARTS[0],
            bots=[FlatBid, FlatBid, FlatBid],
            bot_timeout=None,
        )
        self.addCleanup(game.close)
        play_round(game)
        first = game.previous_winner_id
        self.assertIsNotNone(first)
        play_round(game)
        second = game.previous_winner_id
        assert first is not None
        self.assertEqual(second, (first + 1) % 3)

    def test_score_and_final_display_follow_the_deal(self):
        from harness import finish_game

        game = create_game(
            4,
            rng=Random(8),
            value_chart=VALUE_CHARTS[3],
            bots=[Bot, FlatBid, BidEverything, Bot],
            bot_timeout=None,
        )
        self.addCleanup(game.close)
        dealt: Counter[GemType] = Counter()
        for player in game.players:
            dealt.update(player.hidden_gem_hand)
        while game.auction_gems:
            play_round(game)
        result = finish_game(game)
        self.assertEqual(
            {gem: game.value_display[gem] for gem in STANDARD_GEMS},
            {gem: dealt[gem] for gem in STANDARD_GEMS},
        )
        gem_values = {
            gem: game.value_chart.get_value(game.value_display[gem])
            for gem in STANDARD_GEMS
        }
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
                player.coins + investment_returns + gem_value + missions - loan_repay
            )
            self.assertEqual(score, result.scores[seat])
            self.assertEqual(
                score,
                player.coins
                + player.invested_coins
                + gem_value
                + missions
                + sum(card.end_game_value for card in player.auction_cards),
            )


if __name__ == "__main__":
    unittest.main()
