"""Pricing, legality, tiebreaks, missions, and callback timing."""

import pickle
import time
import unittest
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
from bots import Bot, BotInfo, PlayerInfo, SelfInfo
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


if __name__ == "__main__":
    unittest.main()
