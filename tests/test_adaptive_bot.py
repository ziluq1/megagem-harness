"""Opponent model, shrinkage, tiebreaks, UCB1, jitter, and ablation flags."""

import pickle
import random
import time
import unittest

from bot_definitions.adaptive_bot import (
    BALANCED_PRESET,
    MAX_OBS,
    PRESETS,
    SHRINK_K,
    AdaptiveAllOffBot,
    AdaptiveBot,
    AdaptiveLayer1Bot,
    AdaptiveLayer12Bot,
    AdaptiveLayer13Bot,
    OpponentTrack,
    jitter_bid,
    robust_center,
    shrink,
    ucb1_index,
    win_curve,
)
from bot_definitions.pricing import select_bid
from bots import BotInfo, GameSummary, RoundResult
from constants import AuctionType
from immutable import FrozenMap
from lib import AuctionCard
from test_value_liquidity import _player, make_info

BASE_PARAMS = {
    "lam": 1.0,
    "delta": 1.0,
    "bid_fraction": 0.55,
    "bid_scale": 2.0,
    "laplace_weight": 0.8,
    "denial_weight": 0.4,
}


def _point(bid: int, cap: int) -> list[float]:
    dist = [0.0] * (cap + 1)
    dist[min(cap, bid)] = 1.0
    return dist


def _training_info() -> BotInfo:
    from constants import GemType

    blue = GemType.BLUE_CIRCLE
    return make_info(
        coins=35,
        hidden=(blue, blue),
        others=(_player(1, 30), _player(2, 30)),
        auction_gems=(blue,),
        previous=2,
        self_id=0,
        gems_remaining=12,
        chart_index=0,
    )


def _feed_ratio(bot: AdaptiveBot, ratio: float, rounds: int) -> None:
    """Show both opponents bidding ``ratio`` times the cached value signal."""
    info = _training_info()
    for _ in range(rounds):
        bot.get_bid(info)
        cache = bot.round_cache
        assert cache is not None
        their = int(round(ratio * cache.value))
        result = RoundResult(
            AuctionCard(AuctionType.AUCTION, 1),
            FrozenMap({0: 0, 1: their, 2: their}),
            1,
            their,
            None,
        )
        bot.on_round_end(info, result)


class ShrinkageAndRobustTests(unittest.TestCase):
    def test_shrinkage_ignores_two_or_three_observations(self):
        prior = 0.55
        observed = 0.20
        for n in (2, 3):
            estimate = shrink(observed, prior, n, SHRINK_K)
            self.assertAlmostEqual(
                estimate, (n * observed + SHRINK_K * prior) / (n + SHRINK_K)
            )
            self.assertLess(abs(estimate - prior), abs(estimate - observed))

    def test_shrinkage_follows_a_long_sample(self):
        estimate = shrink(0.20, 0.55, 24, SHRINK_K)
        self.assertLess(abs(estimate - 0.20), abs(estimate - 0.55))

    def test_trimmed_mean_drops_an_outlier(self):
        self.assertEqual(robust_center([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]), 0.0)
        self.assertEqual(robust_center([1.0, 3.0]), 2.0)


class OpponentModelTests(unittest.TestCase):
    def test_constant_ratio_is_learned_and_not_trusted_early(self):
        bot = AdaptiveLayer1Bot()
        bot.on_game_start()
        _feed_ratio(bot, 0.25, 2)
        early = bot.ratio_estimate(1, "AUCTION")
        self.assertLess(abs(early - 0.55), abs(early - 0.25))
        fresh = AdaptiveLayer1Bot()
        fresh.on_game_start()
        lot = _training_info()
        uninformed = fresh.get_bid(lot)
        _feed_ratio(bot, 0.25, 10)
        learned = bot.ratio_estimate(1, "AUCTION")
        # Ten more observations on top of the first two.
        self.assertLess(learned, 0.42)
        self.assertGreater(learned, 0.20)
        trained_bid = bot.get_bid(lot)
        self.assertLessEqual(trained_bid, uninformed)
        track = bot.opponents[1]
        self.assertGreater(len(track.residuals), 0)
        self.assertLessEqual(len(track.observations), MAX_OBS)
        self.assertLessEqual(len(track.ratios), MAX_OBS)
        self.assertLessEqual(len(track.zero_flags), MAX_OBS)
        self.assertLessEqual(len(track.allin_flags), MAX_OBS)

    def test_observation_log_matches_the_bid_time_cache(self):
        bot = AdaptiveLayer1Bot()
        bot.on_game_start()
        info = make_info(
            coins=20,
            others=(_player(1, 12), _player(2, 9)),
            previous=2,
            gems_remaining=10,
        )
        bot.get_bid(info)
        cache = bot.round_cache
        self.assertIsNotNone(cache)
        assert cache is not None
        result = RoundResult(
            AuctionCard(AuctionType.AUCTION, 1),
            FrozenMap({0: 1, 1: 4, 2: 0}),
            1,
            4,
            None,
        )
        bot.on_round_end(info, result)
        obs = bot.opponents[1].observations[-1]
        bid, value, coins, kind, round_number, predicted = obs
        self.assertEqual(bid, 4)
        self.assertAlmostEqual(value, cache.value)
        self.assertEqual(coins, 12)
        self.assertEqual(kind, "AUCTION")
        self.assertEqual(round_number, info.round_number)
        self.assertAlmostEqual(predicted, cache.means[1])
        self.assertAlmostEqual(bot.opponents[1].residuals[-1], 4 - predicted)
        self.assertNotIn(0, bot.opponents)

    def test_lists_stay_capped(self):
        track = OpponentTrack()
        for index in range(40):
            track.record(index % 3, 4.0, 10, "AUCTION", index, 1.0)
        self.assertEqual(len(track.observations), MAX_OBS)
        self.assertLessEqual(len(track.ratios), MAX_OBS)
        self.assertLessEqual(len(track.residuals), MAX_OBS)
        self.assertEqual(track.observations[0][4], 40 - MAX_OBS)

    def test_no_history_matches_the_base_bid_when_ties_are_deterministic(self):
        bot = AdaptiveLayer1Bot()
        bot.on_game_start()
        info = make_info(
            coins=18,
            hidden=(),
            others=(_player(1, 18), _player(2, 18)),
            previous=1,
            self_id=0,
            gems_remaining=12,
            chart_index=2,
        )
        self.assertEqual(bot.get_bid(info), select_bid(info, **BASE_PARAMS))

    def test_support_never_exceeds_coins_or_loan_proceeds(self):
        bot = AdaptiveLayer1Bot()
        bot.on_game_start()
        treasure = make_info(
            coins=15,
            others=(_player(1, 7), _player(2, 0)),
            previous=2,
        )
        context = bot._lot_context(treasure)
        assert context is not None
        rich, _mean = bot._distribution(treasure, treasure.other_players[0], context)
        broke, _mean = bot._distribution(treasure, treasure.other_players[1], context)
        self.assertEqual(len(rich) - 1, 7)
        self.assertEqual(broke, [1.0])
        loan = make_info(
            coins=15,
            auction=AuctionCard(AuctionType.LOAN, 20),
            others=(_player(1, 4),),
            previous=1,
            self_id=0,
        )
        loan_context = bot._lot_context(loan)
        assert loan_context is not None
        dist, _mean = bot._distribution(loan, loan.other_players[0], loan_context)
        self.assertEqual(len(dist) - 1, 4 + 20)
        self.assertAlmostEqual(sum(dist), 1.0)


class WinProbabilityTests(unittest.TestCase):
    def test_curve_is_monotone_including_opening_ties(self):
        cases = [
            (
                [( [0.2, 0.2, 0.3, 0.2, 0.1], 1), ([0.5, 0.0, 0.0, 0.5], 2)],
                0,
                2,
                3,
                False,
            ),
            (
                [(_point(3, 6), 1), (_point(3, 6), 2)],
                0,
                None,
                3,
                True,
            ),
            (
                [([0.1, 0.4, 0.4, 0.1], 2), ([0.0, 0.0, 1.0], 3), (_point(1, 4), 4)],
                1,
                0,
                5,
                True,
            ),
        ]
        for distributions, me, previous, players, _flag in cases:
            for exact in (False, True):
                curve = win_curve(
                    8,
                    distributions,
                    me,
                    previous,
                    players,
                    exact_opening=exact,
                )
                for left, right in zip(curve, curve[1:]):
                    self.assertGreaterEqual(right + 1e-9, left)

    def test_tiebreak_priority_and_opening_coin_flip(self):
        # Previous winner is seat 0, so seat 1 wins ties and seat 0 loses them.
        favored = win_curve(5, [(_point(4, 5), 2)], 1, 0, 3, exact_opening=True)
        trailer = win_curve(5, [(_point(4, 5), 1)], 0, 0, 3, exact_opening=True)
        self.assertAlmostEqual(favored[3], 0.0)
        self.assertAlmostEqual(favored[4], 1.0)
        self.assertAlmostEqual(favored[5], 1.0)
        self.assertAlmostEqual(trailer[4], 0.0)
        self.assertAlmostEqual(trailer[5], 1.0)
        opening = win_curve(5, [(_point(4, 5), 1)], 0, None, 3, exact_opening=True)
        self.assertAlmostEqual(opening[3], 0.0)
        self.assertAlmostEqual(opening[4], 0.5)
        self.assertAlmostEqual(opening[5], 1.0)
        both = win_curve(
            5,
            [(_point(4, 5), 1), (_point(4, 5), 2)],
            0,
            None,
            3,
            exact_opening=True,
        )
        self.assertAlmostEqual(both[4], 1.0 / 3.0)
        self.assertAlmostEqual(both[5], 1.0)
        # Seat 1 has priority over seat 2 and loses to nobody else at a tie.
        mixed = win_curve(
            4,
            [([0.5, 0.0, 0.0, 0.5], 2), (_point(3, 4), 0)],
            1,
            0,
            3,
            exact_opening=True,
        )
        # Opponent 0 bids 3 and has worse priority than seat 1? 
        # previous 0, keys: (id - 1) % 3. id 1 -> 0, id 2 -> 1, id 0 -> 2.
        # Seat 1 beats both on a tie. Opponent 2 ties at 3 half the time.
        self.assertAlmostEqual(mixed[2], 0.0)
        self.assertAlmostEqual(mixed[3], 1.0)

    def test_bid_above_every_budget_wins_for_sure(self):
        curve = win_curve(
            6,
            [(_point(3, 3), 1), (_point(2, 2), 2)],
            0,
            1,
            3,
            exact_opening=True,
        )
        self.assertAlmostEqual(curve[4], 1.0)
        self.assertAlmostEqual(curve[6], 1.0)


class PortfolioTests(unittest.TestCase):
    def test_ucb1_explores_each_preset_then_favors_the_best(self):
        counts = [0.0, 0.0, 0.0, 0.0]
        totals = [0.0, 0.0, 0.0, 0.0]
        picks = []
        for _ in range(4):
            choice = ucb1_index(counts, totals, balanced=BALANCED_PRESET)
            picks.append(choice)
            counts[choice] += 1.0
            totals[choice] += 0.0
        self.assertEqual(picks[0], BALANCED_PRESET)
        self.assertEqual(set(picks), set(range(len(PRESETS))))
        counts = [10.0, 10.0, 10.0, 10.0]
        totals = [0.0, 0.0, 100.0, 0.0]
        self.assertEqual(ucb1_index(counts, totals, balanced=BALANCED_PRESET), 2)

    def test_zero_prior_games_stay_on_the_balanced_preset(self):
        bot = AdaptiveLayer12Bot()
        bot.preset_counts = [5.0, 0.0, 5.0, 5.0]
        bot.preset_totals = [50.0, 0.0, 50.0, 50.0]
        bot.games_completed = 0
        bot.on_game_start()
        self.assertEqual(bot.preset_index, BALANCED_PRESET)
        self.assertEqual(bot.lam, PRESETS[BALANCED_PRESET].lam)

    def test_portfolio_cycles_presets_then_keeps_the_winner(self):
        bot = AdaptiveLayer12Bot()
        info = make_info()
        summary = GameSummary(FrozenMap({0: 10, 1: 10, 2: 10}), (0, 1, 2), 10)
        seen = []
        for _ in range(4):
            bot.on_game_start()
            seen.append(bot.preset_index)
            bot.on_game_end(info, summary)
        self.assertEqual(seen[0], BALANCED_PRESET)
        self.assertEqual(set(seen), set(range(4)))
        bot.preset_counts = [6.0, 6.0, 6.0, 6.0]
        bot.preset_totals = [0.0, 0.0, 60.0, 0.0]
        bot.games_completed = 4
        bot.on_game_start()
        self.assertEqual(bot.preset_index, 2)

    def test_disabling_the_portfolio_always_uses_balanced(self):
        bot = AdaptiveLayer1Bot()
        bot.games_completed = 5
        bot.preset_counts = [4.0, 1.0, 4.0, 4.0]
        bot.preset_totals = [40.0, 0.0, 40.0, 40.0]
        bot.on_game_start()
        self.assertEqual(bot.preset_index, BALANCED_PRESET)
        info = make_info()
        summary = GameSummary(FrozenMap({0: 80, 1: 10, 2: 10}), (0,), 80)
        bot.on_game_end(info, summary)
        self.assertEqual(bot.games_completed, 5)


class JitterTests(unittest.TestCase):
    def test_jitter_stays_on_legal_nonnegative_bids(self):
        utilities = [-2.0, 0.9, 1.0, 0.8, -0.4]
        random.seed(7)
        seen = {jitter_bid(utilities, 2, 1.0, 0.25) for _ in range(200)}
        self.assertTrue(seen)
        self.assertTrue(seen <= {1, 2, 3})
        for bid in seen:
            self.assertGreaterEqual(utilities[bid], 0.0)
            self.assertLessEqual(bid, len(utilities) - 1)

    def test_a_clear_best_bid_is_never_left(self):
        utilities = [0.0, 0.2, 5.0, 0.2, -1.0]
        random.seed(1)
        for _ in range(50):
            self.assertEqual(jitter_bid(utilities, 2, 5.0, 0.25), 2)

    def test_jitter_flag_off_does_not_draw(self):
        bot = AdaptiveAllOffBot()
        bot.on_game_start()
        info = make_info(coins=20, previous=1, gems_remaining=9)
        calls = {"n": 0}

        def _boom(*_args: object, **_kwargs: object) -> int:
            calls["n"] += 1
            raise AssertionError("jitter drew a bid while the layer was off")

        original = random.choice
        random.choice = _boom  # type: ignore[method-assign]
        try:
            first = bot.get_bid(info)
            second = bot.get_bid(info)
        finally:
            random.choice = original
        self.assertEqual(calls["n"], 0)
        self.assertEqual(first, second)


class AblationAndMemoryTests(unittest.TestCase):
    def test_each_layer_constant_can_turn_that_layer_off(self):
        self.assertTrue(AdaptiveBot.use_opponent_model)
        self.assertTrue(AdaptiveBot.use_portfolio)
        # Jitter stayed in the code and is off: the 1000–1199 eval did not
        # show a gain over the opponent model plus the portfolio.
        self.assertFalse(AdaptiveBot.use_jitter)
        self.assertTrue(AdaptiveLayer13Bot.use_jitter)
        self.assertFalse(AdaptiveAllOffBot.use_opponent_model)
        self.assertFalse(AdaptiveAllOffBot.use_portfolio)
        self.assertFalse(AdaptiveAllOffBot.use_jitter)
        self.assertFalse(AdaptiveLayer1Bot.use_portfolio)
        self.assertFalse(AdaptiveLayer1Bot.use_jitter)
        self.assertTrue(AdaptiveLayer12Bot.use_portfolio)
        off = AdaptiveAllOffBot()
        off.on_game_start()
        learned = AdaptiveLayer1Bot()
        learned.on_game_start()
        _feed_ratio(learned, 0.0, 12)
        info = make_info(
            coins=30,
            hidden=(),
            others=(_player(1, 30), _player(2, 30)),
            previous=2,
            gems_remaining=11,
            chart_index=1,
        )
        # The disabled model still bids like the base bot on this lot.
        self.assertEqual(off.get_bid(info), select_bid(info, **BASE_PARAMS))
        # The same history is ignored when the model flag is off.
        silent = AdaptiveAllOffBot()
        silent.on_game_start()
        _feed_ratio(silent, 0.0, 12)
        self.assertEqual(silent.opponents, {})
        self.assertEqual(silent.get_bid(info), off.get_bid(info))

    def test_state_survives_pickling(self):
        bot = AdaptiveBot()
        bot.on_game_start()
        _feed_ratio(bot, 0.3, 6)
        bot.games_completed = 2
        bot.preset_counts = [0.5, 1.5, 0.25, 0.0]
        bot.preset_totals = [1.0, -2.0, 3.5, 0.0]
        restored = AdaptiveBot.__new__(AdaptiveBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(restored.games_completed, 2)
        self.assertEqual(restored.preset_counts, [0.5, 1.5, 0.25, 0.0])
        self.assertEqual(restored.preset_totals, [1.0, -2.0, 3.5, 0.0])
        self.assertEqual(
            restored.opponents[1].observations,
            bot.opponents[1].observations,
        )
        self.assertEqual(restored.opponents[1].residuals, bot.opponents[1].residuals)
        info = make_info(previous=2, gems_remaining=8)
        self.assertIs(type(restored.get_bid(info)), int)

    def test_worst_case_callback_stays_under_half_a_second(self):
        from constants import GemType, WildCardRequirement
        from lib import MissionCard

        blue, green, purple, yellow, pink = (
            GemType.BLUE_CIRCLE,
            GemType.GREEN_OVAL,
            GemType.PURPLE_TRIANGLE,
            GemType.YELLOW_SQUARE,
            GemType.PINK_PENTAGON,
        )
        missions = (
            MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),
            MissionCard(10, {GemType.WILD_CARD: 4}, WildCardRequirement.DIFFERENT),
            MissionCard(10, {blue: 1, green: 1, purple: 1}),
            MissionCard(5, {yellow: 2}),
        )
        info = make_info(
            coins=20,
            hidden=(blue, green, pink),
            auction=AuctionCard(AuctionType.LOAN, 20),
            auction_gems=(blue, yellow),
            missions=missions,
            others=tuple(_player(pid, 20, hidden_count=3) for pid in range(1, 5)),
            unknown={
                blue: 3,
                green: 3,
                purple: 3,
                yellow: 3,
                pink: 3,
            },
            gems_remaining=12,
            previous=4,
        )
        bot = AdaptiveBot()
        bot.on_game_start()
        for opponent in info.other_players:
            track = OpponentTrack()
            for index in range(MAX_OBS):
                track.record(index % 5, 6.0, 20, "LOAN", index, 2.0)
            bot.opponents[opponent.id] = track
        slowest = 0.0
        started = time.perf_counter()
        for index in range(40):
            call = time.perf_counter()
            bid = bot.get_bid(info)
            slowest = max(slowest, time.perf_counter() - call)
            self.assertIs(type(bid), int)
            self.assertGreaterEqual(bid, 0)
            self.assertLessEqual(bid, 20 + 20)
            result = RoundResult(
                AuctionCard(AuctionType.LOAN, 20),
                FrozenMap({player: index % 3 for player in range(5)}),
                1,
                index % 3,
                None,
            )
            ended = time.perf_counter()
            bot.on_round_end(info, result)
            slowest = max(slowest, time.perf_counter() - ended)
        self.assertLess(slowest, 0.5)
        self.assertLess(time.perf_counter() - started, 4.0)


if __name__ == "__main__":
    unittest.main()
