"""Mission value, covers, loans, reveals, endgame, and ablations."""

import pickle
import time
import unittest
from random import Random

from bot_definitions.aggressive_bot import (
    USE_ENDGAME,
    USE_LIQUIDITY,
    USE_MISSIONS,
    USE_REVEAL,
    AggressiveBot,
    AggressiveNoEndgame,
    AggressiveNoLiquidity,
    AggressiveNoMissions,
    AggressiveNoReveal,
    cover_after_we_win,
    gems_short,
    loan_decision,
    mission_increment,
    reveal_index,
    rival_denial,
)
from bot_definitions.pricing import min_covering_bid
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from harness import play_game
from lib import AuctionCard, MissionCard
from test_value_liquidity import _counts, _player, make_info

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS


def _pair(value: int, left: GemType, right: GemType) -> MissionCard:
    return MissionCard(value, {left: 1, right: 1})


class MissionTests(unittest.TestCase):
    def test_shortfall_matches_satisfaction(self) -> None:
        mission = MissionCard(10, {BLUE: 2})
        self.assertEqual(gems_short(mission, _counts()), 2)
        self.assertGreater(gems_short(mission, _counts()), 0)
        self.assertFalse(mission.is_satisfied(_counts()))
        owned = _counts({BLUE: 2})
        self.assertEqual(gems_short(mission, owned), 0)
        self.assertTrue(mission.is_satisfied(owned))
        wild = MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME)
        self.assertEqual(gems_short(wild, _counts()), 3)
        self.assertEqual(gems_short(wild, _counts({GREEN: 3})), 0)

    def test_completion_is_the_bonus_and_progress_is_only_a_slice(self) -> None:
        mission = MissionCard(10, {BLUE: 2})
        rivals = (_player(1, 0), _player(2, 0))
        done = make_info(
            auction_gems=(BLUE, BLUE),
            gems_remaining=2,
            unknown={},
            missions=(mission,),
            others=rivals,
        )
        full = mission_increment(
            done, (BLUE, BLUE), _counts(), rivals, lot_is_ours=True
        )
        self.assertGreaterEqual(full, 9.0)
        self.assertLessEqual(full, 10.0)
        partial_info = make_info(
            auction_gems=(BLUE,),
            gems_remaining=12,
            unknown={BLUE: 4, GREEN: 4, PURPLE: 4},
            missions=(mission,),
            others=rivals,
        )
        partial = mission_increment(
            partial_info, (BLUE,), _counts(), rivals, lot_is_ours=True
        )
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, full)
        self.assertLess(partial, 0.5 * mission.value)

    def test_last_copy_denial_exceeds_a_mission_they_can_finish_later(self) -> None:
        mission = MissionCard(10, {BLUE: 2})
        close = _player(1, 8, won=_counts({BLUE: 1}))
        idle = _player(2, 0)
        last = make_info(
            auction_gems=(BLUE,),
            gems_remaining=1,
            unknown={},
            missions=(mission,),
            others=(close, idle),
        )
        blocked = rival_denial(last, (BLUE,))
        later = make_info(
            auction_gems=(BLUE,),
            gems_remaining=10,
            unknown={BLUE: 4, GREEN: 4},
            missions=(mission,),
            others=(close, idle),
        )
        delayed = rival_denial(later, (BLUE,))
        self.assertGreater(blocked, 0.8 * mission.value)
        self.assertLess(delayed, 0.75 * mission.value)
        self.assertLess(delayed, blocked)

    def test_one_mission_is_not_added_twice(self) -> None:
        mission = MissionCard(10, {BLUE: 2})
        rivals = (_player(1, 0),)
        info = make_info(
            coins=20,
            others=rivals,
            auction_gems=(BLUE, BLUE),
            gems_remaining=2,
            unknown={},
            missions=(mission,),
        )
        both = mission_increment(
            info, (BLUE, BLUE), _counts(), rivals, lot_is_ours=True
        )
        one = mission_increment(
            info, (BLUE,), _counts(), rivals, lot_is_ours=True
        )
        self.assertLessEqual(both, mission.value + 1e-6)
        self.assertLess(one, both)


class CoverAndLoanTests(unittest.TestCase):
    def test_covering_bid_follows_the_tiebreak(self) -> None:
        # Seat 0 just won, so seat 0 is last. Beating seat 1's 4 coins takes 5.
        behind = make_info(
            self_id=0,
            previous=0,
            coins=20,
            others=(_player(1, 4), _player(2, 1)),
        )
        self.assertEqual(min_covering_bid(behind), 5)
        self.assertEqual(cover_after_we_win(behind), 5)
        # Seat 1 is first in line after seat 0, so a matching 4 wins the tie.
        ahead = make_info(
            self_id=1,
            previous=0,
            coins=20,
            others=(_player(0, 4), _player(2, 1)),
        )
        self.assertEqual(min_covering_bid(ahead), 4)
        # The opening tie is a coin flip, so the bot does not count on winning it.
        opening = make_info(
            self_id=0,
            previous=None,
            coins=20,
            others=(_player(1, 4), _player(2, 4)),
        )
        self.assertEqual(min_covering_bid(opening), 5)

    def test_loan_is_refused_without_a_modeled_gain(self) -> None:
        # The visible gem is cheap and the rival stack already prices us out.
        useless = make_info(
            coins=20,
            previous=0,
            self_id=0,
            chart_index=0,
            display={BLUE: 1},
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(BLUE,),
            gems_remaining=10,
            unknown={},
            others=(_player(1, 30), _player(2, 30)),
        )
        bid, net = loan_decision(useless, missions=True)
        self.assertEqual(bid, 0)
        self.assertLessEqual(net, 0.0)
        # We can already pay the cover, so the principal buys nothing new.
        already = make_info(
            coins=5,
            previous=0,
            self_id=0,
            chart_index=1,
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(BLUE,),
            gems_remaining=8,
            unknown={},
            others=(_player(1, 0), _player(2, 0)),
        )
        bid, net = loan_decision(already, missions=False)
        self.assertEqual(bid, 0)
        self.assertLessEqual(net, 0.0)

    def test_loan_is_taken_when_the_plan_clears_the_bid(self) -> None:
        # Chart B pays 20 for a missing color. Broke rivals lose the next tie
        # to a bid of 1, and we lose this tie, so the loan costs 1 to secure.
        planned = make_info(
            coins=0,
            previous=0,
            self_id=0,
            chart_index=1,
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(BLUE,),
            gems_remaining=8,
            unknown={},
            others=(_player(1, 0), _player(2, 0)),
        )
        bid, net = loan_decision(planned, missions=False)
        self.assertGreater(bid, 0)
        self.assertGreater(net, 0.0)
        bot = AggressiveBot()
        bot.use_liquidity = True
        self.assertEqual(bot.get_bid(planned), bid)
        quiet = AggressiveBot()
        quiet.use_liquidity = False
        quiet.use_valued_loans = False
        self.assertEqual(quiet.get_bid(planned), 0)


class RevealEndgameAndSwitchTests(unittest.TestCase):
    def test_reveal_is_a_legal_index_and_empty_hands_do_not_raise(self) -> None:
        bot = AggressiveBot()
        empty = make_info(hidden=())
        self.assertEqual(bot.choose_hidden_gem(empty, {0: 0, 1: 0}), 0)
        mission = MissionCard(10, {BLUE: 1})
        info = make_info(
            coins=20,
            hidden=(BLUE, GREEN),
            auction_gems=(BLUE,),
            gems_remaining=6,
            unknown={},
            missions=(mission,),
            others=(_player(1, 10), _player(2, 10)),
        )
        self.assertEqual(reveal_index(info, {0: 3, 1: 4, 2: 0}), 1)
        self.assertEqual(bot.choose_hidden_gem(info, {0: 3, 1: 4, 2: 0}), 0)
        shown = AggressiveNoReveal()
        shown.use_reveal = True
        self.assertEqual(shown.choose_hidden_gem(info, {0: 3, 1: 4, 2: 0}), 1)

    def test_endgame_bid_stays_inside_the_budget(self) -> None:
        bot = AggressiveBot()
        info = make_info(
            coins=4,
            gems_remaining=2,
            auction_gems=(BLUE,),
            unknown={},
            others=(_player(1, 3), _player(2, 6)),
        )
        bid = bot.get_bid(info)
        self.assertGreaterEqual(bid, 0)
        self.assertLessEqual(bid, 4)
        broke = make_info(coins=0, gems_remaining=2, others=(_player(1, 3),))
        self.assertEqual(bot.get_bid(broke), 0)
        loan = make_info(
            coins=1,
            gems_remaining=2,
            auction=AuctionCard(AuctionType.LOAN, 10),
            auction_gems=(GREEN,),
            others=(_player(1, 4), _player(2, 4)),
        )
        loan_bid = bot.get_bid(loan)
        self.assertGreaterEqual(loan_bid, 0)
        self.assertLessEqual(loan_bid, 11)
        if bot.use_liquidity:
            decision, net = loan_decision(loan, missions=True)
            if loan_bid > 0:
                self.assertGreater(net, 0.0)
                self.assertEqual(loan_bid, decision)

    def test_each_feature_can_be_disabled(self) -> None:
        mission = MissionCard(10, {BLUE: 1})
        info = make_info(
            coins=30,
            gems_remaining=12,
            auction_gems=(BLUE,),
            unknown={},
            missions=(mission,),
            others=(_player(1, 12), _player(2, 12)),
        )
        enabled = AggressiveBot()
        enabled.use_missions = True
        disabled = AggressiveBot()
        disabled.use_missions = False
        self.assertGreater(enabled.get_bid(info), disabled.get_bid(info))
        self.assertFalse(AggressiveNoMissions().use_missions)
        self.assertFalse(AggressiveNoLiquidity().use_liquidity)
        self.assertFalse(AggressiveNoReveal().use_reveal)
        self.assertFalse(AggressiveNoEndgame().use_endgame)
        # The held-out eval turned every idea off. Subclasses can turn one on.
        self.assertFalse(USE_MISSIONS or USE_LIQUIDITY or USE_REVEAL or USE_ENDGAME)
        self.assertTrue(AggressiveNoMissions().use_liquidity)
        import bot_definitions.aggressive_bot as module

        original = module.USE_MISSIONS
        try:
            module.USE_MISSIONS = False
            bot = module.AggressiveBot()
            self.assertFalse(bot.use_missions)
        finally:
            module.USE_MISSIONS = original

    def test_scarcity_charts_are_worth_more_when_the_color_is_missing(self) -> None:
        common = dict(
            coins=25,
            gems_remaining=12,
            auction_gems=(BLUE,),
            unknown={},
            display={},
            others=(_player(1, 8), _player(2, 8)),
        )
        scarce = AggressiveBot().get_bid(make_info(chart_index=1, **common))
        abundant = AggressiveBot().get_bid(make_info(chart_index=0, **common))
        self.assertGreater(scarce, abundant)

    def test_state_survives_pickling(self) -> None:
        bot = AggressiveBot()
        bot.get_bid(make_info())
        bot.fallback_bids = 2
        restored = pickle.loads(pickle.dumps(bot))
        self.assertEqual(restored.fallback_bids, 2)
        self.assertEqual(restored.use_missions, bot.use_missions)
        self.assertEqual(restored.use_liquidity, bot.use_liquidity)
        self.assertEqual(restored.use_reveal, bot.use_reveal)
        self.assertEqual(restored.use_endgame, bot.use_endgame)
        self.assertEqual(restored.clamped_bids, 0)

    def test_worst_case_callbacks_stay_under_half_a_second(self) -> None:
        missions = (
            MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME),
            MissionCard(10, {GemType.WILD_CARD: 4}, WildCardRequirement.DIFFERENT),
            MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),
            MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1}),
        )
        others = tuple(
            _player(index, 20, won=_counts({BLUE: index % 2}))
            for index in range(1, 5)
        )
        info = make_info(
            coins=40,
            hidden=(BLUE, GREEN, PURPLE, YELLOW, PINK),
            others=others,
            auction=AuctionCard(AuctionType.LOAN, 20),
            auction_gems=(BLUE, GREEN),
            gems_remaining=3,
            unknown={gem: 2 for gem in STANDARD_GEMS},
            missions=missions,
            display={BLUE: 2, GREEN: 1},
        )
        bot = AggressiveBot()
        started = time.perf_counter()
        for _ in range(10):
            bid = bot.get_bid(info)
            self.assertGreaterEqual(bid, 0)
            self.assertLessEqual(bid, 60)
        treasure = make_info(
            coins=40,
            hidden=(BLUE, GREEN, PURPLE, YELLOW, PINK),
            others=others,
            auction_gems=(YELLOW, PINK),
            gems_remaining=3,
            unknown={gem: 2 for gem in STANDARD_GEMS},
            missions=missions,
            display={YELLOW: 3, PINK: 4},
            chart_index=3,
        )
        for _ in range(10):
            bid = bot.get_bid(treasure)
            self.assertLessEqual(bid, 40)
            index = bot.choose_hidden_gem(treasure, {0: 2, 1: 1, 2: 0, 3: 4, 4: 0})
            self.assertIn(index, range(5))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed / 20, 0.5)
        self.assertEqual(bot.fallback_bids, 0)

    def test_a_full_game_returns_legal_scores(self) -> None:
        bot = AggressiveBot()
        # play_game builds its own instances; this checks the class is usable.
        result = play_game(
            3,
            bots=(AggressiveBot, AggressiveBot, AggressiveBot),
            rng=Random(7),
            bot_timeout=2.0,
        )
        self.assertEqual(len(result.scores), 3)
        self.assertTrue(all(score >= 0 for score in result.scores))
        self.assertEqual(bot.fallback_bids, 0)


if __name__ == "__main__":
    unittest.main()
