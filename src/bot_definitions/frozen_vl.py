"""Frozen copy of ValueLiquidityBot from before this tuning pass.

``OldVL`` in ``strength_variants.py`` is not a snapshot. It subclasses the live
``ValueLiquidityBot`` and inherits whatever ``get_bid`` that class has now.
This module duplicates the pre-tuning callbacks and the constants they closed
over. Valuation still goes through ``pricing.select_bid``, which this pass
does not change, so the reference stays put while ``ValueLiquidityBot`` moves.

Contributors: CONTRIBUTORS
"""

import time
from collections.abc import Mapping

from bot_definitions.pricing import choose_reveal_index, select_bid
from bots import Bot, BotInfo, GameSummary, RoundResult


class FrozenVL(Bot):
    """Pre-tuning ValueLiquidityBot. Parameters are fixed on the class."""

    # Shadow price of one coin above the point it scores at the end.
    lam = 1.0
    # Shade subtracted from gem value on any positive bid.
    delta = 1.0
    bid_fraction = 0.55
    bid_scale = 2.0
    laplace_weight = 0.8
    denial_weight = 0.4
    time_budget_seconds = 0.25

    def __init__(self) -> None:
        self.lam = 1.0
        self.delta = 1.0
        self.bid_fraction = 0.55
        self.bid_scale = 2.0
        self.laplace_weight = 0.8
        self.denial_weight = 0.4

    def on_game_start(self) -> None:
        try:
            return None
        except Exception:
            return None

    def get_bid(self, info: BotInfo) -> int:
        started = time.perf_counter()
        try:
            if info.current_auction is None:
                return 0
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
        except Exception:
            return 0

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        del bids
        started = time.perf_counter()
        try:
            if not info.self_player.hidden_gems:
                return 0
            if time.perf_counter() - started > self.time_budget_seconds:
                return 0
            return choose_reveal_index(info)
        except Exception:
            return 0

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
