"""Unusual opponents for the AdaptiveBot evaluation. Not a submission.

Contributors: evaluation only, not a submission.
"""

import random

from bot_definitions.pricing import (
    affordable,
    color_expectations,
    lot_gems,
    mission_gain,
)
from bots import Bot, BotInfo
from constants import AuctionType


class ZeroBidBot(Bot):
    """Bid zero on every card."""

    def get_bid(self, info: BotInfo) -> int:
        del info
        return 0


class OverbidBot(Bot):
    """Bid 1.5 times fair value, capped by the legal budget."""

    def get_bid(self, info: BotInfo) -> int:
        try:
            card = info.current_auction
            if card is None:
                return 0
            budget = affordable(info)
            if card.type is AuctionType.AUCTION:
                lot = lot_gems(info)
                values = color_expectations(info, lot)
                fair = sum(values.get(gem, 0.0) for gem in lot)
            else:
                fair = float(card.value)
            bid = int(round(1.5 * fair))
            return min(budget, max(0, bid))
        except Exception:
            return 0


class RandomBidBot(Bot):
    """Uniform random legal bid. Uses the worker ``random`` stream."""

    def get_bid(self, info: BotInfo) -> int:
        try:
            budget = affordable(info)
            if budget <= 0:
                return 0
            return random.randrange(budget + 1)
        except Exception:
            return 0


class MissionAllInBot(Bot):
    """Bid the entire budget when the lot completes a mission, else zero."""

    def get_bid(self, info: BotInfo) -> int:
        try:
            card = info.current_auction
            if card is None:
                return 0
            budget = affordable(info)
            if card.type is not AuctionType.AUCTION:
                return 0
            lot = lot_gems(info)
            gained = mission_gain(
                info.self_player.won_gems, lot, info.available_missions
            )
            if gained > 0:
                return budget
            return 0
        except Exception:
            return 0
