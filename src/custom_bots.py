"""Public bot names for the CLI; add imports and __all__ entries for new bots."""

from bot_definitions.adaptive_bot import AdaptiveBot
from bot_definitions.aggressive_bot import AggressiveBot
from bot_definitions.small_bid_bot import SmallBidBot
from bot_definitions.value_liquidity_bot import ValueLiquidityBot

# SmallBidBot stays importable for the harness example. The three submission
# bots are the names below.
__all__ = ["AdaptiveBot", "AggressiveBot", "ValueLiquidityBot"]
