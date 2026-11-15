"""Baselines for local evaluation. Not exported from custom_bots.

Contributors: baselines only, not a submission.
"""

from bots import Bot, BotInfo
from constants import AuctionType

from bot_definitions.pricing import color_expectations, lot_gems
from bot_definitions.value_liquidity_bot import ValueLiquidityBot


class FractionValueBot(Bot):
    """Bid a fixed fraction of fair gem value, with no liquidity premium.

    Investments are bid at the bonus (the bid is returned). Loans are bid at
    one fifth of the principal. Treasure bids ignore missions, tiebreaks, and
    opponent coin counts.

    Contributors: baselines only, not a submission.
    """

    # TODO: tune — fraction of expected gem value offered for a treasure.
    fraction = 0.55

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        coins = info.self_player.coins
        if card.type is AuctionType.INVEST:
            return min(coins, card.value)
        if card.type is AuctionType.LOAN:
            return min(coins + card.value, max(0, card.value // 5))
        lot = lot_gems(info)
        values = color_expectations(info, lot)
        fair = sum(values.get(gem, 0.0) for gem in lot)
        bid = int(round(self.fraction * fair))
        return min(coins, max(0, bid))


class MyopicEvBot(ValueLiquidityBot):
    """Same pricer as ValueLiquidityBot with lambda and delta set to zero.

    Single-auction expected value, with no shade and no shadow price on cash.
    This is the failure mode the liquidity premium is meant to fix.

    Contributors: baselines only, not a submission.
    """

    lam = 0.0
    delta = 0.0
    denial_weight = 0.0
