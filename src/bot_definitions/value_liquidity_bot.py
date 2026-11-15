"""ValueLiquidityBot: robust analytic bidder for Megagem.

Strategy
--------
Price each treasure from the active value chart and the missions it completes,
then bid as a sealed-bid auction rather than paying that price outright.

The final display count of a color is the gems already showing plus every
hidden starting-hand gem of that color. Our own hand is known. Opponents'
hands are a hypergeometric draw from ``unknown_gems``; the undealt deck is in
that pool but never enters the display. Charts B and D pay for scarcity, and
chart E peaks and then falls, so a higher count is not treated as better.

A legal bid ``b`` is scored as

    P(win | b) * ((gem value + missions + denial - hurdle) - (1 + lambda) * b)

``lambda`` is a shadow price on coins that are spent or locked. Paying it is
what stops the bot from buying every positive-gem auction and then taking bad
loans. The hurdle (``delta``) shades positive bids; a bid of zero ignores it,
so a free gem is still taken. In the last couple of gems both terms go to
zero and leftover cash is spent at face value.

Investments return the bid at the end and add a fixed bonus, so the bid is
not a score cost — only locked liquidity. One coin is held back while treasures
remain. Loans are taken only when the extra coins are cheap relative to what
they can still buy; the principal itself nets to zero.

Opponent bids are a Laplace-uniform mixture around a fraction of a public
value signal, capped by coins and by that signal. Tie priority matches the
harness: the seat after the previous winner wins ties. The first auction
breaks ties at random, so this bot does not rely on winning them. A bid that
clears every rival budget is used whenever the lot is still worth that price.

The reveal does not change the final display. It is chosen to deflate colors
still on the block and to avoid inflating them.

Contributors: CONTRIBUTORS
"""

import os
import time
from collections.abc import Mapping

from bot_definitions.pricing import choose_reveal_index, select_bid
from bots import Bot, BotInfo, GameSummary, RoundResult


def _env_float(name: str, default: float) -> float:
    """Optional experiment override. Unset during tournament play."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


class ValueLiquidityBot(Bot):
    """Analytic value-and-liquidity bidder. See the module docstring."""

    # Shadow price of one coin above the point it scores at the end.
    # Grid-searched against myopic and fraction bidders; 1.0 beat 0.25, 0.5, and 0.75.
    lam = 1.0
    # Shade subtracted from gem value on any positive bid. 1.0 won the same grid.
    delta = 1.0
    # TODO: tune — rival mean bid as a fraction of the public value signal.
    bid_fraction = 0.55
    # TODO: tune — Laplace scale, in coins, of the rival bid distribution.
    bid_scale = 2.0
    # TODO: tune — weight on the Laplace; the rest is a uniform hedge.
    laplace_weight = 0.8
    # TODO: tune — fraction of a rival-only mission counted as denial value.
    denial_weight = 0.4
    # Stop optional work well inside the harness callback budget.
    time_budget_seconds = 0.25

    def __init__(self) -> None:
        self.lam = type(self).lam
        self.delta = type(self).delta
        self.bid_fraction = type(self).bid_fraction
        self.bid_scale = type(self).bid_scale
        self.laplace_weight = type(self).laplace_weight
        self.denial_weight = type(self).denial_weight
        # Sweeps override the submission class only, not baselines that subclass it.
        if type(self) is ValueLiquidityBot:
            self.lam = _env_float("MEGAGEM_VL_LAM", self.lam)
            self.delta = _env_float("MEGAGEM_VL_DELTA", self.delta)
            self.bid_fraction = _env_float("MEGAGEM_VL_FRACTION", self.bid_fraction)
            self.denial_weight = _env_float("MEGAGEM_VL_DENIAL", self.denial_weight)

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
