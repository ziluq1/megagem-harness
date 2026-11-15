"""Importable copies of the three bots with the new rules switched off.

The harness only loads bot classes from an importable module, so the
before/after comparison cannot define these next to the experiment script.
"""

from bot_definitions.adaptive_bot import AdaptiveBot
from bot_definitions.aggressive_bot import AggressiveBot
from bot_definitions.value_liquidity_bot import ValueLiquidityBot


class OldVL(ValueLiquidityBot):
    """ValueLiquidityBot that still buys early loans at the shadow price."""


class OldAdaptive(AdaptiveBot):
    """Unchanged AdaptiveBot, kept so older comparisons still import."""


class OldAggressive(AggressiveBot):
    use_valued_loans = False
