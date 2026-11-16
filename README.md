# Megagem harness

A standard-library-only simulator for 3–5 players, using Python 3.10 or newer.

```sh
python3 src/harness.py 3 --seed 42 --bot-seed 7
python3 src/harness.py 5 --chart C --bots custom_bots:SmallBidBot
python3 src/harness.py 3 --bot-timeout none
```

Without a player-count argument, the CLI asks for one. Chart A is the default.
`--bots` accepts one `module:Class` for everyone or one per player, in seating order.
Every player gets a separate bot instance. The default `Bot` bids zero and reveals
a random hidden gem. `--bot-timeout` is a positive number of seconds or `none`;
it defaults to one second per callback.

## Writing a bot

Put each custom bot implementation in the `src/bot_definitions/` package, then expose
its class through `src/custom_bots.py`. Implementations can live in separate files:

```text
src/
    bot_definitions/
        __init__.py
        small_bid_bot.py
        my_bot.py
    custom_bots.py
```

For example, define your subclass in `src/bot_definitions/my_bot.py`:

```python
from collections.abc import Mapping
from bots import Bot, BotInfo, RoundResult


class MyBot(Bot):
    def __init__(self) -> None:
        self.games_played = 0
        self.rounds_seen = 0

    def on_game_start(self) -> None:
        self.games_played += 1
        self.rounds_seen = 0

    def get_bid(self, info: BotInfo) -> int:
        return min(2, info.self_player.coins)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return 0  # Index into info.self_player.hidden_gems.

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        self.rounds_seen += 1
```

Add its import and public name to `src/custom_bots.py` alongside the existing example:

```python
from bot_definitions.my_bot import MyBot
from bot_definitions.small_bid_bot import SmallBidBot

__all__ = ["MyBot", "SmallBidBot"]
```

Run it with `python3 src/harness.py 3 --bots custom_bots:MyBot`. Multiple implementations
can be selected independently, such as
`--bots custom_bots:MyBot custom_bots:SmallBidBot custom_bots:MyBot`.
Explicit imports make each public bot name easy to find. Keep definitions in their
own modules; importing them into `src/custom_bots.py` makes them available to the CLI.
Workers still import each class from its original `bot_definitions` module, which
also allows checkpoint recovery without any special registration code.

All five callbacks are optional and synchronous:

- `on_game_start() -> None`: called once before the first decision of each game.
- `get_bid(info: BotInfo) -> int`: return a nonnegative, affordable integer bid.
  A loan bid may include the proceeds of that loan.
- `choose_hidden_gem(info: BotInfo, bids: Mapping[int, int]) -> int`: return a valid
  index. `bids` contains validated bids keyed by player ID. Empty hands skip this call.
- `on_round_end(info: BotInfo, result: RoundResult) -> None`: called for every bot
  after the reveal, replenishment, and mission claims. Results include the auction
  card, validated bids, winner ID, winning bid, and revealed gem (possibly `None`).
- `on_game_end(info: BotInfo, result: GameSummary) -> None`: called once after all
  remaining hands are revealed and scoring finishes. Results contain scores keyed
  by player ID, winner IDs, and the winning score. No live game state is included.

`get_bid` defaults to zero, `choose_hidden_gem` defaults to a random valid index,
and the three notifications default to doing nothing. No decorators or async
functions are required. Constructors must accept no arguments. Use ordinary
instance attributes for memory; globals are not checkpointed.

## Information available to bots

`BotInfo` and everything nested within it are immutable snapshots. Count mappings
support normal mapping operations, including `.get()`, `.items()`, and `.values()`.

| Field | Meaning |
|---|---|
| `self_player` | ID, coins, invested coins, hidden gems and hand size, revealed gems, acquired `won_gems`, auction cards, and mission cards |
| `other_players` | Public player information and hidden-hand sizes, ordered by seat; no hidden gem identities |
| `num_players`, `round_number` | Player count and current round; setup is round zero |
| `current_auction` | The current round's auction card, or `None` before the first round |
| `gem_deck_count` | Number of gems in the hidden draw pile |
| `gems_remaining` | Hidden draw pile plus exposed auction gems |
| `unknown_gems` | Full gem catalog minus the display, acquired gems, exposed gems, and this bot's hidden hand; includes opponents' hidden hands and the hidden draw pile together |
| `auction_gems` | Exposed gems in auction order, with the first gem first |
| `value_display` | Counts of revealed starting-hand gems |
| `previous_winner_id` | Previous auction's winner when bidding; current winner after resolution |
| `value_chart`, `available_missions` | The selected chart and unclaimed missions |
| `auction_deck_count` | Number of undrawn auction cards |
| `auction_cards_remaining` | Unordered counts of undrawn auction cards, never their deck order |

Each bidder gets the same public state and their own private hand; current bids
are not disclosed until everyone has bid. The reveal decision sees the auction
award and payment already applied, before any gem reveal or replenishment.
`current_auction` remains the just-resolved card in result notifications.

## Timeouts and memory

Each bot lives in its own persistent spawned worker. It receives snapshots, never
`Player`, `GameState`, or the game's RNG. Only validated integer decisions reach
the game. Invalid bids become zero; invalid reveal choices select a random valid
index. Booleans and floats are invalid decisions. Exceptions and timeouts use the
same fallbacks and log the player and callback responsible.

After every successful callback, the worker checkpoints the bot's instance
attributes and its private `random` module state. On failure, its worker is stopped.
The next call starts a replacement from the last successful checkpoint, without
rerunning `__init__` or `on_game_start`. Changes from the failed callback are lost.
Successful callbacks returning invalid decisions still retain their memory updates.

Ordinary pickleable attributes in `self.__dict__` need no extra code. Snapshots can
themselves be saved in bot memory. Slot-backed fields need custom hooks. For
files, locks, clients, or other resources, override both
`save_state() -> object` and `load_state(state: object) -> None`: export pickleable
memory and recreate resources during restoration. Recovery calls `load_state` on
an instance allocated without running `__init__`. Checkpoints are opaque to the
game process and are unpickled only in a worker. They are in-memory checkpoints,
not saves that persist after the harness exits.

The callback budget includes snapshot transfer and checkpoint creation. Worker
startup/import/restoration has a separate budget of at least five seconds (or the
callback timeout if greater). With `none`, both limits are disabled. Bots can read
`self.timeout_seconds` for their callback allowance. Messages/checkpoints have a
16 MiB transport limit. Process cleanup can add a small delay after a timeout.

`--seed` controls game setup and tiebreaks. `--bot-seed` independently seeds bot
randomness. Use both to reproduce randomized bot games. `import random` inside a
bot uses its own worker's stream. Fallback reveal choices also use separate randomness.
A bot using its own additional random generators should seed and store them itself.

## Python API and reusing bots

Runtime modules live in `src/`. When running your own API script from the project
root, use `PYTHONPATH=src python3 your_script.py` so Python and spawned workers can
find them. The experiment script and notebook add `src/` automatically, so their
commands require no extra setup. No installation or virtual environment is needed.

```python
from random import Random
from bot_runner import BotRunner
from custom_bots import SmallBidBot
from harness import play_game


def main():
    with (
        BotRunner(SmallBidBot) as first,
        BotRunner(SmallBidBot) as second,
        BotRunner(SmallBidBot) as third,
    ):
        bots = [first, second, third]
        for seed in range(2):
            result = play_game(3, bots=bots, rng=Random(seed))
            print(result.scores, result.winner_ids)


if __name__ == "__main__":
    main()
```

A caller can pass bot classes instead, for example `bots=[SmallBidBot] * 3`.
The harness creates and closes those workers automatically. Caller-owned runners
retain memory between games until `on_game_start` selectively resets it. Each
runner's own timeout/seed configuration applies when supplying existing runners.
Separate players must use separate runners.

`play_game()` returns a `GameResult` without printing; use `print_result()` for
console output. Its `state` contains mutable final game state for host inspection.
Bot game-end notifications receive only the immutable `GameSummary`.

For round-by-round inspection, use `with create_game(...) as game`, call
`play_round(game)` while `game.auction_gems` is nonempty, and then `finish_game(game)`.
Repeated finalization neither reveals cards twice nor repeats game-end callbacks.
The main guard shown above is required for scripts that launch spawned workers.
Define custom bot classes at module scope inside `src/bot_definitions/` and expose them
through `src/custom_bots.py`. Avoid defining them inside functions or `__main__`.

## Submitted bots

`src/custom_bots.py` exports exactly `ValueLiquidityBot`, `AdaptiveBot`, and
`AggressiveBot`. Contributor: @ziluq1. Parameters below are the shipped
constants. The host environment does not change them.

### ValueLiquidityBot

A sealed-bid surplus bidder. A bid maximizes `P(win | b)` times surplus:
chart value of the lot, missions it completes, and a small mission-progress
credit, minus a one-coin shade on any positive bid, minus `(1 + λ)` times the
bid. `λ` is the coin shadow price and is 1.0; `δ` is 1.0. Rival bids are a
fixed Laplace-uniform mixture (mean 0.55 of a public value signal, scale 2,
weight 0.8, denial weight 0.4). A lot that only advances a mission may add at
most 1 coin, at 0.35 of the reward times the share of the requirement it
fills. After 4 rival bids, if 70% were 1 coin or less, that progress credit
turns off. The reveal is strategic and does not change the final display.

The rival mixture is not learned. A table of very cheap bidders can still
take gems this bot shades away from. Opening ties are priced as losses.
Phase lambda, covering, the loan plan, and the invest discount are off.

### AdaptiveBot

The same surplus, with `P(win | b)` taken from each opponent's recent bids.
Estimates shrink toward the ValueLiquidity prior (`k = 8`) and keep at most
24 observations. With no evidence the distribution matches that prior. Once a
rival's shrunk rate of bidding 0 is at least 0.45 after 6 auctions, a point
mass at 0 is mixed in and capped at 0.85. The shipped bot stays on the
balanced preset (lambda 1.0, delta 1.0, denial weight 0.4). Portfolio search
and bid jitter are disabled, so a fresh bot does not explore presets and does
not draw a random bid.

Because `get_bid` is replaced, the mission-progress credit and the quiet-table
gate do not run. The measured same-table gap versus ValueLiquidityBot was not
significant. The bot is still worth submitting: it leads on chart D, where the
other two do not, so the three bots are not the same strategy.

### AggressiveBot

A mission bidder with a late-game score margin, not ValueLiquidityBot's
constant liquidity premium. Treasure utility is `P(win)` times the prize,
minus `δ`, minus `(1 + λ)` times the bid. `λ` is 0.5, and 1.0 when the chart
pays more at display count 0 than at count 5 (charts B and D). Missions are
priced when a lot finishes one or moves it forward. The last three gems are
scored as expected final margin. Loans are taken only when a visible lot pays
for them. The reveal returns the first hidden gem, and the cheap-cover rule
is off.

A flat lambda of 0.5 lost on the falling charts. Replacing this bot with
ValueLiquidityBot can still score higher at 5 players or on chart B, even
when AggressiveBot outscores the ValueLiquidityBot at the same table.

### Robustness audit

This check is about crashes and illegal actions, not about who scores more.
1,500 worker games, seeds 0–99, at 3, 4, and 5 players, on charts A–E, with
`--bot-timeout 0.5`. Each of the three bots played every game. Result: 0
exceptions, 0 invalid bids, 0 invalid reveals, and 0 fallbacks. The slowest
`get_bid` was 0.0198 seconds (AggressiveBot). The slowest `on_game_start`,
including worker startup, was 0.224 seconds.

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

That suite was 97 tests, all passing, in 47.2 seconds.

### Held-out evaluation

Separate from the robustness audit. Seeds 2000–2099 (100 seeds). These are
same-table comparisons: the bots sat together, and the gap is one seat's
score minus another's on that deal. It is not a substitution test.

On `[AggressiveBot, ValueLiquidityBot, AdaptiveBot]`, with MyopicEvBot added
at 4 and 5 players:

| Bot | Mean score | Win rate |
|---|---:|---:|
| AggressiveBot | 54.32 | 39.4% |
| AdaptiveBot | 50.84 | 28.7% |
| ValueLiquidityBot | 50.28 | 27.0% |

AggressiveBot's advantage over the ValueLiquidityBot at that table was +4.04
points, 95% CI [+3.20, +4.88]. AdaptiveBot's difference from that
ValueLiquidityBot, on `[AdaptiveBot, ValueLiquidityBot, AggressiveBot]`, was
−0.69 points, 95% CI [−1.58, +0.21], which includes 0. AdaptiveBot was ahead
on chart D (+3.85) and behind on charts B (−2.79) and C (−3.89) and at 3
players (−3.49). That is not a claim that AdaptiveBot is generally stronger.

An earlier ValueLiquidity-only bench, 36 paired seeds, seats rotated, charts
A–E, is a different protocol. Against MyopicEV and FractionValue (108
three-player games) the mean was 104.4 [94.2, 114.7]. Do not compare those
means with the held-out table above. Reproduce the older bench with:

```sh
python3 experiments/bench_bots.py --seeds 36
python3 experiments/bench_bots.py --seeds 16 --players 4
python3 experiments/bench_bots.py --seeds 12 --players 5
```

`src/bot_definitions/baselines.py` holds the myopic and fraction opponents.
They are not exported from `custom_bots.py`. Local logs under `experiments/`
are gitignored. `experiments/compare.py` is a short seating comparison:

```sh
PYTHONPATH=src python3 experiments/compare.py
```

## Experiment examples

The `experiments/` directory contains script and notebook examples that reuse bot runners across
games, print each game's results, and then report total points and games won per
entrant:

- [Python script](experiments/sample_experiment.py): run
  `python3 experiments/sample_experiment.py` from the project root.
- [Notebook using existing bots](experiments/sample_experiment.ipynb): the same workflow
  split into editable cells.

Run notebook cells from top to bottom in a Python 3.10+ Jupyter environment. Each
notebook finds the project whether its kernel starts in the root or `experiments/`.
Edit the game count, seeds, timeout, and participant list to change the experiment.
Labels identify individual entrants, seats stay fixed, and each tied winner
receives one win. Runners retain memory between games and are closed when the
experiment finishes, including when an exception interrupts it.

## Layout and checks

- `src/constants.py`, `src/decks.py`, `src/lib.py`: setup rules, immutable catalogs/cards, typed decks.
- `src/player.py`: mutable player state and bot decision validation.
- `src/bots.py`: the developer-facing bot interface, snapshots, and result types.
- `src/bot_info.py`: builds each player's information without inspecting hidden deck order.
- `src/bot_runner.py`: worker lifecycle, transport, deadlines, and memory recovery.
- `src/immutable.py`: a pickleable read-only mapping used by cards and snapshots.
- `src/harness.py`: setup, round resolution, scoring, and CLI.
- `src/bot_definitions/`: custom bot implementations. The submission bots are
  `ValueLiquidityBot`, `AdaptiveBot`, and `AggressiveBot`. `SmallBidBot` remains
  the harness example. Shared pricing helpers live in `pricing.py`.
  `baselines.py` is for local comparisons only.
- `src/custom_bots.py`: imports and exposes the bot classes available to the CLI.

Tests require no third-party packages. Ruff and mypy are optional development tools:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy
```
