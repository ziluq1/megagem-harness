from random import Random
from itertools import permutations
from bot_runner import BotRunner
from harness import play_game
from custom_bots import ValueLiquidityBot, SmallBidBot
from bot_definitions.baselines import MyopicEvBot, FractionValueBot

GAMES_PER_SEATING = 20           # raise for tighter results
TABLE = [ValueLiquidityBot, MyopicEvBot, MyopicEvBot]   # change per table

def main():
    totals = {}
    for seating in set(permutations(range(len(TABLE)))):
        classes = [TABLE[i] for i in seating]
        labels = [f"{TABLE[i].__name__}#{i}" for i in seating]
        runners = [BotRunner(c) for c in classes]
        try:
            for seed in range(GAMES_PER_SEATING):
                res = play_game(len(classes), bots=runners, rng=Random(seed))
                for p, label in enumerate(labels):
                    t = totals.setdefault(label, [0, 0, 0])
                    t[0] += res.scores[p]
                    t[1] += p in res.winner_ids
                    t[2] += 1
        finally:
            for r in runners:
                r.close()
    for label, (s, w, n) in sorted(totals.items()):
        print(f"{label:28s} mean score {s/n:6.1f}   win rate {w/n:5.1%}   ({n} games)")

if __name__ == "__main__":
    main()
