# Kaggriculture: Getting Started (Tutorial Walkthrough)

Source: [`bovard/kaggriculture-getting-started`](https://www.kaggle.com/code/bovard/kaggriculture-getting-started)
by Bovard Doerschuk-Tiberi & Domino Weir (Kaggle staff), fetched 2026-08-20.

Farm your way to market dominance! Players harvest produce and animal products to sell in a dynamic
market.

## Game Mechanics (quick recap)

- Farmer: one action per turn, 30-day season, 24 turns/day (720 total).
- Crops and animals each have their own seed cost, time to first yield, and total payout.
- Daily care: plants need watering every day or they turn to weeds; animals need feeding or they escape.
- Market prices move with supply — selling a product pushes its price down, and crops vary in how hard
  they crash from a glut.
- Town shops unlock over the season and steadily buy products, lifting prices over time.
- Farm hands can be hired for the day, with increasing costs for each hire per day.
- Farm expansion: start with one quadrant of land and can buy the other three for an escalating fee.
- Shed: holds harvested goods but caps at 100 items — anything past that is discarded at end of day.
- Win condition: whoever has the most money in the bank at the end of the season wins.

Full reference: [`../GAME_GUIDE.md`](../GAME_GUIDE.md).

## 1. Install & sanity-check the environment

```bash
pip install --upgrade "kaggle-environments>=1.32.2"
```

```python
from kaggle_environments import make

env = make("kaggriculture", debug=True)
print(f"Environment: {env.name} v{env.version}")
print(f"Players: {env.specification.agents}")
print(f"Max steps: {env.configuration.episodeSteps}")
```

```
Environment: kaggriculture v0.1.0
Players: [2]
Max steps: 720
```

## 2. Understand the observation

Each turn your agent receives an observation with:

- `player` — your player id (0 or 1).
- `day` / `hour` — current in-game day (0-indexed) and turn within the day (0-indexed, `turnsPerDay` per day).
- `farms` — both players' **public** farm state, indexed by player id:
  - `money` — current bank balance
  - `tiles` — `boardSize × boardSize` grid indexed `tiles[y][x]`: `None` (empty), `"LOCKED"` (unowned quadrant), or a dict for `PLANT`/`WEED`/`COOP`/`PASTURE`
  - `farmer` — `[x, y]` of your main farmer
  - `hands` — `[x, y]` positions of any hired hands active today
  - `unlocked_quadrants` — subset of `["NW", "NE", "SW", "SE"]`
  - `hires_today` — number of hires already made today (drives the next `HIRE` price)
- `market` (shared) — `inventory` and `prices` per product (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER).
- `town` (shared) — `unlocked_shops`, the shops currently generating demand.
- `private` — your **own hidden** state (opponent cannot see it):
  - `shed` — counts of every product/animal stored
  - `seeds` — seed counts per crop
  - `inventories` — per-unit carried inventory; `[0]` is your main farmer, `[1..]` are today's hired hands in order

Tile dict shapes:
- Plant: `{kind: "PLANT", crop, planted_day, watered_today, consecutive_unwatered, yield_units, max_lifespan_step, fertilized_until_day}` — `consecutive_unwatered >= 2` turns the tile to a weed at end of day.
- Weed: `{kind: "WEED"}` — must `DIG` before reuse.
- Coop/Pasture (empty): `{kind: "COOP" | "PASTURE"}`.
- Coop/Pasture (occupied): adds `animal, placed_day, yield_units, fed_today, consecutive_unfed, cared_today, fertilizer_available, pending_care_bonus`. `consecutive_unfed >= 2` means the animal escapes.

Your agent returns a dict of actions: `{"farmer": [...], "hands": [...], "market": [...]}`.

```python
# Run a quick game to see what the observation looks like
env = make("kaggriculture", debug=True)
env.run(["random", "random"])

obs = env.steps[1][0].observation  # step 1 = first action step
items = obs.market.prices.keys()
print(f"Player: {obs.player}")
print(f"Player {obs.player}'s Unlocked Farm Areas: {obs.farms[obs.player].unlocked_quadrants}")
for i in items:
    print(f"{i} Price: {obs.market.prices[i]}")
```

```
Player: 0
Player 0's Unlocked Farm Areas: ['NW']
WHEAT Price: 26
CARROT Price: 36
TOMATO Price: 60
STRAWBERRY Price: 128
MELON Price: 256
EGG Price: 50
MILK Price: 169
WOOL Price: 206
FERTILIZER Price: 100
```

## 3. Agent 1: Melon Maxxer

Straightforward strategy:

1. Whenever it runs out of melon seeds and has the cash, buy one more.
2. Walk to the nearest open tile and plant a melon; if already standing on a melon plant, water it (or
   harvest it once fully grown).
3. Once melons pile up in the shed, only sell them if the market price is above a threshold — otherwise
   hold and wait for a better price.
4. Roll the proceeds back into more seeds and repeat.

This demonstrates the core loop: reading observations, maintaining the farm, and watching the market.
Full code: [`../../src/melon_maxxer.py`](../../src/melon_maxxer.py).

```python
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

MELON_SEED_COST = CROPS["MELON"]["seed"]
MELON_MAX_YIELD_DAY = CROPS["MELON"]["max_yield_day"]
SELL_THRESHOLD = 200

# ... see melon_maxxer.py for _step_toward, _find_target_tile, melon_maxxer ...
```

Test it against the random agent:

```python
env = make("kaggriculture", debug=True)
env.run([melon_maxxer, "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

env.render(mode="ipython", width=800, height=600)
```

```
Player 0: reward=6099.0, status=DONE
Player 1: reward=0.0, status=DONE
```

## 4. What's wrong with this agent?

- It never hires farm hands or buys more land, so it's stuck with one farmer working a single quadrant.
- It only grows melons, so when other crops are fetching a higher price it has nothing to sell.
- It never fertilizes its plants, leaving a yield bonus on the table.
- When it does sell, it dumps the entire inventory in one order, which can crash the melon price below
  the threshold partway through the sale.

Now it's your turn to make improvements!

## 5. Making a submission

You can submit a `main.py`, a `tar.gz`/`zip` containing a `main.py`, or a notebook with a `main.py` or
`submission.tar.gz`.

Three ways to submit:

1. The **Submit Agent** button on the competition homepage, uploading the file.
2. The **Kaggle CLI** — see [`README.md`](README.md) in this directory.
3. Submitting a **notebook** with a `submission.py` or `submission.tar.gz` output (e.g. via `%%writefile
   submission.py` in the last cell, then clicking "Submit to competition" in the notebook viewer).

```python
%%writefile submission.py
# ... paste your final agent code here, e.g. the contents of melon_maxxer.py ...
```

Once you have a working `main.py`, submit it and watch your entry show up on the competition leaderboard.
