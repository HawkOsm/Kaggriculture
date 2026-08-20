# Kaggriculture — How to Play

Source: https://www.kaggle.com/competitions/kaggriculture/overview (fetched 2026-08-20)

## Overview

Two players compete on separate farms to see who can earn the most profit by the end of a 30-day season
(720 turns, 24 turns/day). Your agent controls a main farmer and can hire farm hands to scale up. To
succeed, your agent must:

- Plant, water, fertilize, and harvest a variety of crops.
- Buy, feed, and care for animals to produce eggs, milk, and wool.
- Collect and use fertilizer to boost crop yields.
- Buy neighboring quadrants of land to expand your farm's footprint.
- Trade on a dynamic market where prices react to sales and town demand.

**Win condition:** whoever has the most money in the bank at the end of the season wins (ties possible).
Unsold inventory does not count toward the final total.

## Timeline (as fetched)

- **2026-07-29** — Start Date
- **2026-09-23** — Entry Deadline / Team Merger Deadline
- **2026-09-30** — Final Submission Deadline
- **2026-10-01 → ~2026-10-15** — Games continue running until leaderboard convergence, then final.

## Evaluation

- Each day your team may submit up to 5 agents (bots). Only the **latest 2 submissions** are tracked/played and used for final leaderboard evaluation.
- Every bot plays Episodes against similarly-rated bots on the ladder; newer bots play more frequently.
- Only your best-scoring bot is shown on the leaderboard; track all submissions on your Submissions page.
- On upload, a **Validation Episode** runs your agent against a copy of itself. Failures are marked `Error` (download agent logs to debug); otherwise the submission gets a default rating and joins matchmaking.
- **Ranking:** Elo-like skill rating. Winning (most coins after 720 turns) raises rating, losing lowers it; magnitude depends on the rating gap. Ties pull ratings together. The coin *margin* doesn't matter — only win/loss/tie.
- **Final evaluation:** after the deadline, games keep running ~2 weeks to reduce uncertainty, then a final Bradley-Terry tournament produces the final leaderboard.

## Prizes

10 places × $5,000 = **$50,000** total.

## Object Types

| Type | Yield Type | Seed Cost | Base Market Price | Time to First Yield | Time to Max Yield | Subsequent Yields | Max Yield | Action Cost | Yield/tile/day |
|---|---|---|---|---|---|---|---|---|---|
| Wheat | One-time | 10 | 25 | 2 days | 4 days | none | 6 (4 unfertilized) | 1 | 0.80 |
| Carrot | One-time | 20 | 35 | 2 days | 3 days | none | 4 (3 unfertilized) | 1 | 0.75 |
| Tomato | Ongoing | 50 | 60 | 8 days | 11 days | every day ×4 | 4 | 1 | 0.33 |
| Strawberry | Ongoing | 100 | 120 | 10 days | 16 days | every other day ×4 | 4 | 1 | 0.24 |
| Melon | One-time | 80 | 250 | 10 days | 10 days | none | 6 | 1 | 0.55 |
| Goose/Egg | Ongoing | 300 | 50 | 4 days | NA | every day, indefinitely | 4 held | 1 + 1 (build coop) | 1.00 |
| Cow/Milk | Ongoing | 400 | 160 | 8 days | NA | every two days, indefinitely | 6 held | 1 + 1 (build pasture) | 0.50 |
| Sheep/Wool | Ongoing | 500 | 200 | 6 days | NA | every three days, indefinitely | 6 held | 1 + 1 (build pasture) | 0.33 |
| Fertilizer | NA | 100 | — | — | — | — | — | 1 | — |

Notes:
- "Yield/tile/day" for crops = total harvested ÷ days occupied, watering daily and harvesting at peak. For animals it's the steady-state production rate once the first yield lands.
- "Max Yield" for animals is `max_held`, the cap on unharvested product on the tile — not a lifetime total.
- Melon's bonus window is ages 6–12, but unfertilized it hits the cap of 6 at age 10 (ages 11–12 add nothing); fertilized it caps at age 8.
- Wheat/Carrot only reach their listed Max Yield (6/4) **with fertilizer**; watering alone peaks at 4/3.
- Tomato/Strawberry are "ongoing" but capped: 4 scheduled yields each (tomato at ages 8–11; strawberry at ages 10/12/14/16), then the plant decays into a weed.

All plants must be watered daily (weed after 2 consecutive missed days). All animals must be fed daily using wheat (escape/unrecoverable after 2 consecutive missed days).

## Actions

24 turns/day, 30 days/season = 720 total turns. One action per turn.

### Farmer / Farm Hand movement & shed
- `NORTH / SOUTH / EAST / WEST` — move one cell. Off-board moves are no-ops. Locked tiles are passable (you can walk across, but tile actions no-op there), except shed access (`PICKUP`/`DROP`/`PLACE`-into-shed) which works from any shed-adjacent tile even if locked.
- `PICKUP <item> [n]` — move up to n (default 1) of an item from the shed into inventory. Seeds are a separate slot, never picked up (PLANT consumes them directly).
- `DROP` — dump entire inventory into the shed (must be shed-adjacent). Overflow past `shedCapacity` is discarded.

### Plants
- `PLANT` — plant a purchased seed. If too many units try to plant more seeds than you have in a turn, **none** are planted.
- `WATER` — water a plant (once/day; later waterings that day are no-ops).
- `HARVEST` — gather produce. Removes the plant if it has no further yields. Adds to inventory.
- `FERTILIZE` — doubles the per-day yield bonus for the next 3 days (only applies on days the plant is also watered).

### Animals
- `PLACE <item> [n]` — standing on a matching empty structure (GOOSE→coop, SHEEP/COW→pasture) places one animal (n ignored); standing shed-adjacent, drops up to n of an item into the shed (capped by `shedCapacity`).
- `FEED` — feed with wheat (once/day).
- `HARVEST` — collect eggs/milk/wool.
- `COLLECT_FERTILIZER` — collect 1 fertilizer/day from a surviving animal (whether or not fed/cared for). Doesn't accumulate if left uncollected.
- `CARE` — once/day; banks a yield bonus (see Animal Care below).

### Animal Care
- End of day: if fed **and** cared for, `pending_care_bonus` += 1 (unfed days don't bank).
- On a scheduled production day, if fed, the entire banked bonus is added to that yield (plus base 1), then resets to 0.
- If unfed on a production day, base 1 is still produced but the bank is discarded.
- Bank is indirectly capped by the animal's `max_held`.

### Terrain / other
- `BUILD_COOP` / `BUILD_PASTURE` — add a structure to an unoccupied tile.
- `DIG` — remove a plant, weed, or empty coop/pasture (no-op if a coop/pasture has an animal on it).
- `PASS` — default no-op.

### Market actions
Up to `maxMarketOrdersPerTurn` (default 10) per turn; extras are silently dropped. Processed in order, concurrently with the opponent's orders.

- `BUY_SEED <item> <n>`
- `BUY_ANIMAL <item> <n>`
- `BUY_PRODUCT <item> <n>` — only WHEAT and FERTILIZER are buyable this way.
- `SELL <item> <n>`
- `HIRE` — hire a farm hand for the day; cost rises with each hire that day.
- `BUY_LAND` — unlock a new 5×5 quadrant. Costs: $1k, $2k, $4k.

## Watering / Feeding Details

- A new seed starts with `consecutive_unwatered = 1` (planting day counts as day 1). Left unwatered the same day, it hits 2 at end-of-day refresh and becomes a weed that night — **no grace period**.
- A newly placed animal starts with `consecutive_unfed = 0`, so it survives its first day unfed.
- Watering one-time crops during their yield bonus window increases yield; this does **not** apply to ongoing crops/animals.

## Harvest Yields

- **One-time crops** (wheat, carrot, melon): starting at half of `max_yield_day` (rounded up), watering during the bonus window adds +1 unit/day to total harvestable yield; fertilized adds +2/day instead.
- **Ongoing crops** (tomato, strawberry): fixed-interval scheduled production, base yield 1/production; if fertilized AND watered that day, yield doubles to 2.
- Once a plant hits max lifespan, remaining yield decays by 1 every other turn until 0, then it becomes a weed.
  - One-time crops reach max lifespan 1 day after `max_yield_day`.
  - Ongoing crops start decaying 1 day after cumulative production count hits `max_yield` (regardless of whether harvested).

## Map & Shed

- Farm is a `boardSize × boardSize` grid (default 10×10), split into four 5×5 quadrants; you start owning NW only and can buy the rest at escalating cost.
- Weeds can spawn on empty unlocked cells and must be cleared (`DIG`) before reuse.
- Opponents can see your farm layout but **not** your shed contents.
- **Shed**: capacity 100 items (excludes seeds); overflow at end-of-day drop (or mid-day `PLACE`) is discarded, no overflow buffer.
- The shed sits at the board center and is never a tile itself (never appears in `tiles`). "Orthogonally adjacent to the shed" = one of the four center tiles: `(half-1,half-1)`, `(half,half-1)`, `(half-1,half)`, `(half,half)` for `half = boardSize // 2` — at default size 10 that's `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`, one per quadrant. Only NW starts unlocked, so 3 of those 4 tiles start locked — but the shed remains reachable from all of them since it's never locked itself.

## Farmer / Farm Hand

- `HIRE` cost: `farmHandCostMult * fib(n)` where n = hires already made today (fib starts 1,1,2,3,5,8,13,…). Default mult 1 → costs 1,1,2,3,5,8,13,21,… resetting daily.
- A hired hand spawns shed-adjacent in a free space following NWSE order; ties broken by least-occupied, then NWSE preference. Spawn ignores lock state — since the farmer starts at (4,4), the first hire of the day goes to (5,4), locked until NE is bought (still reachable — locked tiles are passable).
- At day end, all inventory (farmer + hands) drops into the shed if room exists; anything that doesn't fit is discarded. Hands disappear at day end and must be re-hired.

## Town Buildings

- New shops unlock every `townShopUnlockInterval` days (default 3), drawn uniformly at random **with replacement** from the shop table — duplicates possible. Unlocking stops after 8 total instances; once unlocked a shop stays active for the rest of the game.
- Each unlocked shop instance consumes 1 of every product it demands every `townShopSellInterval` turns (default 4) — i.e. 6/day per demanded product per instance; single-product shops consume 2×.
- The town center consumes 1 of every product (excluding fertilizer) every `townCenterSellInterval` turns (default 24, i.e. once/day) — flat for the whole season.

| Shop Type | Increases Demand For |
|---|---|
| Bakery | eggs, wheat |
| Pizza Shop | milk, tomatoes, wheat |
| Brunch Spot | eggs, wheat, strawberries |
| Yarn Store | wool (2×) |
| Ice Cream Shop | strawberries, milk, wheat |
| Pet Cafe | carrots (2×) |
| Smoothie Shop | strawberries, milk |
| Farmers Market | wheat, carrots, tomatoes, strawberries |

## Market Mechanics

- Seeds/animals: unlimited supply, fixed prices.
- Sell prices move dynamically per resource and persist across days.
- Every product (+fertilizer) starts with market inventory `I0 = 10,000` — far above any realistic single-game production volume, so inventory stays positive.
- **Selling**: orders (any player, any quantity) are processed one unit at a time, concurrently across players (e.g. both players' `SELL CARROT 10` orders each get the current price for unit 1 simultaneously, then price may shift, repeat). If price is floored at $1, the unit still sells but isn't added back to market inventory (keeps the floor responsive).
- **Buying**: only `WHEAT` and `FERTILIZER` are buyable via `BUY_PRODUCT` (everything else sells but can't be bought back). Selling has no such restriction — everything, including animal-collected fertilizer, can be `SELL`ed. Market inventory drains via town consumption (free) and player `BUY_PRODUCT` orders (same one-unit-at-a-time concurrent procedure). If money runs out mid-order, the order stops.
- Buy price is quoted at post-buy inventory; sell price at pre-sell inventory — so an immediate buy-then-sell of the same item, market otherwise unchanged, nets exactly zero.

### Price Function

```
price(inv) = base + sign · amp · f(|inv − I0|)
  sign = +1 if inv < I0 (scarcity → price up), −1 if inv > I0 (glut → price down)
  amp  = target · base / f(T)          (derived, not stored)
  f    ∈ { linear, sq, sqrt, log, log10, hinge }   (log uses ln(1+x), so f(0)=0)
```
Floored at $1, rounded to nearest dollar.

- `hinge` depends on `T`, not just `x`: with `u = x/T`, evaluates to `u + 8·max(0, u−1)²` — linear below `T`, steep quadratic above it. `f(T)=1` by construction, so `target` keeps its usual meaning.
- `T` = production capacity of one 5×5 field over a 24-day calibration window at optimal watering, no fertilizer (animal totals pre-discounted 30% for wheat-feed overhead + 1 day to build coop/pasture). Deliberately shorter than the 30-day season because opening days are setup-heavy.
- `target` means "moving T units past I0 shifts price by target × base." Different `f`/`target` per side make similar-production resources behave very differently strategically (e.g. wheat panics on scarcity but absorbs gluts; melon barely reacts to scarcity but crashes hard on overproduction; wool mirrors melon at smaller scale). Premium resources (base > $100: strawberry, melon, milk, wool) use `above_target > 1` — even modest gluts drive them to the $1 floor.
- Carrot/tomato/egg use `hinge` on the scarcity side — prices stay near base under ordinary demand, then rise sharply past `T`.

| Resource | Base | I0 | T | Below func | Below target | Above func | Above target | P(I0−T) | P(I0+T) | P(I0+2T) |
|---|---|---|---|---|---|---|---|---|---|---|
| Wheat | 25 | 10,000 | 400 | sqrt | 0.80 | log | 0.20 | $45 | $20 | $19 |
| Carrot | 35 | 10,000 | 450 | hinge | 1.00 | sqrt | 0.70 | $70 | $10 | $1 |
| Tomato | 60 | 10,000 | 200 | hinge | 0.40 | sqrt | 0.60 | $84 | $24 | $9 |
| Strawberry | 120 | 10,000 | 100 | sqrt | 0.70 | linear | 1.60 | $204 | $1 | $1 |
| Melon | 250 | 10,000 | 300 | log | 0.20 | sq | 3.60 | $300 | $1 | $1 |
| Egg | 50 | 10,000 | 332 | hinge | 0.40 | log | 0.20 | $70 | $40 | $39 |
| Milk | 160 | 10,000 | 122 | sqrt | 0.60 | linear | 1.60 | $256 | $1 | $1 |
| Wool | 200 | 10,000 | 105 | log | 0.20 | sq | 3.20 | $240 | $1 | $1 |
| Fertilizer | 100 | 10,000 | 200 | linear | 0.40 | linear | 0.40 | $140 | $60 | $20 |

Defaults live in `MARKET_PARAMS` in `kaggriculture.py`. Per-resource overrides (sparse subset of `base`, `I0`, `T`, `below_func`, `below_target`, `above_func`, `above_target`) can be supplied at episode creation via `env.configuration["marketParams"]`, e.g. `{"WOOL": {"above_target": 0.95}}`.

## Turn Processing Order

1. Action validation
2. Player actions recorded (simultaneous)
3. Market actions processed in order, per player
4. Town buy actions (shops + town center reduce inventory)
5. Update observations
6. Day refresh (if applicable): plant/animal condition updates, fed/watered reset to false
7. Market refresh: prices update from previous turn's sells
8. Income update: bank updated from buys/sells
9. Farm update: clear harvested plants, consumed inventory items, add new plants/animals, etc.

## Observation Format

Top-level observation passed to each agent:

```json
{
  "player": 0,
  "day":    0,
  "hour":   0,
  "farms":  ["<farm 0>", "<farm 1>"],
  "market": {
    "inventory": { "WHEAT": 0, "CARROT": 0 },
    "prices":    { "WHEAT": 0, "CARROT": 0 }
  },
  "town": {
    "unlocked_shops": ["BAKERY"]
  },
  "private": {
    "shed":        { "WHEAT": 0, "GOOSE": 0, "FERTILIZER": 0 },
    "seeds":       { "WHEAT": 0, "CARROT": 0 },
    "inventories": []
  }
}
```

`private` is per-player only — you never see the opponent's shed/seeds/inventories.

Each `farm` dict (public, both players can see both):

```json
{
  "money": 0,
  "tiles": [["tile", "..."]],
  "farmer": [0, 0],
  "hands": [[0, 0]],
  "unlocked_quadrants": ["NW"],
  "hires_today": 0
}
```

A tile is one of:
- `None` — empty, unlocked
- `"LOCKED"` — quadrant not yet bought
- a plant dict: `{"kind": "PLANT", "crop": "WHEAT|CARROT|TOMATO|STRAWBERRY|MELON", "planted_day": int, "watered_today": bool, "consecutive_unwatered": int, "yield_units": int, "max_lifespan_step": int, "fertilized_until_day": int}`
- a weed dict: `{"kind": "WEED"}`
- an animal structure dict: `{"kind": "COOP"|"PASTURE", "animal": "GOOSE"|"COW"|"SHEEP"|null, "placed_day": int, "yield_units": int, "fed_today": bool, "consecutive_unfed": int, "cared_today": bool, "fertilizer_available": bool, "pending_care_bonus": int}`

## Configuration Defaults

| Parameter | Default | Description |
|---|---|---|
| `episodeSteps` | 720 | Total turns in the season (24 × 30) |
| `boardSize` | 10 | Tiles per side of each player's square farm (four 5×5 quadrants) |
| `startingMoney` | 3000 | Coins each player starts with |
| `maxMarketOrdersPerTurn` | 10 | Max market orders processed per player per turn; extras dropped |
| `turnsPerDay` | 24 | Turns per in-game day |
| `shedCapacity` | 100 | Max non-seed items the shed can hold |
| `weedSpawnChance` | 0.005 | Per-tile probability of a weed spawning on an empty unlocked tile at end-of-day refresh |
| `townShopUnlockInterval` | 3 | Days between town shop unlocks (with replacement, capped at 8 instances) |
| `townShopSellInterval` | 4 | Turns between consumption ticks per town shop instance |
| `townCenterSellInterval` | 24 | Turns between town center consumption ticks |
| `seed` | null | Optional deterministic episode seed; cleared from config after read |

Per-crop seed costs and per-product base prices are **not** configurable — see Object Types / Price Function above.

## Submission Resources & FAQ

- Submissions ≤ 100 MiB.
- Daily submission limit: 5. Only your most recent 2 are active.
- Your files land in `/kaggle_simulations/agent/` — set imports accordingly.
- Runtime: 8 GiB HDD, 6.5 GiB RAM, 1.6 vCPUs.

See the [Docker image](https://github.com/Kaggle/docker-python) for the environment/Python setup.
