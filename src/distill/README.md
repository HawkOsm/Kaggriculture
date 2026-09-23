# src/distill — Replay-Distillation Agent

The top-of-ladder Kaggriculture agents (prvsiyan, boatlee, kaito, …) are not clever
runtime algorithms — they are **literal fixed action tapes** distilled from strong
public game replays, wrapped in thin routing/repair. This package reimplements that
methodology (legal per competition RULES §2.4/2.11/2.10 — public replays are usable,
a tape baked into the submission is fine; no runtime data fetch).

## What it does

1. **Record** the strong local opponent agents winning across seeds → a **tape
   library** (each tape = a 720-turn action sequence).
2. **Replay** one strong tape whole-game. The engine no-ops any position-mismatched
   field action, while the position-independent economic schedule (hire/buy/sell,
   herd, crop plan) carries it. This out-produces our reactive champion ~6–7×.
3. (Optional) **Route** among several tapes for adaptivity.

## Files

| file | role |
|------|------|
| `replay_lib.py` | parse a kaggle_environments episode replay; bit-exact reconstruction (action[t] = steps[t+1][seat].action) |
| `build_library.py` | record source agents winning vs a pool across seeds → `library.json` |
| `tape_agent.py` | **the deliverable core**: replay a single distilled tape whole-game |
| `router_agent.py` | multi-tape profile router (8-turn lock) — the adaptivity enhancement |
| `package.py` | bake a chosen tape into a self-contained `submission/main_distill.py` |
| `build_router_v2.py` | prefix-aligned shop router → `submission/main_router_v2.py` (best fully-own agent) |
| `build_hanif_plus.py` | hanif's public agent + 4 measured patches → `submission/main_hanif_plus.py` (see CREDITS.md) |
| `build_router.py`, `tape_repair.py`, `improve_hanif.py`, `improve_base.py` | documented negatives (router v1, tape repair, front-run layer, Roxy tape swap); outputs removed from `submission/` |

Large data (`library.json`, ~67MB) stays under `.claude/scratch/distill/` — code lives here.

## Verified results (single best tape = stevenleehans #113, vs 4-opp pool, money-compare)

| seed set | win-rate | note |
|----------|----------|------|
| 2000–2095 | **75/96 = 0.781** | champion 11/96 = 0.115 |
| 4000–4095 | 70/96 = 0.729 | not seed-luck |
| unseen opponents | 58/72 = 0.806 | never in library |

Per-opponent (2000–2095): pilkwang 24/24, cropdusta 19/24, **prvsiyan 15/24, boatlee
17/24** (the walls the reactive champion never touched). Only loss: kawa_route (0/12,
a ~3k-margin fixed-vs-fixed matchup a router could flip).

**Caveat:** a single static tape has no adaptivity — dominant vs a mostly-static pool,
but the true ladder rank (thousands of unseen agents) is only known once submitted.
