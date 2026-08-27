# Config/optimization infrastructure — audit notes

Written after a request to check whether the config/search system — which grew incrementally
over one long session (13 knobs → 22 → 25 → 29+) rather than from an upfront design — has
accumulated real structural problems. It has, though nothing currently causing wrong behavior in
a running search. This is a punch list for a deliberate redesign pass, not a bug report.

## How the pieces relate today

Four places all know about "what a config knob is," maintained independently, by hand:

1. **`agent.py`'s `DEFAULT_CONFIG`** — the actual runtime default for every key. Source of truth
   for values, not for search ranges.
2. **`agent.py`'s `KNOB_SPECS`** — `(key, low, high, is_int)` tuples, consumed by the RL path
   (`decode_knobs`, `train_rl.py`) to turn a policy network's output into a config.
3. **`optimize.py`'s `sample_config(trial)`** — a hand-written function calling
   `trial.suggest_float/int/categorical(key, low, high)` once per knob, building the dict Optuna
   actually searches.
4. **`train_rl.py`'s `FEATURE_NAMES`/`extract_features`** — a separate hand-written list of what
   the RL policy can *observe* about the game state (not the same thing as what it can *set*).

None of these four derive from a shared spec. Every new knob this session required editing up to
three of them by hand (`DEFAULT_CONFIG` + `KNOB_SPECS` + `sample_config`), and nothing enforces
that the edits stay consistent with each other.

## Concrete problems found (this session, via direct audit)

- **A silently-overridden duplicate assignment.** `optimize.py` had accumulated three separate
  `DEFAULT_STUDY_NAME = "..."` lines from three rounds of "rename the study, keep the old
  reasoning as a comment" edits. The *last* one in the file wins at import time — an earlier edit
  this session left the *older* name active while a large comment block above it described (and
  looked like it set) a newer one. Fixed by commenting out the two superseded assignments, but the
  pattern that caused it (append a new rename, leave the old code + comment below) is still how
  every rename this session was done, and will do this again on the next one unless the pattern
  changes.

- **A dead search dimension.** `sell_backlog_multiple` was in `DEFAULT_CONFIG`, `KNOB_SPECS`, and
  `optimize.py`'s search space, but the code path that read it (`_market_orders`'s force-sell
  logic) was reverted earlier in the session — deliberately, with a comment explaining why, but
  the knob was left wired into both search spaces "in case a smarter version gets revisited."
  Every trial since then has spent one of its ~29 dimensions on a parameter with *zero* effect on
  behavior. (Trimmed from both `KNOB_SPECS` and `sample_config` today; left in `DEFAULT_CONFIG`.)
  **There is no automated way to catch this class of bug** — a knob silently going dead when the
  code that read it changes. It was only found by grepping for every reference by hand.

- **`sample_config`'s output config is incomplete, saved only by an accidental safety net.**
  `sample_config`/`_resolve_search_config` never set `season_days`, `wind_down_days`,
  `sell_fraction_min`, `sell_fraction_max`, `broke_phase_days_frac`, or
  `broke_phase_reserve_scale` — six keys that real runtime code (`_market_orders`,
  `_dynamic_sell_fraction`) does read. Every Optuna trial's config dict is missing them entirely.
  This doesn't crash only because `make_agent(config)` does `cfg = dict(DEFAULT_CONFIG);
  cfg.update(config)` — DEFAULT_CONFIG's own values fill the gap. That's a real, working safety
  net, but it's *implicit*: nothing documents that `sample_config` is allowed to be a partial
  config, or that this specific merge is the reason it's safe. A future refactor of `make_agent`
  that removed or reordered that merge would silently break every search.

- **The RL feature vector is stale relative to the rule-based agent's own state-awareness.**
  `FEATURE_NAMES`/`extract_features` (17 features: day/season fractions, cash/land/unit fractions
  for both sides, 7 price ratios) predates every mechanism added this session that reasons about
  *opponent* state beyond raw cash/land/units — `_opponent_profile`'s concentration/scale, and
  `_standing_asset_value`'s full mark-to-market valuation. If RL training were resumed today, the
  policy would have no way to observe the exact signal the relative-wealth mechanism reacts to,
  or the concentration/scale signal the race-discount mechanism reacts to. RL training is dormant
  this session (an earlier run this session found a trained policy decisively worse than the
  rule-based agent), so this isn't causing active harm, but it means the two optimization paths
  (Optuna over the rule-based agent, RL over a learned policy) have quietly diverged in what
  state they even have access to.

- **Bound consistency between `KNOB_SPECS` and `sample_config` is currently fine — but only by
  discipline, not enforcement.** Checked today: every shared knob's `(low, high)` matches exactly
  between the two lists, including a recent narrowing (`risk_scale_min`, 0.1–1.0 → 0.7–1.5) that
  happened to get applied to both. There is nothing that would have caught it if it hadn't been.

## The recurring root cause

Every one of the above is the same shape: **two or more places need to agree about a fact, they
were each hand-edited independently, and nothing checks that they still agree.** This project
already has one precedent for fixing exactly this class of bug well —
`build_submission.py`'s `_check_completeness`, which walks the AST of the functions being
extracted into the submission and raises before writing if any referenced module-level name isn't
in the extraction list. That exists specifically because the same silent-drift bug bit twice by
hand before the check was added (see `docs/tests/LOG.md`). Nothing equivalent exists for the
config/search-space layer.

## The other recurring manual convention: study renaming

Separately from the above, the Optuna study has been renamed five times this session
(`v2_scaling` → `v3_dynamic_sell`, abandoned → `v4_dynamic_priority` → `v5_relative_wealth` →
`v6_near_tier_weighted`), each time by hand, each time because either the search-space *shape*
changed (new/removed keys) or the objective's *meaning* changed (different opponent pool). This
is because `optuna.MedianPruner` judges each new trial's intermediate progress against the median
of *all prior trials at that step* in the same persistent study — reusing a study across either
kind of change silently handicaps every new trial regardless of its actual quality. This was
diagnosed properly (see `docs/tests/LOG.md`, "MedianPruner history contamination") and worked
around correctly every time, but it's still a manual convention resting on remembering to do it —
exactly the kind of thing that already went wrong once this session (a rename that was supposed
to happen didn't actually get committed, and a 1000-trial search ran against the stale study
before anyone noticed).

## Parameter reference

Every key currently in `agent.py`'s `DEFAULT_CONFIG` (50 total), what it actually does, and
whether it's tunable by each of the two search paths. **RL** = listed in `KNOB_SPECS`, so a
trained policy network can set it. **Optuna** = suggested in `optimize.py`'s `sample_config`, so a
search trial can set it. A "no" in both columns means the value is currently fixed — either a
deliberate safety clamp / game constant, or a knob nobody's gotten around to exposing yet (the two
read very differently; see the notes column).

Grouped to match `DEFAULT_CONFIG`'s own layout, not alphabetically.

### Selling

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `sell_fraction_base` | 0.5 | yes | yes | Baseline fraction of shed inventory sold per item per turn, before the two adjustments below. |
| `sell_fraction_day_weight` | 0.1 | yes | yes | Shifts the sell fraction by days-remaining-in-season — positive means pickier (sell less) early, less picky as the season runs out. |
| `sell_fraction_cash_weight` | 0.1 | yes | yes | Shifts the sell fraction by current cash-on-hand — positive means pickier when already cash-flush (can afford to hold out for a better price). |
| `sell_fraction_min` | 0.15 | no | no | Hard floor on the computed sell fraction — safety clamp, not searched, so a bad coefficient combo can't produce a negative threshold. |
| `sell_fraction_max` | 0.9 | no | no | Hard ceiling on the computed sell fraction — same kind of safety clamp. |
| `cash_scale` | 1500 | no | yes | Reference cash level the "how flush are we" feature is normalized against inside the sell-fraction formula. |
| `max_sell_chunk` | 10 | yes | yes | Cap on how many units of one item get sold in a single turn, so one order can't crash its own price mid-sale. |
| `sell_backlog_multiple` | 2.5 | no | no | **Dead.** Was meant to force-sell once shed backlog crossed a threshold; that logic was tried and reverted (permanently realizes losses on crashed prices). Left in `DEFAULT_CONFIG` only in case a smarter version gets revisited — genuinely unused by any code path today. |

### Cash reserves / spending gates

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `money_reserve` | 100 | yes | yes | Flat dollar floor the agent won't spend below (except cheap seed top-ups) — the general "don't go broke" buffer for land/animal purchases. |
| `broke_phase_days_frac` | 0.0 | no | no | Fraction of the season (from day 0) during which every reserve gate gets scaled down by `broke_phase_reserve_scale`. `0.0` = the phase never triggers, i.e. this mechanic is off. |
| `broke_phase_reserve_scale` | 1.0 | no | no | How much to shrink reserves during the broke phase. `1.0` = no shrinkage even if the phase were active. Together with the key above, a real but currently-disabled "spend into the red early" mechanic — a flat always-on version of this was tried and regressed (see `docs/tests/IDEAS_TRIED.md`). |
| `seed_money_floor` | 20 | no | yes | Minimum cash allowed when buying a seed — seeds are cheap, so this floor sits close to (or inside) `money_reserve` on purpose. |
| `hire_money_floor` | 10 | no | yes | Minimum cash allowed when hiring — deliberately its own small floor instead of sharing `money_reserve`, so a temporary cash dip doesn't block the hiring that would fix low income. |
| `hire_reserve_multiple` | 2.0 | yes | yes | Extra safety multiple on hire cost required before hiring (on top of `hire_money_floor`). |
| `animal_reserve_multiple` | 2.5 | yes | yes | Extra safety multiple on animal cost required before `BUY_ANIMAL` — animals are $400-500 vs a hire's few dollars, so this gets a stronger brake than hiring's. |

### Hiring / labor

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `max_hires_per_day` | 5 | yes | yes | Hard cap on hires per day. Raising it alone (e.g. to 12) has repeatedly regressed — the current dispatcher can't keep that many hands productively busy (see `IDEAS_TRIED.md`, "Dispatch / routing efficiency"). |
| `hire_backlog_ratio` | 1.5 | yes | yes | Hires only when pending tasks (harvest/water/feed/weeds/fertilize/empty tiles) per current unit exceed this ratio — this is what actually produces the observed hiring ramp over the first several days. |

### Land / structures / animals

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `startup_days` | 2 | no | yes | Days before general spending (beyond survival) starts. |
| `land_startup_days` | 4 | no | yes | Days before land purchases specifically start — delayed separately so early hiring/seed spending doesn't compete with land for the same thin starting cash. |
| `land_utilization_threshold` | 0.75 | yes | yes | Occupancy fraction of the current footprint required before buying the next land quadrant — prevents unlocking land the crew can't reach yet. |
| `animal_enabled` | True | no | yes | Master toggle: buy/place animals at all. |
| `enable_coop` | False | no | yes | Whether to use COOP/GOOSE in addition to PASTURE/COW/SHEEP — off by default; traced strong opponents run pure pasture. |
| `max_structures` | 16 | no | yes | Ceiling on total pasture+coop tiles. **Currently the live champion (`best_config.json`) has this at 4** — see the target-structures bug in `docs/tests/LOG.md` (fixed in code, but the searched *value* is still small; isolated testing found raising it alone regresses, see `IDEAS_TRIED.md`). |
| `pasture_target_ratio` | 0.3 | yes | yes | Target pasture count as a multiple of current unit count (farmer + hands), capped by `max_structures` — building itself is free, only the animal costs money. |
| `buy_fertilizer` | False | no | yes | Whether to proactively buy fertilizer — usually not worth it once any animals are running and producing it for free. |

### Crops

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `crops` | all 5 crop types | no | no (indirect) | Which crops the agent is willing to grow at all. Not searched directly — `optimize.py` instead searches a `crop_profile` categorical (`diversified`/`fast_cash`/`melon_plus_short`) that gets converted into this list after sampling, so `crops` itself never appears as a raw Optuna dimension. |
| `diversification_weight` | 0.15 | yes | yes | Score penalty per tile already growing a given crop, so a turn's plantings spread across crops instead of piling into whichever currently scores highest. |

### Season timing

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `season_days` | 30 | no | no | Total game length in days — a game constant, not a strategy knob. |
| `wind_down_days` | 3 | no | no | In the final N days: stop hiring/land/animal spending, sell shed inventory regardless of price. Also a fixed constant, not searched. |

### Opponent-adaptive (race discount)

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `opponent_awareness_enabled` | True | no | no (hardcoded True) | Master toggle for reacting to the opponent's incoming supply at all. Deliberately hardcoded `True` in `optimize.py` — a past noisy search disabled a strictly-useful feature by mistake (see `LOG.md`). |
| `opponent_incoming_threshold` | 3 | no | yes | How many ripe/near-ripe units of an item the opponent needs (visible from their public tiles) before we treat it as "incoming supply" worth reacting to. |
| `opponent_race_discount` | 0.7 | no | yes | Base multiplier applied to our own sell threshold when the opponent's about to dump supply — sell into the still-healthy price first. |
| `opponent_lookahead_days` | 2 | no | yes | How many days ahead to scan the opponent's tiles for crops that will ripen soon (not just already-ripe ones). |
| `opponent_concentration_sensitivity` | 0.3 | yes | yes | Scales `opponent_race_discount` further by the opponent's *concentration* (their stake in that specific item) × *scale* (their overall land/labor size) — makes the discount opponent-adaptive instead of one fixed number for every opponent. |

### Relative-wealth risk scale (CPPI / tournament theory)

| Key | Default | RL | Optuna | What it does |
|---|---|---|---|---|
| `relative_wealth_enabled` | True | no | no (hardcoded True) | Master toggle for the whole mechanism. Hardcoded on for the same reason as `opponent_awareness_enabled` — the numeric knobs below already reduce to a no-op, no reason to let a noisy trial disable the mechanism entirely. |
| `wealth_margin_scale` | 3000.0 | yes | yes | Dollar scale that normalizes (our total value − opponent's total value) into a fraction before `risk_sensitivity` is applied. |
| `risk_sensitivity` | 0.0 | yes | yes | How strongly that normalized margin shifts the risk multiplier away from 1.0. **`0.0` = structural no-op** — see `LOG.md`, the mechanism's promising-looking 4-episode result evaporated at 10 episodes, so this isn't a hand-picked value. |
| `risk_scale_min` | 1.0 | yes | yes | Floor on the risk multiplier — `1.0` means reserves never loosen below normal even when behind (the theory-symmetric `<1.0` version tested strictly worse; see `IDEAS_TRIED.md`). |
| `risk_scale_max` | 2.0 | yes | yes | Ceiling on the risk multiplier — how much reserves can tighten up when comfortably ahead. |

### Dispatch priority weights

11 keys, one per `_plan_units` task tier, all `yes`/`yes` (RL + Optuna searchable), all defaulting
to values that reproduce the exact hand-tuned execution order that existed before this became
tunable (higher number = earlier in the turn's priority order): `priority_weight_feed` (100),
`priority_weight_care` (95), `priority_weight_harvest` (90), `priority_weight_fertilize` (85),
`priority_weight_collect_fertilizer` (80), `priority_weight_water` (75),
`priority_weight_empty_coop_place` (50), `priority_weight_empty_pasture_place` (45),
`priority_weight_weeds` (20), `priority_weight_empty_build` (15), `priority_weight_empty_plant`
(10). Shed-pickup tiers and the final drop/pass fallback aren't part of this list — they're
per-unit prerequisites, not tasks competing for a turn's labor the same way these 11 are.

### A pattern worth noting from building this table

**RL=no, Optuna=yes shows up on 14 different keys** (`cash_scale`, `seed_money_floor`,
`hire_money_floor`, `animal_enabled`, `enable_coop`, `max_structures`, `startup_days`,
`land_startup_days`, `buy_fertilizer`, `opponent_incoming_threshold`, `opponent_race_discount`,
`opponent_lookahead_days`, plus the two hardcoded-true toggles). That means even if RL training
were resumed today, the policy network would be structurally unable to ever set roughly a third of
what Optuna can search — not because those knobs don't matter for RL, but because `KNOB_SPECS`
was never extended to include them. This is a second, more precise version of the "RL feature
vector is stale" finding above: it's not just that RL can't *observe* the new state (concentration/
scale/asset-value), it's that RL's *action space* has quietly fallen behind Optuna's on knobs from
well before this session's newest mechanisms. Same root cause, same fix candidates.

## Directions worth considering (not decided, not implemented)

- **Single declarative spec.** One list of `(key, default, low, high, type, searchable_by_optuna,
  searchable_by_rl_policy, observable_by_rl_features)` tuples that `DEFAULT_CONFIG`, `KNOB_SPECS`,
  and `sample_config` are all *generated from*, instead of independently hand-maintained. Biggest
  structural fix available; also the biggest rewrite.
- **A completeness check for the config layer**, modeled on `build_submission.py`'s
  `_check_completeness`: at minimum, assert every `DEFAULT_CONFIG` key is either read somewhere in
  `agent.py`'s actual logic or explicitly whitelisted as intentionally-inert (catches the
  `sell_backlog_multiple` class of bug automatically instead of by manual grep).
- **Auto-derived study names.** Hash `sample_config`'s key set + bounds + the resolved opponent
  list into the study name automatically, so a search-space or opponent-pool change *can't* run
  against a stale, incompatible study by accident — removes the manual-renaming convention
  entirely instead of relying on remembering to follow it.
- **Either retire or repair the RL path's feature parity.** Decide whether RL training is still a
  live direction this project wants; if yes, `FEATURE_NAMES` needs the concentration/scale/
  asset-value signals added. If RL is effectively shelved, it's worth saying so explicitly rather
  than letting it silently rot further with every new rule-based mechanism.
- **Make `sample_config`'s partial-config contract explicit**, or make it complete. Either
  document (or assert) that `sample_config` is allowed to omit keys because `make_agent` merges
  over `DEFAULT_CONFIG`, or have `sample_config` build off `dict(DEFAULT_CONFIG)` itself so the
  full config is always self-contained and the merge isn't load-bearing by accident.

## What's already been fixed vs. what's just documented here

Fixed so far (small, low-risk, already tested): two separate duplicate `DEFAULT_STUDY_NAME` lines
(one caught same-day, one caught a session later after it had silently run a full search under
the stale name), and trimming the genuinely-dead `sell_backlog_multiple` knob out of both search
spaces (this one took two passes too — the first edit to `agent.py`'s `KNOB_SPECS` landed, but the
matching edit to `optimize.py`'s `sample_config` didn't, and went unnoticed for an entire
intervening search). That second miss is itself worth taking seriously as evidence for the
"nothing checks these stay in sync" argument above — the fix was decided, described, and *believed
done* the first time, and it still drifted. Everything else in this document is audit findings
only — no other design decisions made.
