# Credits

Third-party code used in this repo, and exactly how it's used. None of this is submitted
to the competition, in whole or in part — everything here lives in `src/opponents/` as
local benchmark/sparring opponents only. Our actual submission (`submission/main.py`) is
built exclusively from `src/agent.py` + `src/farm_utils.py` + `src/best_config.json`
(see `src/build_submission.py`).

## Official starter kit

- **`src/farm_utils.py`**'s movement helpers, and the original agent this repo evolved
  from, are based on the official Kaggle-provided starter notebook: Bovard
  Doerschuk-Tiberi, ["Kaggriculture: Getting
  Started"](https://www.kaggle.com/code/bovard/kaggriculture-getting-started).
- **`src/opponents/melon_maxxer.py`, `multi_crop.py`, `quick_start_agent.py`** (removed
  2026-08-21, see `docs/tests/LOG.md`) were verbatim/lightly-adapted copies of starter-kit
  reference agents. Replaced once real, much stronger opponents (below) were available —
  they were giving false confidence (decisive wins against them meant little once
  compared against actual 2500+ leaderboard-strength play).

## Pulled opponent agents (`src/opponents/`)

All nine below were pulled via the Kaggle API (`kaggle kernels pull`) from public
notebooks on the competition's [Code
tab](https://www.kaggle.com/competitions/kaggriculture/code) — the first five on
2026-08-21, the remaining four (`pilkwang`, `romanrozen`, `prvsiyan_frontier`,
`tran_hh`) added the same day after a parallel search — after confirming (directly,
by name-matching the public leaderboard, or via `kaggle kernels status` on the
notebook slug) that their strategy scores well above 2500. See `docs/tests/LOG.md`
for the selection process, including why `salemali7`'s and `bruceqdu`'s notebooks
(first batch) and `denizeryilmaz`'s `v111-8c4s-economic-core-premium-lead` (second
batch) were *excluded* as near-duplicates of agents already in the pool.

Each file's own docstring carries this same attribution; this table is just the summary.

| File | Source notebook | Author | Underlying strategy credited to |
|---|---|---|---|
| `kawa_route_agent.py` | [V20-Adaptive-R1 \| Multi-Route Agent](https://www.kaggle.com/code/boatlee/v20-adaptive-r1-multi-route-agent) | boatlee | "Kawa" (reconstruction from 12 public replays, per the notebook's own docstring) |
| `boatlee_v16_agent.py` | [V16-RC5 \| High-Score 8C/4S Premium Market Lead](https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead) | boatlee | No player-reconstruction claim in the source notebook — appears to be boatlee's own front-running/market-timing strategy, distinct in mechanism from V20's multi-route classifier despite sharing some low-level utility code |
| `rayk_c95_agent.py` | [Kaggriculture: Findings from Zero to Top Meta](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta) | Rayk Kretzschmar | "Lev Neganov episode 91587143 player 1" (distilled trajectory, per the notebook), plus the author's own added controller logic |
| `saiteja_agent.py` | [Kaggriculture \| Pure Architecture (2600+ Elo) V3](https://www.kaggle.com/code/saitejabandaruin/kaggriculture-pure-architecture-2600-elo-v3) | Sai Teja Bandaru | "Public automatylicza schedule replica, reconstructed from 48 public games" (per the notebook's own docstring) |
| `kaito_agent.py` | [25/27 Strict-Future \| v27 Midgame Meta Reset](https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset) | Kaito Fukami | "team Ezzzzzekki, submission 55390428, episode 91493566, seat 0" (per the notebook's own machine-readable attribution card) |
| `pilkwang_agent.py` | [Kaggriculture: Structured Economic Policy](https://www.kaggle.com/code/pilkwang/kaggriculture-structured-economic-policy) | pilkwang | pilkwang's own deterministic scenario-aware economic policy |
| `romanrozen_agent.py` | [Strong Barnyard Economist](https://www.kaggle.com/code/romanrozen/strong-barnyard-economist) | romanrozen | romanrozen's own adaptive replay controller (season route + near-clone premium preemption) |
| `prvsiyan_frontier_agent.py` | Composite of three notebooks: [Kaggriculture Frontier \| The Soil Remembers Rain](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain), [Kaggriculture Frontier \| The Moon Counts Melons](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons), [Kaggle Frontier Lab \| Strategy Improvement](https://www.kaggle.com/code/prvsiyan/kaggle-frontier-lab-strategy-improvement) | prvsiyan | prvsiyan's "Frontier" strategy family -- the shipped agent composes the soil/route module, the moon/terminal-liquidation module, and the strategy-improvement module from these three notebooks (see the `_RC5_NS`/`_MOON_TERMINAL_NS`/`_MODAL_NS` namespaces in the file) |
| `tran_hh_agent.py` | No source notebook -- direct reconstruction from public replay [episode 89674601](https://www.kaggle.com/competitions/kaggriculture/leaderboard), seat 0 | n/a (replay-derived, not a notebook) | Tran H Hoang (per the replay's own player attribution) |

## Near-tier opponent agents (`src/opponents/`)

Unlike the nine above (all 2500+ Elo, pulled to represent aspirational strength), these
two were deliberately pulled from the **middle of the live leaderboard** (~rank
4000-4500 of 5684 at pull time, score 430-510) to match our own current live rating
(~467) -- see `docs/tests/LOG.md` for why: the live ladder matches submissions against
opponents near their own rating, not the strongest bots, so a benchmark pool that's
uniformly 2500+ tells us nothing about whether we're winning the games that actually
move our rating right now.

| File | Source notebook | Author | Leaderboard rank/score at pull (2026-08-21) |
|---|---|---|---|
| `rajan1673_agent.py` | [kagriculture](https://www.kaggle.com/code/rajan1673/kagriculture) | rajan jha | #4478/5684, 434.7 |
| `chaitanyajamble_agent.py` | [Kaggriculture](https://www.kaggle.com/code/chaitanyajamble/kaggriculture) | Chaitanya Jamble | #4058/5684, 507.2 |

Both extracted from a `%%writefile` cell in their respective notebooks (standard
pattern for Kaggle notebook agents), verbatim aside from adding the file's attribution
docstring and the `<name>_agent = agent` export alias. Checked against the existing
nine for duplication (AST def-name-set diffing, same methodology as the 2500+ pool) --
zero meaningful overlap with any of them.

### Why these two specifically

Chose the closest score matches to our own live rating available with a real
`%%writefile`-style submittable agent (not a starter-kit copy or a pure analysis
notebook -- e.g. `nagatakengo/kaggriculture-movements-top-xx`, initially considered for
its near-identical score of 474.5, turned out to be a replay-analysis notebook with no
`agent(obs)` function at all, not usable as an opponent). Both decisively beat `random`
(28k/43k vs 0, weaker than the 2500+ pool's 77k-174k, consistent with their tier) and
gave a genuinely informative result against our current submission: **4W-0L vs
`rajan1673_agent`** (avg margin +25,046), **0W-4L vs `chaitanyajamble_agent`** but by a
much smaller margin (avg -7,548) than anything against the 2500+ pool (-100k to -135k).

### Why these nine specifically

Cross-checked notebook authors against the public leaderboard where possible (e.g. "Rayk
Kretzschmar" = leaderboard rank #23 at 2755.8; "Bruce" [bruceqdu, excluded] = rank #36 at
2713.8) and otherwise relied on explicit score claims in the notebook title/description
(e.g. "2600+ Elo"). All nine, run locally, decisively beat `random` (77k-174k vs 0). Head
to head against `kawa_route_agent`: `boatlee_v16`/`rayk_c95`/`saiteja`/`kaito` all lose by
10-15k (close, not blowouts); `prvsiyan_frontier` is essentially even (121,463 vs
122,988); `romanrozen` and `tran_hh` are solidly competitive (86,328 vs 108,254 and
53,938 vs 83,188); `pilkwang` is the weakest of the nine but still clearly above starter-
kit level (35,549 vs 68,311) — real, distinct strength across the pool, not a
rubber-stamp set.

### Why `salemali7`, `bruceqdu`, and `denizeryilmaz` were pulled but not kept

- `salemali7`'s "3094-score-kaggriculture" notebook shares the *exact* docstring, internal
  codename ("BL-MDgogo-10C4S-R0"), and ~90% of its helper-function names with
  `kawa_route_agent.py` — a republish of the same underlying "Kawa" reconstruction, not a
  distinct strategy.
- `bruceqdu`'s "My 2026-08-04 High-Score Pipeline" notebook is explicitly another
  reconstruction of "automatylicza" (single fixed episode replay), the same source player
  `saiteja_agent.py` already reconstructs (from 48 games, a broader synthesis) — kept
  `saiteja_agent.py` instead as the more general version.
- `denizeryilmaz`'s `v111-8c4s-economic-core-premium-lead` notebook is a relabel of
  `boatlee_v16_agent.py`'s source ("V16-RC5 | High-Score 8C/4S Premium Market Lead"): 100%
  function-name overlap (16/16), same underlying strategy, attribution stripped in the
  version initially pulled in as `deniz_v111_agent.py` — deleted once traced back.

## Usage notes

- These agents are used exclusively as local benchmark/sparring opponents: for
  `src/run_match.py` one-off verification matches, and in `src/optimize.py` /
  `src/train_rl.py`'s opponent pools during local tuning and RL training.
- None of this code, in whole or part, is used in or derived into
  `submission/main.py`. Several of the source notebooks explicitly describe themselves as
  reconstructions of a *third* party's play (not the notebook author's own hidden
  source) — reusing that as our own submission would misattribute someone else's
  strategy (and, per the reconstructing author, possibly the strategy of a person who
  never published it themselves).
