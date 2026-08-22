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

## 2026-08-22 additions (`src/opponents/`)

Four more opponents pulled the same way, to broaden the pool beyond the original
9 (2500+ tier) + 2 (near-tier) split -- two more near-tier agents plus two
agents in between near-tier and the 2500+ pool. **Not yet wired into
`src/optimize.py`'s `OPPONENT_REGISTRY`/default pool or `src/train_rl.py`'s
`OPPONENT_POOL`/`OPPONENT_WEIGHTS`** -- landed as files + attribution only,
pending a deliberate decision on how to weight them in.

| File | Source notebook | Author | Leaderboard rank/score at pull (2026-08-22) | Tier |
|---|---|---|---|---|
| `ektarr_agent.py` | [Diversified Scheduler Baseline \| Kaggriculture](https://www.kaggle.com/code/ektarr/diversified-scheduler-baseline-kaggriculture) | Maxim | #3831/5684, 554.5 | near-tier |
| `nagatakengo_agent.py` | [Kaggriculture](https://www.kaggle.com/code/nagatakengo/kaggriculture) | nk | #4567/5684, 428.5 | near-tier |
| `premaananda108_agent.py` | [Economics-Driven Rule Agent (EcoBot v2)](https://www.kaggle.com/code/premaananda108/economics-driven-rule-agent-ecobot-v2) | Prema Ananda | #2604/5684, 850.4 | mid-tier |
| `sakhawathossen_agent.py` | [Kaggriculture Final Hybrid Champion](https://www.kaggle.com/code/sakhawathossen/kaggriculture-final-hybrid-champion) | Sakhawat Hossen | #1421/5684, 1601.9 | strong |

`ektarr_agent.py` and `premaananda108_agent.py` were extracted from real
`%%writefile main.py` / triple-quoted-source-string cells (the two standard
patterns already used elsewhere in this pool), verbatim aside from adding the
file's attribution docstring and the `<name>_agent = agent` export alias.
`nagatakengo_agent.py` is the same, except its original `def agent(obs,
config):` had no default for `config` -- changed to `config=None` for
consistency with the rest of the pool (e.g. `pilkwang_agent.py`,
`saiteja_agent.py`); no other behavior changed. `sakhawathossen_agent.py` was
extracted from a zlib+base85-compressed embedded string; it is internally
titled "C166 anti-H4 meta counter" in the source notebook and is, like
`tran_hh_agent.py`, a replay-derived fixed-action-tape agent (a large
precomputed per-step action trace) with front-running/meta-counter logic
layered on top -- not a from-scratch rule engine like the other three in this
batch.

All four checked against the full existing pool (this batch plus the original
11) via AST def-name-set diffing (Jaccard similarity of top-level function
names) -- zero meaningful overlap for any of the four kept. Each also beats
`random` locally, scaling with tier: `nagatakengo_agent` 27,786, `ektarr_agent`
37,763, `premaananda108_agent` 140,942, `sakhawathossen_agent` 169,790 (vs 0).
Against our current submission (`submission/main.py`), one seed each:
`ektarr_agent` is close (23,784 vs 25,177 -- essentially a coin flip, a good
near-tier signal), `nagatakengo_agent` loses to us (34,906 vs 14,649),
`premaananda108_agent` beats us clearly (33,167 vs 108,955), and
`sakhawathossen_agent` beats us decisively (22,060 vs 124,052) -- consistent
with their leaderboard tiers.

Note: `nagatakengo/kaggriculture` (this batch, real `agent(obs, config)`
function, score 428.5) is a **different notebook** from the already-excluded
`nagatakengo/kaggriculture-movements-top-xx` (score 474.5, no agent function
at all, see "Why these two specifically" above) -- same author, two separate
notebooks, only one of which is a usable submittable agent.

### Candidates pulled but rejected this round

Duplicate/near-fork (AST def-name-set Jaccard > 0.7 against an existing pool
file, per this project's duplicate-detection threshold):

- `tetsutani/adaptive-farming-strategy-for-kaggriculture` (130 votes, claimed
  2099.3) -- 100% function-name overlap with `kawa_route_agent.py`; same
  embedded "Kawa" reconstruction source, just a fancier presentation notebook
  around it.
- `romantamrazov/kaggriculture-hamburger` (128 votes, claimed 2348.7) -- 100%
  overlap with `tran_hh_agent.py` (`_TRAN_BASE_AGENT`, `_TRAN_TERMINAL_AGENT`,
  etc.); an ablation-testing notebook built on top of the exact same
  Tran-Hoang-reconstruction source, packaged as the "Anchor Exact" candidate.
- `flexonafft/kaggriculture-multi-route-farming-agent` (86 votes) and
  `kunaldesale2408/kaggriculture-2026-v1` (18 votes, claimed 2571.0) -- both
  are the byte-identical "Kawa" source used by `kawa_route_agent.py`
  (`kunaldesale2408`'s copy matches to the exact SHA-256).
- `indarkarhana/rank-top10-read-the-market-choose-the-farm` (69 votes, claimed
  2149.5) -- the same "BL-MDgogo-10C4S-R0" codename already identified as a
  `kawa_route_agent.py` republish when `salemali7`'s copy was excluded (see
  "Why `salemali7`..." above); a third copy of the same source.
- `ameythakur20/kaggriculture-deterministic-farm-planning-agent` -- 100%
  overlap with `boatlee_v16_agent.py`; its embedded `main.py` literally opens
  `"""V16-RC5-PremiumMarketLead for Kaggriculture."""`, boatlee's own docstring
  title, unchanged.
- `andrewsokolovsky/kaggriculture-breaking-the-tie` (65 votes) -- 0.737
  Jaccard overlap with `kaito_agent.py`; an ablation/ranking-tuning notebook
  built directly on top of a "recovered V22" source from the same family of
  shared helper functions (`_impact_score`, `_weed_repair_action`, etc.) as
  `kaito_agent.py` and `romanrozen_agent.py`, not a distinct strategy.
- `rauffauzanrambe/kaggriculture-finding-conditional-fresh-vegetable` -- 0.758
  Jaccard overlap with `romanrozen_agent.py`; a meta-analysis notebook whose
  embedded "C68" agent is explicitly built by combining a public field route
  with `romanrozen_agent.py`'s shared route-family helper functions.

No usable `agent(obs)` function:

- `jeffmarcecadet/reverse-engineering-rank-5-thunder-thunder` -- pure static
  analysis writeup (replay statistics and prose), no agent code at all.
- `mansiaggarwal88/kaggriculture-build-your-first-farming-agent` -- a
  beginner tutorial notebook; its only agent (`my_wheat_farmer`) is an
  intentionally toy single-crop function built for teaching, not the
  author's actual scoring strategy, and not representative of their 510.5
  leaderboard score.
- `jek1wantaufik/building-a-kaggriculture-ai-agent` -- assembles its
  submission from a Kaggle Model input (`/kaggle/input/models/...`) not
  included in the notebook pull; no retrievable source.

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
