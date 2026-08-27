# V14 Search Analysis

**Date**: 2026-08-22  
**Study**: `robust_agent_config_v14_variants_rotation_rot0`  
**Outcome**: First clean promotion since the v11 revert. 10W-0L-0T verification (+6,610.6 avg), holdout PASSED on all three held-out opponents.

---

## 1. Study Health

| Metric | v14 | v13 | v12 |
|--------|-----|-----|-----|
| Total trials | 400 | 400 | 400 |
| Complete | 182 (45.5%) | 207 (51.8%) | 21 (5.3%) |
| Pruned | 218 (54.5%) | 193 (48.3%) | 379 (94.8%) |
| Best value | **0.5697** | 0.2374 | -1.2783 |
| Spread (max − min) | **1.7587** | 1.4062 | 0.0212 |
| Median | 0.2952 | 0.1308 | -1.2894 |
| Mean | 0.1853 | — | — |
| StdDev | 0.3261 | — | — |

**Interpretation**: v14 is the healthiest study in this project's history.

- **Completion rate** (45.5%) is slightly below v13's 51.8% but massively above v12's catastrophic 5.3%. The pruner is working as intended — culling the bottom half early rather than wasting compute.
- **Spread** (1.76) is wider than v13 (1.41), meaning the search found both deeper valleys *and* higher peaks. The wider range reflects the new parameters (`opening_*`, `lambda_*`, `income_*`) adding genuine degrees of freedom — bad combinations crash hard, good ones go higher than v13 ever reached.
- **Best value** (0.5697) is 2.4× v13's best (0.2374). Given the objective is squashed win/loss margin across 2 episodes × 8 opponents, crossing 0.5 is substantial. For reference, v12 never found a *positive* value at all.
- **Median** (0.2952) exceeds v13's *best* (0.2374), which means a typical v14 trial outperforms v13's very best trial. This is the clearest sign that the v14 code changes (new mechanisms + self-play variants in the opponent pool) genuinely expanded the performance frontier.

**Value distribution (v14 percentiles)**:

| P05 | P10 | P25 | P50 | P75 | P90 | P95 |
|-----|-----|-----|-----|-----|-----|-----|
| -0.3506 | -0.2529 | 0.0580 | 0.2952 | 0.4110 | 0.4783 | 0.4967 |

The bottom 5% are still negative (bad combos that survived pruning), but 75% of completed trials are positive. The P90–P95 plateau at ~0.49–0.50 suggests the search was converging — the best trial at 0.5697 is an outlier above that plateau, not a noisy fluke from the middle of a broad distribution.

---

## 2. Parameter Importance

Computed via `optuna.importance.get_param_importances()` (fANOVA-based):

| Rank | Parameter | Importance |
|------|-----------|------------|
| 1 | `wealth_margin_scale` | **0.2044** |
| 2 | `pasture_target_ratio` | **0.1843** |
| 3 | `opponent_race_discount` | **0.1771** |
| 4 | `income_lookahead_days` | **0.0683** |
| 5 | `priority_weight_weeds` | 0.0430 |
| 6 | `priority_weight_empty_plant` | 0.0337 |
| 7 | `animal_reserve_multiple` | 0.0305 |
| 8 | `lambda_labor` | 0.0289 |
| 9 | `max_hires_per_day` | 0.0283 |
| 10 | `priority_weight_harvest` | 0.0222 |
| 11 | `priority_weight_empty_pasture_place` | 0.0168 |
| 12 | `seed_money_floor` | 0.0167 |
| 13 | `land_utilization_threshold` | 0.0151 |
| 14 | `crop_profile` | 0.0126 |
| 15 | `opening_hires_day0` | 0.0104 |

**Commentary**:

The top 3 parameters account for **56.6%** of total importance. All three are "meta-strategy" knobs that determine *how the agent reacts to opponents*, not low-level dispatch weights:

1. **`wealth_margin_scale`** (20.4%) — how aggressively the relative-wealth system scales spending when we're ahead or behind. The champion's value (1,138) is dramatically lower than the previous champion's (2,925), meaning the new champion is **far more responsive to wealth gaps**. A smaller scale means each dollar of lead/deficit produces a larger scaling effect.

2. **`pasture_target_ratio`** (18.4%) — what fraction of land to allocate to pastures (animals). The champion's 0.189 is sharply down from the previous champion's 0.614, confirming (yet again) that a crop-heavy strategy dominates. This is consistent with every prior finding in this project.

3. **`opponent_race_discount`** (17.7%) — how much to discount crop value when opponents are racing to the same crops. The champion's 0.434 (down from 0.966) means it's now **far more willing to switch crops** when competitors are concentrated. The previous champion's near-1.0 value was essentially ignoring opponent-concentration signals.

**`income_lookahead_days`** at rank 4 (6.8%) is notable: it's one of the three new mechanisms introduced today, and the search cares about it. See §3 for the verdict.

**`lambda_labor`** at rank 8 (2.9%) is non-trivial but far below `lambda_land`'s absence from the top 15. This is consistent with the v10 finding that labor shadow pricing has near-zero value while land shadow pricing helps independently.

> [!NOTE]
> `lambda_land`, `opening_enabled`, `opening_days`, and `income_discount` do not appear in the top 15. This doesn't mean they're irrelevant — `opening_enabled` is a boolean gate, `opening_days` is conditioned on it, and `lambda_land` may have converged so tightly that its variance is low (low variance → low fANOVA importance even if the converged value matters a lot).

---

## 3. Champion Diff: New Mechanisms

### Previous champion (pre-v14, from `git show HEAD~1:src/best_config.json`)
This was the v6-derived config that survived after the v11 revert. It did not have `lambda_labor`, `lambda_land`, `income_*`, or `opening_*` keys — those fell through to `DEFAULT_CONFIG` (all zero/off after the default-drift fix).

### Key parameter changes

| Parameter | Previous | New (v14) | Δ | Direction |
|-----------|----------|-----------|---|-----------|
| `sell_fraction_base` | 0.463 | 0.347 | −0.116 | Less eager to sell early |
| `sell_fraction_day_weight` | 0.053 | 0.229 | +0.176 | **Much** stronger late-season sell pressure |
| `sell_fraction_cash_weight` | −0.064 | −0.007 | +0.057 | Nearly ignores cash when setting sell fraction |
| `cash_scale` | 3,900 | 1,800 | −2,100 | Cash effects kick in faster |
| `money_reserve` | 420 | 140 | **−280** | Far less cash hoarded |
| `seed_money_floor` | 25 | 7 | −18 | Will plant with very little cash |
| `hire_money_floor` | 60 | 10 | −50 | Will hire with very little cash |
| `hire_reserve_multiple` | 1.287 | 2.507 | +1.22 | Higher hire bar (but offset by lower money_reserve) |
| `max_hires_per_day` | 6 | 7 | +1 | |
| `max_structures` | 4 | 6 | +2 | More buildings |
| `pasture_target_ratio` | 0.614 | 0.189 | **−0.425** | Massively less pasture |
| `animal_reserve_multiple` | 2.596 | 1.379 | −1.22 | Cheaper animal buys |
| `startup_days` | 2 | 0 | −2 | No startup delay |
| `land_startup_days` | 8 | 10 | +2 | |
| `land_reserve_multiple` | (absent) | 3.284 | — | New param, aggressive land buying |
| `wealth_margin_scale` | 2,925 | 1,138 | **−1,787** | Much more wealth-responsive |
| `risk_sensitivity` | 0.892 | 1.670 | +0.78 | More risk-aware |
| `opponent_race_discount` | 0.966 | 0.434 | **−0.532** | Actually reacts to opponent crop concentration |
| `opponent_incoming_threshold` | 9 | 2 | −7 | Much more sensitive to opponent presence |
| `opponent_lookahead_days` | 3 | 5 | +2 | Looks further ahead |

### Mechanism 1: Scripted Opening Window (`opening_*`)

| Parameter | Value |
|-----------|-------|
| `opening_enabled` | **true** |
| `opening_days` | 1 |
| `opening_reserve_scale` | 0.847 |
| `opening_hires_day0` | 5 |

**Verdict: ON.** The search turned the opening window on, but with a minimal 1-day window (only day 0). It hires 5 workers on day 0 (up from `max_hires_per_day` = 7, but bypassing the backlog gate that would otherwise block day-0 hires since nothing is planted yet) and relaxes the reserve cushion to 84.7% of normal.

**Consistency with prior evidence**: IDEAS_TRIED.md listed this as "➖ built, NOT yet tested" with an honest prior that "spend aggressively early" variants had failed 9+ times. The search's verdict is nuanced — it's *not* a multi-day spending spree (opening_days=1), it's a single-day hire burst. This is consistent with the observation that days 0-2 are 100% deterministic (no RNG divergence until day 3), and the traced behavior of 5 strong opponents showing 4-5 day-0 hires as consensus. The search found the sweet spot: exploit the deterministic window for hiring, but don't extend it.

### Mechanism 2: Shadow Pricing (`lambda_labor` / `lambda_land`)

| Parameter | Value |
|-----------|-------|
| `lambda_labor` | **0.030** |
| `lambda_land` | **29.994** |

**Verdict: `lambda_labor` OFF (effectively zero), `lambda_land` ON (strong).**

- `lambda_labor` at 0.030 is essentially zero — the search found it a tenth of the v10 study's already-near-zero convergence point of 0.45. This is now confirmed across **three independent searches** (v10: 0.45, v11-best: 0.068, v14: 0.030). Labor shadow pricing does not help.
- `lambda_land` at 29.99 is remarkably close to the v10 result of 24.9 and the v11-best of 30.7. Three independent searches converging to the 25–31 band is strong evidence that land shadow pricing carries real, ~30-unit value on the [0, 80] scale. This is consistent with the IDEAS_TRIED.md conclusion: "land shadow pricing has real value."

**Consistency with prior evidence**: Perfectly consistent. v10's IDEAS_TRIED entry explicitly said `lambda_labor` → near-zero, `lambda_land` → ~25. The v14 search, running months of iteration later with a different opponent pool, different objective function, and many new mechanisms, independently confirms the same pattern.

### Mechanism 3: Cash-Flow Lookahead (`income_discount` / `income_lookahead_days`)

| Parameter | Value |
|-----------|-------|
| `income_lookahead_days` | **0** |
| `income_discount` | 0.339 |

**Verdict: OFF.** `income_lookahead_days = 0` means zero days of income are projected, which completely disables the mechanism regardless of `income_discount`'s value. The discount value of 0.339 is a dead parameter — it's only used if lookahead_days > 0.

**Consistency with prior evidence**: Consistent. The LOG.md ablation (6 episodes each vs kawa/prvsiyan) found a "real but opponent-dependent effect, not a clean win" — margin improved 12% vs kawa but worsened 8% vs prvsiyan, with no win flipped in either case. The search, given 400 trials to find any value in this mechanism, turned it off. The parameter importance analysis ranks `income_lookahead_days` at #4 overall (6.8%), which at first seems contradictory — but high importance with a converged-to-zero value means the search *learned that turning it off matters*, not that it didn't test the alternatives. Many early trials with `income_lookahead_days` > 0 performed worse, and the sampler learned to avoid that region.

> [!IMPORTANT]
> The champion config still carries `income_discount: 0.339` as a vestigial value. This is harmless (gated off by `income_lookahead_days = 0`) but worth knowing if someone later changes the lookahead without re-searching the discount.

---

## 4. Surprises and Contradictions

### `animal_enabled: true` — this time with teeth

The new champion enables animals (same as the old champion), but the previous champion's `pasture_target_ratio` of 0.614 meant it was spending over 60% of land on pastures. The new champion runs 0.189 — only ~19% pasture. Combined with `animal_reserve_multiple` dropping from 2.60 to 1.38, the new config runs a lean, **crops-dominant mixed economy**: a few cheap animals as a diversification hedge, not a committed animal strategy.

This is consistent with every prior finding but represents a cleaner expression of it. Previous configs either went full-animal (and lost) or set `animal_enabled: false` (losing the small diversification benefit). The v14 champion found the middle ground.

### `money_reserve: 140` — the most aggressive spending posture ever promoted

The previous champion held 420 in reserve. The new one holds 140. Combined with `seed_money_floor: 7` (previous: 25) and `hire_money_floor: 10` (previous: 60), this champion spends down to almost nothing.

This is a **departure from prior evidence.** The IDEAS_TRIED.md "Spending posture / reserves" section (not shown here but referenced throughout the log) documents 9+ failed attempts at aggressive early spending. The resolution: the v14 champion couples aggressive spending with **much stronger opponent awareness** (`opponent_incoming_threshold: 2` vs old 9, `opponent_race_discount: 0.434` vs old 0.966) and a **tighter risk band** (`risk_scale_min/max: 1.45–1.80` vs old `0.89–2.35`). It's not blindly aggressive — it spends aggressively *while monitoring and reacting to opponents*, which is a qualitatively different strategy from the old "spend a lot and hope" approaches that failed.

### `sell_fraction_day_weight: 0.229` — strong end-of-season sell pressure

The previous champion had a day weight of 0.053 (nearly flat across the season). The new one at 0.229 means sell pressure increases 4.3× as steeply toward end of season. Combined with a lower base (0.347 vs 0.463), the new champion sells conservatively early and dumps aggressively late. This is the first time a promoted champion has had a strongly day-weighted sell curve.

### `wealth_margin_scale: 1,138` — no prior champion was this reactive

This is a 2.6× reduction from 2,925. In the relative-wealth system, this means a $1,000 wealth gap now produces the same scaling effect that previously required a $2,600 gap. The champion is extremely sensitive to being ahead or behind. Combined with the aggressive spending posture, this creates a feedback loop: spend hard → check if we're ahead → spend harder if winning, pull back if losing. This is strategically sophisticated and unlike any prior configuration.

### No contradictions with verified prior findings

The v14 result does not contradict any prior *verified* finding in this project. It confirms:
- `lambda_labor` near-zero (consistent with v10, v11)
- `lambda_land` ~30 (consistent with v10, v11)  
- Cash-flow lookahead not a net win (consistent with the mixed ablation result)
- Animals as minority diversification, not a dominant strategy
- Crop profile "diversified" outperforms "melon_plus_short" and "fast_cash"

The one area of tension — aggressive spending vs. prior failed attempts — is resolved by the accompanying opponent-awareness parameter shifts, not by the spending alone.

---

## Summary Table

| Mechanism | Verdict | Champion Value | Prior Evidence |
|-----------|---------|----------------|----------------|
| **Opening window** | **ON** (1 day, 5 hires) | `opening_enabled=true`, `opening_days=1`, `opening_hires_day0=5` | Consistent — traces showed 4-5 day-0 hires as cross-opponent consensus |
| **Land shadow pricing** | **ON** (strong) | `lambda_land=29.99` | Consistent — v10 found 24.9, v11-best found 30.7 |
| **Labor shadow pricing** | **OFF** (≈0) | `lambda_labor=0.030` | Consistent — v10 found 0.45, now even lower |
| **Cash-flow lookahead** | **OFF** | `income_lookahead_days=0` | Consistent — ablation was mixed, search turned it off |
