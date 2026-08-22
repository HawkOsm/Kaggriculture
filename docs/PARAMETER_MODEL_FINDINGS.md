# Parameter-model findings — survey of 15 public Kaggriculture agents

Gathered 2026-08-22 via a 3-layer research pipeline (discovery → extraction → synthesis) over
public competition notebooks. Raw extractions: `../.claude/scratch/paramresearch/`. Leaderboard
context: 5,829 teams, rank 1 ≈ 3145, our score ≈ **500** (rank ~4000–4500, bottom quartile).

---

## 1. The leaderboard splits cleanly at ~1800

| Tier | Agents | Mechanism |
|---|---|---|
| **1770–2719** | raykkretzschmar, boatlee, denizeryilmaz, tetsutani, pilkwang(precomputed), prvsiyan, romanrozen | **Precomputed replay tape** — base64/base85 + zlib compressed action sequences, or compiled blobs, with a small reactive override layer |
| **≤2008** | pilkwang(structured-economic-policy) 2008, alexandergremyakov 1660, anasriaz 1272, chaitanyajamble 516, rajan1673 427 | Genuinely reactive per-turn logic |

**Every agent above ~1800 replays a precomputed sequence.** The highest confirmed genuinely-reactive
agent is Pilkwang's structured-economic-policy at **2008** — and the same author publishes a
precomputed-schedule variant at the same score, so even that is an ambiguous ceiling.

**Practical read:** a live reactive heuristic — which is what we are — appears to top out around
**~2000**. That is still **4x our current ~500**. Exceeding ~2000 looks to require a tape.

## 2. Parameter counts run *inverse* to score

| Agent | Score | Params |
|---|---|---|
| boatlee rc5 | 2545 | 3 |
| raykkretzschmar rank-your-agent | 2719 | 4 |
| pilkwang precomputed | 2008 | 4 |
| denizeryilmaz | 2244 | 6 |
| pilkwang structured | 2008 | 14 |
| tetsutani | 2061 | 14 |
| boatlee rc2 | 2545 | 15 |
| anasriaz | 1272 | 16 |
| romanrozen | 1770 | 16 |
| alexandergremyakov | 1660 | 21 |
| **ours** | **~500** | **59** |

Nobody in the survey exceeds 21 parameters. We have **59** — roughly 3x the count of agents
scoring 4x higher. Tape agents have few parameters because the parameters govern only the
*override layer*, not the strategy (romanrozen/tetsutani: *"The parameters solely govern these
adaptive overrides"*).

**Count alone is not the driver** — `rajan1673` (11 params) scores 427 and `chaitanyajamble`
(20) scores 516, both near us. What separates Pilkwang's 14 from their 20 is *what the parameters
mean*: real marginal-price/supply-demand economics and opponent schedule detection, versus more
threshold knobs. The lesson is not "delete parameters", it is **"parameters should encode
economics and phase structure, not more thresholds."**

## 3. Structural patterns worth adopting

- **Phase-scoped parameters with explicit day windows.** Near-universal among the reactive tier:
  `from_day`/`until_day` (raykkretzschmar), `HERD_EXPANSION_DAY=7`, `HERD_FINAL_DAY=11`,
  `ANIMAL_PURCHASE_LAST_DAY=18`, `LAND_OPEN_DAYS=(5,9)` (pilkwang), `_PREEMPT_START=120` /
  `_PREEMPT_STOP=680` (romanrozen, tetsutani), land buying restricted to days 11–20
  (chaitanyajamble), `second_goose_buy_day=3` / `second_goose_buy_deadline=8`
  (alexandergremyakov). We have almost none of this — our behaviour is governed by continuous
  thresholds that apply uniformly all game.
- **Terminal liquidation is universal.** Day-29 / step-716 dump of all shed inventory, since
  unsold goods score nothing. Present in rajan1673 (427) through romanrozen (1770). We have
  `wind_down_days`, which is the same idea — worth verifying it actually fires correctly.
- **Policy decoupled from logic.** raykkretzschmar's agent *"simply modifies a central POLICY
  dict and passes it to a generic scheduler"* — declarative phase-scoped policy rather than
  thresholds embedded throughout imperative code.
- **Opponent state detection.** Pilkwang checks a `_schedule_signature` (tile counts + unlocked
  quadrants at steps 24/192/264) to identify which opponent script it faces; boatlee does
  "near-mirror" state matching at steps 216/240/264; prvsiyan picks among pre-recorded routes by
  nearest-neighbour feature match. Several strong agents explicitly *fingerprint the opponent*.

## 4. The methodology gap (largest single finding)

Two of the top-scoring authors tune against **fixed seeds**:

- raykkretzschmar (2719): *"manually trying 32 configs and testing on held-out seeds"*
- boatlee (2545): *"hand-tuned against local frozen 12-seed panels"*

**Episodes in this environment are fully seedable** (`configuration={"seed": N}`, documented at
`GAME_GUIDE.md` line 262) and this was verified directly: seed 42 run twice returns byte-identical
results; seed 7 differs. **We have never used it** — every ablation, search and verification to
date ran unseeded.

The seed effect is enormous: against the same config, kawa scores **63,732 on seed 42 and 111,771
on seed 7** — a 75% swing from world randomness alone, far larger than most config differences we
have been trying to measure. This is the direct cause of the repeated small-sample false positives
(the relative-wealth result, the 4-episode "win" that evaporated, the v11 promotion that had to be
reverted).

With a frozen seed panel, comparisons become **paired** — both configs play identical worlds, so
seed variance cancels instead of drowning the signal. This is why the top authors get decisive
answers from ~32 hand-designed configs while we get noise from thousands of Optuna trials. They
were not searching harder; they were measuring properly.

---

## Recommended actions, in order

1. **Seed all evaluation.** Add a fixed seed panel to `play_episode`/`evaluate_config`/
   `verify_candidate`/`verify_holdout`. Cheapest change here and it improves the reliability of
   everything downstream. Do this before any further searching.
2. **Add explicit phase windows** for animal purchase, land purchase and herd expansion, matching
   the near-universal pattern above. Our `opening_*` work is the first instance of this; extend it
   to mid/late game rather than relying on uniform thresholds.
3. **Audit the 59 parameters.** Cut or merge knobs that encode nothing economic. Target the 15–20
   range that every surveyed agent lives in, and prefer parameters that express economics
   (marginal price, opportunity cost) over parameters that express thresholds.
4. **Consider opponent fingerprinting.** Several strong agents identify which script they face
   from public tile state at fixed steps. We already have `_opponent_profile`; this is a natural
   extension and the opponents are, as established, mostly fixed routes.
5. **Decide on tapes deliberately.** Reactive tops out ~2000. Reaching that would be a 4x
   improvement and is the sensible near-term goal. Going beyond it is a separate architectural
   decision (see `ROADMAP.md` Phase 3, Option A).
