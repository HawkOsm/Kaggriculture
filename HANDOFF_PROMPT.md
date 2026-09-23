# Kaggriculture — continuation prompt

Continue my Kaggriculture competition work. Everything below is measured, not assumed.
Repo: /home/osm/Projects/Kaggriculture, branch `distill-investigation`, Kaggle user `HawkOsm`.
Competition deadline **2026-09-30**, ~9,773 teams, top score ~3,155. Medal ≈ top 10% (~rank 977 ≈ score ~2,410).

## GOAL
Maximise ladder score with an agent that is **our own work**. Do NOT submit a verbatim copy of
anyone's agent (I did that once for calibration; it's off-limits now). Building ON a public base
with real, measured modifications IS fine and must be credited in CREDITS.md.

## ARCHITECTURE DIRECTION (settled)
Use **adaptive tape** = a library of distilled replay tapes + a routing rule that picks between them.
- Pure reactive agents are a dead end (our own reactive champion tops out at 820).
- A single frozen tape tops out ~1,400.
- Adaptive tape is what every top agent actually ships (kawa, boatlee, Roxy, hanif all do it).

## CURRENT STATE
Best own agent: `submission/main_router_v2.py` (committed 4078c43) — **not yet submitted, quota was blocked**.
Out-of-sample, 360 fresh-seed games vs a rank-1–40 field:
```
ROUTER_v2              100/120 = 0.833   mean margin +40598
best single tape        99/120 = 0.825   mean margin +34582
main_ladder.py (LIVE)   92/120 = 0.767   mean margin +21204
```
Real ladder scores so far (this is ground truth; local tests are only a proxy):
```
main_hanif_base.py  2229.7   <- VERBATIM COPY of hanif's public agent. Do not reuse. Displace it.
main_ladder.py      1408.1   <- our distilled tape, currently the best OWN agent live
main_distill.py      951.8
main_improved.py     896.5
main.py (reactive)   820.7
main_roxy_base.py    812.2
```

### Update 2026-09-23: hanif base + our patches (`submission/main_hanif_plus.py`)
Built by `src/distill/build_hanif_plus.py` (4 asserted patches on hanif's public agent, credited in
CREDITS.md). Not yet submitted. Fresh-seed paired validation (2 blocks x 100 contexts):
```
vs prvsiyan_current   +618 / +543 mean margin per game (better in 53/60), wins 7/60 vs base 3/60
vs hanif mirror       51 W / 9 L of 60   (base hanif: ties)
guards (roxy, kawa, 2 top tapes)   wins unchanged 78/80, margin down up to ~466
head-to-head vs main_router_v2: 18/20 wins, mean margin +19002 (plain hanif: identical)
```
So hanif_plus is the strongest agent we have (it inherits hanif's 2229.7-level strength), but most of
that strength is hanif's; router_v2 remains the best agent that is entirely our own work.
Dead ends from this round: re-routing hanif's 40 prefix-aligned routes (CV gain ~0); grafting our
replay tapes onto hanif (0 of 2,465 share its opening); 22/40 hanif constants are dead code;
prvsiyan's clone-race layer never fires vs hanif (its edge is in economy layers).
Harness: `.claude/scratch/hanif_route/` (common.py, variants.py, anknob.py).

## IMMEDIATE NEXT STEPS
1. Choose what to submit: `submission/main_hanif_plus.py` (strongest) or
   `submission/main_router_v2.py` (fully our own). Submit the preferred one LAST.
2. The wider routing fit for router v2 (`.claude/scratch/kaggle_data/router_signal_big.py`,
   2,240 games) died without output; rerun it if continuing the router-v2 line, rebuild via
   `src/distill/build_router_v2.py`, and only ship if it beats v2 on fresh seeds.
3. Further ideas, in rough value order: more continuations per shop; richer routing signal
   (2-shop sequence, opponent farm composition at step 144); routers over OTHER prefix-aligned
   families (THIRD FARM CLUB 208 @72, Sida Zuo 129 @72, Majkel1337+QQ 66 @72).

## HOW THE ROUTER WORKS (and why v1 failed)
- v1 switched at step 72 into continuations recorded in DIFFERENT games -> their market orders
  fired against a state that never happened -> scored 0.417 vs a single tape's 0.500. Dead end.
- v2 draws every continuation from ONE **prefix-aligned family**: 171 tapes byte-identical
  through step 144, so our farm state at the switch is exactly what each continuation assumes.
  `build_router_v2.py` asserts this and refuses to build otherwise. Families are in
  `.claude/scratch/kaggle_data/families.json`.
- Routing signal = which town shop is unlocked first (visible by step 72; decided at 144).
  Shops set 1.5–2.0x demand multipliers, which is the mechanism that makes them discriminate.
  Measured map: YARN_STORE->0918#218, PET_CAFE->0918#217, ICE_CREAM_SHOP->0918#219,
  FARMERS_MARKET->0917#209, BRUNCH_SPOT->0918#218, else best-overall 0917#207.

## DATA AND TOOLING (all already built)
- `.claude/scratch/kaggle_data/top_tapes.json` (588 tapes, day 2026-09-20) plus
  `top_tapes_2026-09-{17,18,19}.json` (~1,877 more). 2,465 total, from top-40 teams' winning
  ladder games. Extraction is **bit-exact verified** (replaying both seats reproduces the
  episode's exact final rewards) with alignment `action[t] = steps[t+1][seat]['action']`.
- `families.json` (prefix-aligned groups), `ws_*.json` (small working sets).
- `.claude/scratch/decoded_agents/` — decoded public agents for study: hanif_current.py,
  prvsiyan_current.py, roxy_v21r1.py, kawa's 10 tapes, plus knob_*.py variants.
- `.claude/scratch/foreign_tapes/tapes/` — 18 tapes harvested from 320 public notebooks.
- `src/distill/` — package.py, package_ladder.py, build_router.py, build_router_v2.py,
  tape_repair.py, improve_hanif.py, build_library.py, replay_lib.py.
- Kaggle CLI is authenticated at `.venv/bin/kaggle` (no ~/.kaggle/kaggle.json needed).
- Daily episode dumps: `kaggle datasets download -d kaggle/kaggriculture-episodes-YYYY-MM-DD`
  (~600MB zip, ~640 episodes, ~21GB unpacked — NEVER unpack, stream with zipfile).

## DEAD ENDS — do not redo these
- **All RL**: config-tuning, state-conditioned, BC->PPO (0%), reward-shaped (degenerate farmer
  that plants but never waters). The game's sparse reward + 720-step horizon defeats it.
- **Tape repair** (substituting provably-dead taped actions): 12.8% of taped work actions ARE
  no-ops, but reclaiming them is worth ~0 (paired delta -43 and -1863). Kept as a negative.
- **Knob-tuning hanif's 60 layers**: mostly EXACTLY zero delta (dead code paths). Only
  `_ALT_MODE='TomatoInsteadOfCow'` was positive (+1735 money, 15/24 vs 14/24 wins) = noise-level.
- **Foreign tapes**: all 18 harvested tapes lose to our own #467 on a 90-game test
  (ours 79/90 vs best foreign 66/90). The public tape supply has been searched.
- **Swapping our tape into Roxy's `_EXPERT_TAPE`**: scored 727.7 vs their 812.2 — their tape is
  co-tuned with their router.
- **Forking public notebooks to inherit rank**: Roxy is LB #61 (2826.7) but their PUBLISHED agent
  scores 812. (Note: hanif's published agent scored 2229.7 vs their 2409 — so weakening is not
  universal, but you cannot assume a notebook performs at its author's rank.)

## METHODOLOGY RULES (learned the hard way — follow these)
1. **The ladder is ground truth.** Local tape-vs-tape only ranks *static tapes against each other*;
   it cannot measure adaptivity. hanif beat our tape 4/4 locally yet the ladder gap is 2229 vs 1408.
2. **Ladder noise is ±90.** The SAME tape scored 1318.5, 1411.4 and 1320.7. Ignore smaller diffs.
3. **Scores converge slowly** — hanif's went 600 -> 2229 over hours. Never conclude from an early read.
4. **Never conclude from one game.** I claimed a +40k breakthrough from a single seed; the paired
   test said neutral. Use paired A/B: same seeds, same opponents, one variable.
5. **Always re-rank finalists on seeds never used for selection.** Two "perfect 48/48" tapes and one
   "better" tape all collapsed out-of-sample. In-sample routing gain +9107 -> +6016 out-of-sample.
6. **Select against a DIVERSE field** (rank bands 1–40), scored on win-rate AND worst case, not mean.
   Selecting against 6 similar opponents produced a specialist that would have been a downgrade.

## OPERATIONAL GOTCHAS
- **Kaggle's loader picks the LAST callable by insertion order.** Any layered agent must end with
  `agent = globals().pop("agent")`. Without it a helper function gets called as the agent and the
  run silently scores exactly 3000 (starting money). This cost me hours.
- **Do not poll the Kaggle CLI in a loop** — it got the process killed. Query sparingly.
- **Memory**: worker pools that load the big tape JSONs hit ~4GB EACH and filled all 30GB. Build a
  small working-set JSON (2–3MB) first, use `mp.get_context('fork')`, cap workers ~6, and guard with
  a loop that kills the job if free memory <3GB.
- **Orphaned workers**: `pkill -f <script>` misses forked children (different cmdline). Also
  `pkill -9 -f multiprocessing`. But killing a forkserver while a pool is alive DEADLOCKS that pool
  (workers sit at ~0% CPU in state S).
- **agy is unreliable for long jobs** — it spawned a background task and exited three times,
  producing nothing ("terminating N background task(s) on exit"). Run long jobs directly with nohup.
- **Rank follows active submissions**, so submitting a weak agent drops you (#3110 -> #5093 once).
  Submit your best agent LAST.

## RULES / ETHICS
- Kaggle staff confirmed: "Using public replays to train, build, inform your submission is allowed
  and encouraged." Public Code-tab code is OSI-licensed per Foundational Rules §3.6.
- Distilling public REPLAY DATA is fine and is the intended meta. Submitting someone's AGENT
  verbatim is not what I want, even though it is license-permitted. Credit everything in CREDITS.md.
- Git and Kaggle actions: just do them, but submit only our own work.

## OPERATING MODE: AUTONOMOUS
Work continuously without waiting for me. Do not ask permission for routine steps — just do them
and report. Specifically:
- Run long jobs detached (`nohup ... & disown`) and watch them with a monitor loop that also kills
  the job if free memory drops below 3GB. Never block on a foreground `sleep`.
- When the daily Kaggle quota resets, submit the best VALIDATED own agent automatically, best last.
- After each experiment: if it beats the incumbent out-of-sample, package it, commit it with the
  numbers in the message, push, and submit. If it does not, record it as a documented negative in
  the commit/memory and move to the next idea. Never ship a result that is inside the ±90 noise band.
- Keep a running priority queue of ideas; when one dies, start the next without prompting.
- Only stop to ask me if something is genuinely irreversible or outside the rules above.

## AUTO-SUBMIT
`scripts/auto_submit.sh` waits for the quota to reset and submits the best validated agent, then
logs to `.claude/scratch/auto_submit.log`. Re-launch it after any submission with the new best file.
