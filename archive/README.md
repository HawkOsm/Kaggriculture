# archive/ — documented dead ends

Code kept for the record, not used by any current submission. Each was measured and rejected;
see `HANDOFF_PROMPT.md` (DEAD ENDS) and `docs/tests/LOG.md` for the numbers.

| path | what it was | why it's here |
|------|-------------|---------------|
| `rl2/` | behaviour-cloning of top agents → PPO | 0% win rate; sparse reward + 720-step horizon defeats RL |
| `distill/build_router.py` | router v1: switch at step 72 into continuations from *different* games | 0.417 vs a single tape's 0.500 — continuations fire against a state that never happened (v2 fixes this with prefix-aligned families) |
| `distill/tape_repair.py` | replace provably dead taped work actions | paired delta ≈ 0 (−43, −1863) |
| `distill/improve_hanif.py` | premium front-run layer on the hanif base | paired delta −156, neutral |
| `distill/improve_base.py` | swap our tape into Roxy's `_EXPERT_TAPE` | 727.7 live vs their 812.2 — tape is co-tuned with its router |

Scripts compute the repo root from their own depth (same as under `src/`), so they still run from here.
