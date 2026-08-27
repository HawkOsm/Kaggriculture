# Kaggriculture

Local workspace for Kaggle's [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition — build an autonomous AI agent to run a virtual farm, trade on a dynamic market,
and compete head-to-head on a live leaderboard. $50,000 prize pool (10 × $5,000). Entry deadline:
**September 23, 2026**; final submission deadline: **September 30, 2026**.

## Layout

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — **start here.** Current strategic position, phased plan, and explicit non-goals (things already tried and ruled out).
- [`docs/RULES.md`](docs/RULES.md) — official competition rules (snapshot; always re-check the
  [live rules page](https://www.kaggle.com/competitions/kaggriculture/rules) before relying on any
  deadline or clause).
- [`docs/GAME_GUIDE.md`](docs/GAME_GUIDE.md) — full game mechanics reference we wrote from the public Overview
  page: object types, actions, market pricing formulas, observation format, config defaults.
- [`docs/kaggriculture/`](docs/kaggriculture/) — the **official** starter kit bundle downloaded via
  `kaggle competitions download kaggriculture` (README.md + AGENTS.md), gated behind accepting the
  competition rules on Kaggle. Cross-checked against `GAME_GUIDE.md` — they agree.
- [`docs/starter_kit/`](docs/starter_kit/) — our own walkthrough docs: CLI setup / local-test / submit workflow,
  plus a tutorial for the official "Melon Maxxer" example agent.
- [`docs/tests/`](docs/tests/) — log of every local test outcome (benchmark runs, correctness checks, before/after
  comparisons). This is a project rule, not optional -- see [`AGENT.md`](AGENT.md).
- [`input/`](input/) — source of verified top-placing solutions from comparable past Kaggle simulation competitions, kept as hyperparameter/architecture reference (gitignored, local only; each subfolder has a `PROVENANCE.md`, index at `input/index.json`).
- [`src/`](src/) — the actual agent code (not under `docs/`):
  - `farm_utils.py` — shared movement/geometry helpers (`step_toward`, `closest`, `act_or_move`,
    `shed_tiles`) used by our own agents.
  - `agent.py` — our current best agent (exported as `robust_agent`): crash-safe, multi-unit
    (farmer + hands), dynamic crop scoring, hires/land/animals/fertilizing all wired up.
  - `opponents/` — local benchmark opponents pulled from public Kaggle notebooks (see
    [`../CREDITS.md`](CREDITS.md) for attribution) — never part of our own submission.
  - `run_match.py` — CLI to run any two agents against each other locally, with an optional HTML replay.

## Quick Start

```bash
pip install -U kaggle-environments kaggle
```

From the repo root:

```bash
python src/run_match.py agent:robust_agent kawa_route_agent:kawa_route_agent
```

See [`starter_kit/README.md`](docs/starter_kit/README.md) for the full local-test-and-submit workflow
(Kaggle CLI setup, accepting the rules, downloading data, submitting an agent, checking the leaderboard).
