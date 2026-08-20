# Kaggriculture Starter Kit

Source: https://www.kaggle.com/competitions/kaggriculture/overview ("Getting Started: Test Locally & Submit")
and https://www.kaggle.com/code/bovard/kaggriculture-getting-started (fetched 2026-08-20)

Full game rules/mechanics: [`../GAME_GUIDE.md`](../GAME_GUIDE.md)
Official competition rules: [`../RULES.md`](../RULES.md)
Official starter kit bundle (downloaded, gated on Kaggle): [`../kaggriculture/README.md`](../kaggriculture/README.md), [`../kaggriculture/AGENTS.md`](../kaggriculture/AGENTS.md)

## Test Locally

Install the environment (any recent release that includes Kaggriculture):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook — pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([agent, "random"])  # or env.run(["main.py", "random"]) to load from a file

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
env.render(mode="ipython", width=1200, height=800)

# Or dump a replay JSON for the visualizer / offline analysis
import json
with open("replay.json", "w") as f:
    json.dump(env.toJSON(), f)
```

Three built-in agents are available by name: `"pass"`, `"random"`, and `"starter"` (a deterministic baseline).

## Set Up the Kaggle CLI

```bash
pip install kaggle
```

You'll need a Kaggle account. Download your API credentials at https://www.kaggle.com/settings/api
("Generate New Token").

**Recommended: API token file.** Save the token string to `~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative auth: `kaggle auth login` (OAuth browser flow), or `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`.

Verify:

```bash
kaggle competitions list -s "kaggriculture"
```

## Accept the Competition Rules

Before submitting, accept the rules on the Kaggle website: go to
https://www.kaggle.com/competitions/kaggriculture and click **Join Competition**.

Verify:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download kaggriculture -p kaggriculture-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an agent function.

```bash
# Single file agent
kaggle competitions submit kaggriculture -f main.py -m "Wheat loop v1"

# Multi-file agent — bundle into a tar.gz with main.py at the root
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit kaggriculture -f submission.tar.gz -m "Multi-file agent v1"

# Notebook submission
kaggle competitions submit kaggriculture -k YOUR_USERNAME/kaggriculture-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

```bash
kaggle competitions submissions kaggriculture         # check status, note the submission ID
kaggle competitions episodes <SUBMISSION_ID>           # list episodes once it's played some games
kaggle competitions episodes <SUBMISSION_ID> -v         # CSV output for scripting

kaggle competitions replay <EPISODE_ID>                 # download replay JSON
kaggle competitions replay <EPISODE_ID> -p ./replays

kaggle competitions logs <EPISODE_ID> 0                 # agent logs, player 0
kaggle competitions logs <EPISODE_ID> 1 -p ./logs        # agent logs, player 1

kaggle competitions leaderboard kaggriculture -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit kaggriculture -f main.py -m "v1"

# Check status / review / leaderboard
kaggle competitions submissions kaggriculture
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0
kaggle competitions leaderboard kaggriculture -s
```

## Submission Constraints

- ≤ 100 MiB total.
- 5 submissions/day; only your most recent 2 are active/scored.
- Files land in `/kaggle_simulations/agent/` — set imports accordingly.
- Runtime: 8 GiB HDD, 6.5 GiB RAM, 1.6 vCPUs.
- **No ingress/egress** during episode evaluation — your agent may not pull in or send out any information beyond the Submission + Environment (see `../RULES.md` §2.12).

## Files in This Kit

The docs live here (`docs/starter_kit/`); the actual agent code lives in [`../../src/`](../../src/) at
the repo root.

| File | What it is |
|---|---|
| [`../../src/farm_utils.py`](../../src/farm_utils.py) | Shared movement/geometry helpers (`step_toward`, `closest`, `act_or_move`, `shed_tiles`) used by `multi_crop.py` and `robust_agent.py`. Pure "how do I get from A to B" logic, no strategy. |
| [`../../src/quick_start_agent.py`](../../src/quick_start_agent.py) | Minimal wheat-buy/plant/water/harvest/sell loop agent from the competition Overview page. Good first read. |
| [`../../src/melon_maxxer.py`](../../src/melon_maxxer.py) | Verbatim copy of the example agent from `bovard/kaggriculture-getting-started`, walked through in [`TUTORIAL.md`](TUTORIAL.md). Ready to submit as-is (rename to `main.py` or submit directly). Deliberately not wired to `farm_utils.py` -- kept as a faithful copy of the official notebook. |
| [`../../src/multi_crop.py`](../../src/multi_crop.py) | Our first iteration on `melon_maxxer`: grows a rotation of WHEAT/CARROT/TOMATO/MELON instead of melon only, with per-crop sell thresholds and chunked sells. Single farmer, single quadrant, no animals/fertilizing. |
| [`../../src/robust_agent.py`](../../src/robust_agent.py) | Current best agent. Crash-safe (never raises), controls farmer + all hired hands via a shared priority task queue, dynamic crop scoring by live `$/tile/day`, and actually spends money: hires hands, buys land, builds coops/pastures, buys+places animals, fertilizes. Config-driven (`DEFAULT_CONFIG`) so thresholds can be tuned without touching logic. |
| [`../../src/run_match.py`](../../src/run_match.py) | CLI to run any two agents against each other locally (`python run_match.py robust_agent:robust_agent melon_maxxer:melon_maxxer`), with an optional `--render out.html` replay. |
| [`TUTORIAL.md`](TUTORIAL.md) | Walkthrough of the official getting-started notebook: environment setup, reading observations, building `melon_maxxer`, its known weaknesses, and how to submit. |
