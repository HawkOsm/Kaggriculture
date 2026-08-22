# What other Kaggriculture competitors are actually doing — research notes

## How this was gathered, and where the research pivoted

The original ask was for winner write-ups from **older, concluded** Kaggle simulation
competitions (Kore 2022, Halite III, Lux AI, etc.) published post-competition on GitHub, on the
theory that someone has already solved "how much to risk given relative position" in a similar
game. That search hit a real wall: Kaggle's own discussion/notebook pages are JS-heavy and don't
render through this session's fetch tooling, and general web search didn't surface a specific,
quotable "here's the winning risk formula" writeup for those older competitions — those threads
exist, but not indexed at that level of detail.

What worked instead: **Kaggriculture itself has a live public-notebook culture**, and the
`kaggle kernels pull` CLI (already used successfully this session to source opponent agents)
retrieves full notebook source, including every markdown/prose cell — not just code. Four
`kaggle kernels list --search ... --competition kaggriculture` results turned out to be real
strategy essays, not just agent code, from competitors currently near or at the top of the live
leaderboard. That's arguably better source material than a concluded, different game: it's the
same rules, the same market mechanics, and current. Sections 1-4 below are from this pass.

**A second, more disciplined pass then went back to the original older-competition ask** (see
Section 5): rank candidate past Kaggle simulation competitions by mechanical similarity to
Kaggriculture first, pull the real top-30 leaderboard for the closest match via the `kaggle`
CLI (not scraped, not guessed), and check each name for public work before reading anything.
Kore 2022 (ranked closest by mechanic — a literal spend-now-to-grow-capacity-later resource
economy) turned out to have a very low open-sourcing rate among its top 10; Lux AI 2021 (ranked
second) had real hits starting at rank 1.

**No changes made to `CREDITS.md`** per instruction — these are research notes, not sourced
opponent code. Full kernel/leaderboard refs are given inline below for traceability if that
changes later.

---

## 1. Pilkwang Kim — [Kaggriculture: Structured Economic Policy](https://www.kaggle.com/code/pilkwang/kaggriculture-structured-economic-policy) (real closed-form math)

This is the closest thing to an actual answer to this session's "what's the real math" question.
Genuine formulas, not just heuristics.

**The objective is framed as a difference, explicitly, from the start** — matching this project's
own "only win/loss matters" finding independently:

$$\max_\pi\;\mathbb E_\pi\!\left[B_T^{(p)}-B_T^{(1-p)}\right]$$

**Production value has an explicit feasibility-before-deadline condition** — not just "does the
crop finish" but the full chain (crop cycle finishes, and its harvest, and its return-trip, and
its sale, all before T):

$$V_c(t)=\Pr(\text{completed})\,y_c\,\widehat p_c(t+g_c)-s_c-\lambda_W w_c-\lambda_L g_c
\qquad\text{admissible only if } t+24g_c+\tau_c^{\text{harvest}}+\tau_c^{\text{return}}<T$$

This is a direct, formal answer to this session's own "crop scoring doesn't discount for
season-end placement" finding (see the parameter discussion in this session, point 3) —
$\lambda_L g_c$ is exactly a *land opportunity cost per day occupied*, which is the missing term
our own `_crop_score` doesn't have. Livestock gets the same treatment with discounting
($\gamma^{u-t}$) over only its scheduled production turns.

**A genuine closed-form hiring-demand formula**, not a hand-tuned ratio:

$$H_t^*=\min\!\left(\overline H(d_t),\ \max\!\left[H_t^{\text{floor}}, \left\lceil\frac{J_t+2R_t}{7}\right\rceil\right]\right)$$

— desired hands as a function of due jobs $J_t$ *and* assets exposed to a second missed service
$R_t$ (double-weighted — protecting against a second miss matters more than the backlog count
alone), capped by a maturity-dependent ceiling that itself steps up on day 21. Ordinary hiring
additionally requires *residual* liquidity after the hire: $B^{M,\text{res}}_{t,k}\ge\max(20,
3F_{H_t})$ — a reserve requirement, but one keyed to the *marginal hire's own cost* ($F_{H_t}$,
the Fibonacci wage), not a flat dollar figure independent of what's being bought.

**Market price-impact has a proven symmetry result directly relevant to `opponent_race_discount`**:
racing to sell $x$ units ahead of a non-reacting opponent block of $q$ units moves *both* banks by
the exact same rectangle of marginal price steps — $\Delta_{\text{ours}}=\Delta_{\text{theirs}}$
— so the gain to the difference-objective is exactly *twice* the price impact, computable in
closed form with no simulation:

$$\Delta\!\left[B_T^{(p)}-B_T^{(1-p)}\right]=2\sum_{i=0}^{x-1}\sum_{j=0}^{q-1}\delta_r(I+i+j)$$

This formalizes exactly what our own race-discount mechanism is trying to do heuristically. **One
important caveat this notebook proves and ours doesn't check**: the self-impact sell-ordering
rule (rank sells by how much each one's own quote would decay) is only sound when *we're the sole
seller of that product that turn*. When the opponent sells the same product at the same time, the
rule **inverts** — leading the decline just lets their (often larger) block follow immediately at
the reduced price, moving the difference-objective against us. The notebook's own policy
deliberately does *not* reorder its sell block for this reason, because the current field of top
agents run near-identical production and collide on the same products/turns constantly. Worth
checking whether our own `max_sell_chunk`/ordering logic has this blind spot.

**Shared-market demand has a documented "knee"**: the scarcity-side price curve is piecewise —
gentle below a threshold, then quadratic and unbounded above it ($f^-_r(x)=x/T_r+\gamma\max(0,
x/T_r-1)^2$). A product can look calm right up until a cumulative deficit crosses $T_r$, then
"runs away." Town-shop demand is drawn *with replacement* from a fixed catalogue, so seasonal
demand for a given product is a random variable, not a constant — "a plan fixed before the draw is
therefore sized against a demand it has not observed."

---

## 2. Rayk Kretzschmar — [Kaggriculture: notes from replay hunting](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta) (methodology, not math)

A working diary of an actual competitive iteration process (already an opponent in our pool,
`rayk_c95_agent`), most valuable for **evaluation discipline and one directly validating
negative result**.

**Sample sizes far larger than this session has been using.** Their protocol: play both seats,
treat a loss to the previous-best or to the unmodified parent as a veto *regardless of aggregate
win rate*, use multiple genuinely different opponent families, and freeze parameters before one
untouched final block. Concrete numbers from the diary: a "held-out validation" used **900 games
over six new seeds**; a final tournament used **918 games, zero failures**, with results fit to a
**Bradley-Terry model** (pairwise-comparison rating, not raw win/loss counts) rather than
aggregate win percentage. This session's own promotion gate uses **10 episodes** — two orders of
magnitude smaller. The earlier "the relative-wealth mechanism's 4-episode win evaporated at 10
episodes" incident this session is exactly the failure mode this competitor's protocol is
explicitly designed to catch.

**A land-expansion negative result that independently confirms this session's own finding.**
Quoting directly: *"Top public leaders almost always take NE+SW, rarely SE."* They ran a dedicated
screen — three implementation families for a fourth quadrant, gated multiple ways (always, only
when behind, only when not ahead, opponent-reactive) — against **450 games**: *"every new
four-quadrant variant lost 10-0"* to their best agent, including reconstructed fully-productive
routes from a different strong public competitor, by margins of **14,608 to 36,980**. Their
diagnosis: reusable hands become idle too late to service the new land before day-refresh,
extra crew costs more than incremental output, unlocking more land exposes more empty tiles to
random weed spawns, and expanded production deepens the shared-market glut. This is an
independent, much larger-scale confirmation of this session's own repeated "raising
`max_structures`/land alone regresses" finding (`docs/tests/IDEAS_TRIED.md`) — not a fluke of our
own dispatch limitations, a structural property of this game at this meta.

**A specific tactical insight about market order position**: a real loss pattern was the opponent
buying up cheap WHEAT before this agent's own feed-purchase order resolved in the same turn,
leaving too little for a full feed order and starving an animal. The fix wasn't "buy more" — it
was moving the *existing* order to the *first* slot in the turn's order list. Order position
within a turn is itself a resource; something our own `_market_orders` doesn't currently reason
about (orders are built in a fixed code sequence, not deliberately prioritized by urgency).

**Win-probability, explicitly, over margin**: *"maximize the chance of winning the matchup, not
the size of an already-won bank."* Their own diary shows an agent (C70) that was already crushing
weak opponents but plateaued below rating 3000 — the losses were narrow, close games against a
handful of specific strong opponents, and *increasing the margin against already-beaten opponents
could not move a win/loss-only rating*. Directly the same lesson as this session's "coin margin
doesn't matter" framing, from an independent source with the receipts to back it.

---

## 3. Roman Rozen — [Barnyard Economist](https://www.kaggle.com/code/romanrozen/strong-statr-baseline-agent-lb-950) (architecture, not math)

Already an opponent in our pool (`romanrozen_agent`). The notebook's central design principle is
**separation of concerns between season-scale and turn-scale decisions** — worth naming directly
against this session's own "big picture vs narrowly-bounded local knobs" discussion:

> *"A season route owns capital and logistics; runtime owns only what the environment can
> invalidate. Every attempt to blur that line made the agent less predictable without making it
> richer."*

Concretely: a full-season route decides capital allocation, labor, and planned sales *once*, as a
coherent whole; only narrow, tightly-scoped runtime patches (a blocked tile from a weed, sell-slot
ordering, a bounded/repaid timing shift on a premium sale near a near-clone opponent) are allowed
to touch it turn-by-turn, and even those carry an explicit invariant ("blast radius") — e.g. a
timing shift *must* be repaid: exactly the quantity moved earlier is subtracted from the later
sale it would have come from, so a timing optimization can never silently become a production
change.

**A phased view of the season as a pipeline, not 720 independent turns**: BOOTSTRAP (day 0-4,
first labor/animals) → EXPAND (day 4-9, land/routes/crops) → ROTATE & COMPOUND (day 9-20,
harvest/care/reinvest) → PROTECT VALUE (day 20-28, flow guard/sale order) → CASH closeout (day
28-30). Their own framing: *"the useful question is not 'what action pays most now?' but 'which
action improves the amount of cash that can still reach the bank before turn 720?'"* — a direct,
qualitative statement of exactly the ROI-over-remaining-time idea from this session's parameter
discussion (point 3), applied to every decision category at once rather than crop-scoring alone.

**One explicitly-named unexploited edge, directly relevant to our own `_opponent_profile`/
`_standing_asset_value` mechanisms**: *"the opponent's farm is fully visible in
`obs['farms'][1-player]` — their tiles, crops and planting days. A large melon block planted on
day 0 is a dated announcement of a sale on day 10, and almost nothing in public agents reads it."*
This is a stronger claim than what our own opponent-profile scan currently does — we already read
tile state for concentration/scale and (with `_standing_asset_value`) full-yield valuation, but
this notebook is specifically calling out *planting day → dated future sale* as underexploited,
which is closer to genuine opponent-schedule forecasting than a static snapshot valuation.

---

## 4. Kaito Fukami — [v27 Midgame Meta Reset](https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset) (confirms the "fixed-route" finding, plus a stricter eval protocol)

Already an opponent in our pool (`kaito_agent`), currently near the top of the live leaderboard
(167 votes on this notebook, highest of everything pulled).

**Independently confirms this session's own "MAJOR REFRAME" finding** (that kawa/prvsiyan are
pre-solved fixed-route scripts, not live strategies) for a *third* top agent: this notebook's own
attribution card states the 719-action backbone is *"credited as Ezzzzzekki's observable public
replay behavior... not claimed as hidden-source recovery or as a newly invented production
schedule"* — i.e. kaito's own agent, too, is built by distilling another team's replayed action
sequence into a fixed route, with only a thin reactive layer (weed repair, sell-slot ordering) on
top. The notebook also states plainly that **26 of the current top 30 teams share the same single
opening signature** (1 COW / 4 SHEEP / HIRE4) — the public meta has converged hard on one shared
skeleton, and competitive edge has moved almost entirely into *continuation and market timing*,
not what to build.

**A meaningfully stricter evaluation protocol than this session has used.** They explicitly
separate a "development outer" holdout (used to pick between architectures — contaminated by
having informed the choice) from a one-time "strict-future" gate: only episodes with
`EpisodeId` greater than a fixed cutoff timestamped *after* the policy was frozen count as the
real promotion test, specifically to stop a policy from being selected against data it could have
been indirectly fit to. Their own citation: [Beating Your Own Best Agent Is The Wrong
Test](https://www.kaggle.com/code/dariushafshar/beating-your-own-best-agent-is-the-wrong-test) —
a title that states outright what this session's own promotion gate does not currently guard
against (every verification this session has been champion-vs-candidate only, always against
data/opponents already used to select the candidate).

**A rejected idea worth knowing before re-trying it**: a per-seat routing mechanism (different
policy for seat 0 vs seat 1) looked like a real improvement on their inner validation split, but
only fixed 1 of 3 real historical losses, while a single fixed non-routed policy fixed all 3 with
comparable aggregate performance — *"seat-specific validation performance was mistaken for a
generally useful seat mechanism."* Worth remembering given how much of this session's own
methodology (alternating-seat testing) implicitly treats seat symmetry as a given.

---

## 5. Older concluded competitions: Kore 2022 and Lux AI 2021 (real leaderboards, mixed hit rate)

**Method**: candidate past competitions were ranked by mechanical similarity to Kaggriculture
first — Kore 2022 (spend-Kore-to-spawn-ships-now vs bank it, 1v1, terminal comparison) ranked
closest, Lux AI (resource collection + city investment under day/night survival pressure) second,
Halite third, everything else (Connect X, Hungry Geese, Rock Paper Scissors, Google Research
Football) too mechanically distant to be useful for this specific question. Real top-30
leaderboards were pulled via `kaggle competitions leaderboard <slug> -d` (CSV, not scraped), then
each name checked for public Kaggle kernels (`kaggle kernels list --user <name>`) and GitHub.

**Kore 2022 (top 10 checked): low hit rate.** Only 1 of the top 10 usernames had any public Kaggle
kernel at all, and it was unrelated (a different, later competition). This tracks: Kore 2022 was a
*scored, currently-live* competition when these agents were active — publishing a working
competitive agent while ranking still matters actively hurts your own position, unlike
Kaggriculture's current culture of public sharing (see Section 4's note that 26/30 top teams now
share one converged opening — the public meta itself may be *why* sharing feels lower-cost there
now). The confirmed #1 solution (Harm Buisman) has an official Kaggle "writeup" page, but that
page — like every other Kaggle page type tried this session (notebooks, discussions) — is
JS-rendered and unreadable by any fetch tool available here.

**Lux AI 2021, rank 1 (Toad Brigade, team member `pressman1`) — real GitHub hit, and a genuinely
different kind of answer.** [`IsaiahPressman/Kaggle_Lux_AI_2021`](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)
is real, public, and fetchable (GitHub READMEs are static, unlike anything on kaggle.com). The
finding itself is a surprise relative to Sections 1-4: **the winning approach was pure deep RL
(self-play, FAIR's IMPALA with UPGO + TD-lambda + KL-divergence regularization against a frozen
teacher model), not a hand-built economic formula.** Their own account: *"the RL approach began to
beat the rules-based one... and seemed to be improving monotonically without any signs of
plateauing, so the rules-based agent was abandoned."* No explicit spend-vs-save threshold, no
opponent-position risk model — policy stability (via the frozen-teacher KL term) substituted for
what a formula-based agent would need an explicit reserve/risk calculation for. One value-function
observation worth keeping: their trained agent stayed *"very confident about victory even when it
is far behind in city count,"* i.e. it learned to weight territory/position over raw resource
count — a learned analog to this project's own win/loss-over-margin distinction, arrived at by
training rather than by an explicit formula. Directly relevant to this session's earlier decision
to deprioritize RL (an RL run this session was "decisively worse" than the rule-based agent) —
this is real evidence RL *can* win this class of game, just apparently not yet with this project's
training setup/budget.

**Lux AI 2021, rank 4 (Team Durrett, `bomac1` + `tairatsuchiya`) — a real dead end.** This pair is
notable for being a *repeat* strong competitor: `bomac1` also placed 3rd in Kore 2022. Despite
strong results in two different games, neither team member has a findable GitHub account
(`github.com/bomac1` returns a genuine 404) or public Kaggle kernel under a discoverable name.
Not every strong, repeat competitor publishes — worth remembering before assuming a leaderboard
position implies a checkable trail.

**Lux AI 2021, rank 6 (`zaharch`/"nosound") — a real, portable methodology finding.**
[Bradley-Terry rating system for Kaggle sim comps](https://www.kaggle.com/code/zaharch/bradley-terry-rating-system-for-kaggle-sim-comps)
is a dedicated notebook (not buried in a diary like Section 2's mention of the same method) that
states the actual problem with Kaggle's live rating system directly: *"the influence of an episode
only takes the history into account and not the future"* — i.e. it's a sequential, path-dependent
update, so the same underlying strength can produce different displayed ratings depending purely
on the order opponents were faced in (this project's own session log has an example of exactly
this: two byte-identical `main.py` submissions on Kaggriculture's ladder showing ratings of 1210.1
and 2182.2, purely from different opponent/seed history — see `rayk`'s notebook, Section 2). The
fix is a proper one-shot maximum-likelihood fit over every recorded game at once, not a running
update. The actual model, in full, is short enough to be directly portable:

```python
# R: per-submission rating, init 1000 (Elo scale). idx_i/idx_j: winner/loser indices per game.
Ri, Rj = R[idx_i], R[idx_j]
Qi, Qj = 10 ** (Ri / 400), 10 ** (Rj / 400)
loss = torch.log(Qi + Qj) / torch.log(torch.tensor(10.0)) - Ri / 400   # -log-likelihood, i beat j
# gradient descent over ALL games at once; re-center each step (R -= R.mean() - 1000)
# since Bradley-Terry ratings are identified only up to a global additive constant
```

This is a genuine, small, working alternative to a simple win-count gate — worth weighing against
this project's own 10-episode `wins > losses` promotion gate (see the Evaluation methodology
connection below).

**Lux AI 2021, rank 9 (`vitoque`) — checked, no additional content.** Confirms an imitation-learning
approach (title: "Lux AI with IL (decreasing learning rate)") but the notebook is pure training
code with no markdown/prose cells — nothing further to extract beyond the method name itself.

---

## Direct connections back to this session's parameter discussion

- **Point 2 (cash reserve vs risk analysis)**: pilkwang's hire-reserve condition
  ($B^{M,\text{res}}\ge\max(20,3F_{H_t})$) ties the reserve requirement to the *marginal
  purchase's own cost*, not an independent flat floor — closer to what this session's discussion
  was reaching for than the current `money_reserve` + bounded `risk_scale` design.
- **Point 3 (ROI / every decision affects the big picture, not just crops)**: pilkwang's
  $\lambda_L g_c$ land-opportunity-cost term and romanrozen's season-phase framing are both real,
  independently-arrived-at answers to exactly this — a shared land/time opportunity cost applied
  uniformly across categories, not a special case for crops.
- **Point 1 (relative-wealth mechanism not touching selling)**: pilkwang's Section 8 ("the
  objective is a difference, not a level") gives the formal reason *withholding a sale* only helps
  when we're not also the larger supplier in that window — a real candidate mathematical basis for
  how a wealth-position-aware sell adjustment should actually be shaped, if that connection gets
  built later.
- **The land-expansion finding** (point raised across several `IDEAS_TRIED.md` entries this
  session) has real independent confirmation at far larger sample size (450 games) from a
  currently-competitive agent, not just this project's own smaller-scale tests.
- **Evaluation methodology** is the single biggest gap this reading surfaced that isn't about any
  one parameter: every other source here runs the promotion decision on hundreds to low-thousands
  of games, with an explicit veto-on-any-real-loss rule and (in kaito's case) a genuinely
  never-seen-before holdout slice. This session's 10-episode gate is real progress over hand-picked
  configs, but it's the weakest link relative to what competitive practice here looks like.
  Section 5's Bradley-Terry finding gives a concrete, small, working alternative to the raw
  `wins > losses` count this project's gate currently uses — worth prototyping against the
  existing verification episodes before deciding whether it's worth adopting.
- **RL isn't necessarily a dead end** (Section 5, Lux AI rank 1) — this session deprioritized RL
  after one run underperformed the rule-based agent, but a real Kaggle sim-competition winner beat
  a comparably-sophisticated rules-based agent with self-play RL once training matured. Not a
  reason to restart RL training now, but worth not treating the earlier result as a permanent
  verdict on the approach itself, only on that one training run's setup/budget.

---

## 6. Scripted openings in past competitions — GitHub survey (2026-08-22)

Searched real repositories via `gh` (not blog summaries) for open-sourced solutions to Kore 2022,
Lux AI S1/S2, Halite IV, Hungry Geese, ConnectX, RPS and the Santa series, looking specifically
for hardcoded/scripted early-game phases blended with reactive play. Collected source for the
solutions found lives in `../input/` with per-solution `PROVENANCE.md`; index at
`../input/index.json`.

Based on a direct search of Kaggle simulation competition repositories via the GitHub CLI, here is an analysis of "opening book" or hardcoded start sequence strategies used by open-source solutions.

### Explicit Opening Books Found

### ConnectX
We observed explicit hardcoded opening sequences seamlessly blended with reactive mid/late-game logic in multiple top ConnectX repositories:

1. **[LeonaRaging/connectx](https://github.com/LeonaRaging/connectx)**
   - **Pattern:** Uses a strict, phased hybrid approach based on the exact move count. 
   - **File Reference:** In [`src/main.py`](https://github.com/LeonaRaging/connectx/blob/main/src/main.py), if `moves < 8`, the agent queries a massive 9MB pre-computed lookup table ([`src/opening_book.py`](https://github.com/LeonaRaging/connectx/blob/main/src/opening_book.py)). Once moves exceed 8, it hands control to an AlphaZero network, and eventually (moves > 17) to a deeper Alpha-Beta Minimax solver.
   - **Author Notes:** The author intentionally designed this to "optimize for different game phases," ensuring rapid, mathematically proven correct responses to common opening traps before engaging computationally expensive searches.

2. **[buiducanh220802/connectX](https://github.com/buiducanh220802/connectX)**
   - **Pattern:** Also relies heavily on an "In-Memory Opening Book" statically embedded as constants.
   - **File Reference:** The opening database is stored in a C++ header (`book_data.hpp`).
   - **Author Notes:** The README explicitly highlights this as "Layer 1" of their architecture. The author notes that hardcoding the WeakC4 graph guarantees an opening execution under 50 microseconds. This prevents the agent from triggering Kaggle Sandbox's strict 2-second timeout on the critical first moves and evades early-game state traps.

### Parameterized Early-Game Logic Found

### Halite IV
Rather than rigid action sequences (like action tables), we observed parametrized, tunable rule modifications for the early game.

1. **[ttvand/Halite](https://github.com/ttvand/Halite)** (Winning Solution)
   - **Pattern:** The agent does not memorize a sequence of absolute positions. Instead, it alters its heuristic behavior during the opening phase using global turn counters.
   - **File Reference:** Inside the `Logic` directory (e.g., `main_rule_based.py` and `rule_actions_v2.py`), specific parameters like `early_game_return_boost_step = 50` and `early_game_return_base_additional_multiplier = 0.1` are heavily utilized.
   - **Author Notes:** These values were not guessed; they were treated as searchable parameters and tuned in a "stable opponents pool" simulation to find the most robust constants for early-game resource caching without being too brittle against unpredictable opponent spawns.

### No Scripted Openings Found (Pure RL / Heuristics)

For several of the more complex environments, winning or top-tier repositories leaned entirely into pure Deep Reinforcement Learning (RL) from step zero, with no hardcoded openings detected during codebase inspection.

1. **Lux AI (Season 1)**
   - **Repo:** [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) 
   - **Findings:** A search across the codebase for opening scripts/heuristics yielded nothing. The author's extensive writeup explicitly notes the agent was trained "starting from a random initialization" and developed its own opening strategies (like scorching the earth or rushing forests) entirely via RL, rather than relying on hardcoded coordinates or early-turn overrides.

2. **Hungry Geese**
   - **Repo:** [takedarts/hungry-geese](https://github.com/takedarts/hungry-geese) (5th Place)
   - **Findings:** Inspected `src/agent.py` and models. The solution relies exclusively on PyTorch-driven CNNs continuously evaluating state from the very first frame. No explicit rule-based opening books were implemented.

3. **Rock Paper Scissors**
   - **Repo:** [thisisbowen/RPS-Kaggle-1st-Place-Solution](https://github.com/thisisbowen/RPS-Kaggle-1st-Place-Solution) (1st Place)
   - **Findings:** The agent utilizes a dynamically updating Multi-Armed Bandit (MAB) framework to select from a pool of sub-agents (decision trees, historical bots) continuously. It does not play a hardcoded deterministic opening sequence.

### Conclusion for Kaggriculture
If the first 3 days of Kaggriculture are strictly deterministic, the **ConnectX approach** (rigid table/graph up to the non-deterministic transition point) is heavily validated by prior art, as it preserves compute budgets and completely eliminates timeout risks during the critical setup phase. Conversely, the **Halite IV approach** of tuning parameterized multipliers for the early-game could provide a more robust fallback if your bot needs to remain flexible against slightly different map seeds.

---

## 7. Opening-book theory (general game AI) — background, weakly sourced

**Reliability caveat:** this section was produced by a research agent that returned only vague,
unlinked citations ("chess/Go literature discussions", "Game AI Pro") despite being asked for
links. Treat it as a useful vocabulary and hypothesis source, **not** as verified research. The
concrete, checkable findings are in sections 5 and 6 above, which cite real repositories.

Research notes on prior art for scripted opening windows in game AI.

### 1. Mechanism: How Engines Use and Leave Opening Books
In classic game AI (such as traditional chess and Go engines), an **opening book** is a pre-calculated database of established, optimal move sequences. 
- **Use:** Engines look up moves instantaneously during the opening phase. This saves computational time, avoids deep search trees that are too broad early on, and guarantees the engine reaches a strong middlegame position.
- **"Leaving Book":** An engine leaves its opening book when it reaches a board state not present in its database (a "novelty"). This usually happens because an opponent deviates from known theory or the game naturally progresses beyond the book's depth (typically 15–30 moves).
- **The Handoff:** Once out of book, the engine transitions from instantaneous retrieval to live calculation. It begins using its search algorithms (e.g., Alpha-Beta pruning, Monte Carlo Tree Search) and its heuristic evaluation function to analyze the state space dynamically.

*Sources: Concept generally established in computer chess literature; supported by explanations of engine transitions (e.g., Chess.com, Lichess engine behavior).*

### 2. Pitfalls and Failure Modes of Opening Books
While opening books guarantee strong early positions against optimal play, they introduce specific vulnerabilities:
- **"Book Rigidity" and Brittleness:** Engines may lack an understanding of the *principles* behind the opening moves. If an opponent plays a technically sub-optimal but tactically tricky novelty (or an unexpected human-like deviation), the engine is suddenly thrust into unfamiliar territory and may fail to adapt properly, playing predictably or missing the opponent's inaccuracies.
- **The Win-Rate Trap:** Relying on static historical win-rates from databases can lead the AI to evaluate a book move as "good" based on high-level positional play, while missing specific local tactical refutations that live search would have caught. A book can confidently lead the AI into a position where it is tactically worse off than if it had evaluated from scratch.
- **Echo Chambers and Blind Spots:** If an opening book is generated via self-play (common in Go), model deterioration can occur. The AI might favor lines simply because they worked against its own earlier versions, creating a narrow repertoire and blind spots against unconventional strategies.

*Sources: AI chess/Go failure mode discussions; model deterioration in self-play (e.g., AlphaZero/Stockfish community analysis).*

### 3. Parameterized and Policy-Driven Opening Books
Modern engines have moved away from literal fixed sequences toward parameterized, policy-driven early-game heuristics that generalize better against deviations.
- **Stochastic Move Selection (Temperature):** Engines like AlphaZero do not use a traditional hardcoded opening book. Instead, they use a **temperature parameter ($\tau$)** during search. For the first ~30 moves, the temperature is set high ($\tau=1$), making move selection stochastic rather than deterministic. This creates a probability distribution over strong moves, forcing a varied, parameterized repertoire rather than a single brittle line. 
- **Parameterizing High-Level Rules:** In strategy games (like RTS or economy simulations), developers parameterize high-level decision rules (e.g., build orders, resource allocation percentages) rather than specific low-level actions. These parameters can be tuned via RL or evolutionary algorithms. This provides a strong early-game bias while leaving the execution to reactive lower-level policies, preventing the brittleness of a fixed sequence.

*Sources: AlphaZero paper (Silver et al.); Game AI Pro on parameterization in RTS games.*

### 4. Reinforcement Learning / Hybrid Agent Terminology
In literature regarding distinct behavioral phases (like a deterministic opening window vs. a reactive mid-game), several terms of art apply:
- **Warm-Starting / Heuristic Initialization:** Initializing an agent with prior knowledge (such as a heuristic rule or a scripted PID controller) to jumpstart its early performance before allowing RL to take over. This avoids the "cold start" problem and provides a safe baseline.
- **Phase-Based Reinforcement Learning:** A specific architectural pattern where tasks are decomposed by game phase (e.g., Opening, Midgame, Endgame). A "Phase Detector" evaluates the game state and shifts the agent's reward function, strategy, or active policy accordingly. This perfectly maps to having a distinct policy for a deterministic opening window.
- **Hierarchical Control (Hierarchical RL / Task Networks):** Decoupling high-level strategy (the "Commander") from low-level execution (the "Specialist"). The agent can be given a high-level opening directive (e.g., "aggressive expansion"), which is then executed reactively by lower-level policies, combining an opening script with reactive adaptability.

*Sources: Literature on Phase-Based RL (e.g., arXiv, MDPI papers on game AI); Heuristic-Guided RL research.*
