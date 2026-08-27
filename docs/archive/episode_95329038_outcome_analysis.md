# Episode 95329038 Outcome Analysis

Source link:
- https://www.kaggle.com/competitions/kaggriculture/leaderboard?submissionId=55614463&episodeId=95329038

Date analyzed:
- 2026-08-20

## What I could reliably extract
I could not retrieve raw replay JSON directly (public endpoint patterns and unauthenticated internal API calls returned 404/HTML). Instead, I extracted the structured state from Kaggle's embedded episode player for this exact URL and moved to the final step (720/720).

Verified episode facts from the player:
- Seed: 812665366
- Final step: 720 / 720
- Final in-game clock: Day 30 / 30, Turn 24 / 24
- Player 1 result: [Loss] Arman Tuganbaev 3050 (-3)
- Player 2 result: [Win] Ryo Hasegawa 3164 (+3)
- Final coins:
  - Player 1: 85363
  - Player 2: 91383
- Final money gap: +6020 coins in favor of Player 2

## Reverse-engineered outcome dynamics
These are inferred from observed board states and final values:

1. The match was rating-close but materially cash-separated.
- Elo/rating delta is only +/-3, suggesting similarly rated opponents.
- In-game economy still ended with a sizable +6020 coin lead for the winner.

2. Player 2 appears to have sustained superior economy throughput.
- At an earlier sampled playback point (~step 65), Player 2 already held a strong liquidity lead (278 vs 47 coins).
- That early momentum persisted to the final step.

3. Late-game market state strongly devalued animal outputs.
- Final displayed market prices included:
  - milk: 1
  - wool: 5
- This implies a late-game price crash in animal products, so outcomes likely depended on adaptation timing and conversion efficiency rather than static livestock commitment.

4. Final board composition suggests cleaner conversion and fewer dead turns for Player 2.
- Winner board still retained active productive structures with very high final cash.
- Loser side showed more low-value/idle-looking cells (including visible weeds), consistent with reduced marginal output in the closing phase.

## Confidence and limits
- High confidence: winner/loser, rating changes, seed, final-step stats, final coin totals.
- Medium confidence: strategic interpretation of why the gap emerged (inferred visually from board snapshots and market values).
- Unknown from accessible data: exact action-by-action decision trace, exact mapping of submissionId 55614463 to one side without authenticated replay metadata.

## Practical takeaway for this repo
For strategy tuning, this episode looks like a good example of:
- Surviving late-game commodity volatility.
- Converting early liquidity edge into sustained compounding.
- Avoiding endgame board deadweight when prices collapse.
