"""Local Optuna search over adaptive_agent's tunable surface (see
search_space.py). Frames tuning as beating a reigning champion: each trial's
candidate config plays a fixed opponent pool (real 2500+ Kaggle opponents +
two near-tier + the current champion) and is scored by risk-adjusted,
squashed reward margin (candidate - opponent), not raw reward. Every run
tries to beat whatever the previous run's winner was, so
adaptive_best_config.json only moves forward -- gated by BOTH a dedicated
candidate-vs-champion verification (the only thing that decides promotion,
not the search's own noisy mixed-pool trial value) AND an out-of-sample
holdout check against opponents excluded from every search pool.

Ported from the pre-restructuring optimize.py that tuned `robust_agent`
(same architecture, different config schema underneath -- see
search_space.py). Resumable: reruns with the same --study-name/--storage
continue the same Optuna study instead of starting over.

Usage:
    python src/optimize.py --n-trials 150 --n-jobs 8
    python src/optimize.py --n-trials 50 --n-jobs 1 --opponents kawa,pilkwang
"""
import argparse
import datetime
import json
import multiprocessing
import os

import optuna

from .config import (
    DEFAULT_STUDY_NAME, DEFAULT_STORAGE, OPPONENT_REGISTRY, LOG_PATH,
    BEST_CONFIG_PATH, HOLDOUT_ROTATIONS,
)
from .search_space import load_champion_config, sample_config
from .scoring import make_objective, DEFAULT_WIN_SCALE


class _TolerantFixedTrial(optuna.trial.FixedTrial):
    """A FixedTrial that supplies a sensible default for any parameter absent
    from the stored trial, instead of raising. This lets us reconstruct a full
    config from an OLD best trial after new knobs were added to the search space
    -- otherwise `sample_config` hits a suggest_*() for a key the old trial
    never recorded and the whole promotion step dies. Missing categoricals take
    the first choice; missing int/float take the low bound (both match how a new
    knob is expected to default before it has been searched)."""

    def suggest_categorical(self, name, choices):
        if name not in self.params:
            return choices[0]
        return super().suggest_categorical(name, choices)

    def suggest_int(self, name, low, high, **kwargs):
        if name not in self.params:
            return low
        return super().suggest_int(name, low, high, **kwargs)

    def suggest_float(self, name, low, high, **kwargs):
        if name not in self.params:
            return low
        return super().suggest_float(name, low, high, **kwargs)
from .verification import verify_candidate, HoldoutGate


def _worker(study_name, storage, n_trials, opponent_names, episodes_per_opponent, champion_config,
            risk_aversion, tail_quantile, tail_weight, win_scale, seeded):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(
        make_objective(
            opponent_names, episodes_per_opponent, champion_config,
            risk_aversion=risk_aversion, tail_quantile=tail_quantile, tail_weight=tail_weight,
            win_scale=win_scale, seeded=seeded,
        ),
        n_trials=n_trials,
    )


def _log_outcome(study, opponent_names, episodes_per_opponent, n_trials, n_jobs, had_champion_before,
                  verification_episodes, wins, losses, ties, verify_margin, promoted,
                  holdout_gate, holdout_detail=None, holdout_passed=True):
    date = datetime.date.today().isoformat()
    best = study.best_trial
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    if holdout_detail:
        rows = "; ".join(
            f"{n}: cand {cw}W {cm:+.1f} vs champ {hw}W {hm:+.1f}{' **REGRESSED**' if reg else ''}"
            for n, (cw, cm, hw, hm, reg) in holdout_detail.items()
        )
        holdout_txt = f" **Out-of-sample holdout check ({'PASSED' if holdout_passed else 'FAILED'} on rotation {holdout_gate.rotation_idx}): {rows}.**"
    else:
        holdout_txt = " Holdout check not run (candidate already rejected head-to-head, or --holdout-episodes 0)."
    entry = f"""## {date} — optimize.py: Optuna search over adaptive_agent config
- Command: `python src/optimize.py --n-trials {n_trials} --n-jobs {n_jobs} --opponents {','.join(opponent_names)} --holdout-rotation {holdout_gate.rotation_idx}`
- Result: {n_complete} complete trials, {n_pruned} pruned. Search's own best trial margin (noisy, {episodes_per_opponent} episodes x {len(opponent_names)}-opponent pool): {best.value:.4f}. **Dedicated candidate-vs-champion verification ({verification_episodes} episodes): {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}.**{holdout_txt}
- Notes: opponent pool was {opponent_names}, champion going in was `{'adaptive_best_config.json' if had_champion_before else 'adaptive_agent/config.py defaults (first run)'}`. Promotion requires BOTH gates: the dedicated candidate-vs-champion verification (wins > losses) AND no material regression against the active held-out opponents {holdout_gate.active_holdout} (rotation {holdout_gate.rotation_idx}), which are excluded from every search pool. {"Promoted to new champion (adaptive_best_config.json updated; remember to re-run python src/build_adaptive_submission.py)." if promoted else "Did NOT clear the bar -- adaptive_best_config.json left unchanged."} Best trial params: `{json.dumps(best.params)}`.

"""
    text = LOG_PATH.read_text()
    marker = "---\n"
    idx = text.index(marker) + len(marker)
    LOG_PATH.write_text(text[:idx] + "\n" + entry + text[idx:].lstrip("\n"))
    print(f"Logged outcome to {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100, help="total trials across all workers")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--episodes-per-opponent", type=int, default=2, help="must be even for balanced seating")
    parser.add_argument("--holdout-rotation", type=int, default=0, help="index of the holdout rotation to use")
    parser.add_argument(
        "--opponents", default="DEFAULT",
        help=f"comma-separated: {sorted(OPPONENT_REGISTRY)} or champion",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME,
                         help="Optuna study name; suffixed with the rotation index if left default")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds, across all workers")
    parser.add_argument("--verification-episodes", type=int, default=10,
                         help="dedicated candidate-vs-champion episodes deciding promotion")
    parser.add_argument("--holdout-episodes", type=int, default=4,
                         help="episodes per held-out opponent for the out-of-sample check; 0 disables it")
    parser.add_argument("--holdout-tolerance", type=float, default=0.5,
                         help="how much worse than the champion a candidate may be on a held-out opponent, as a "
                              "fraction of the champion's own margin, before promotion is blocked")
    parser.add_argument("--risk-aversion", type=float, default=0.2)
    parser.add_argument("--tail-quantile", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.3)
    parser.add_argument("--win-scale", type=float, default=DEFAULT_WIN_SCALE)
    parser.add_argument("--seeded", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.win_scale <= 0:
        args.win_scale = None

    rotation_idx = args.holdout_rotation % len(HOLDOUT_ROTATIONS)
    holdout_gate = HoldoutGate(rotation_idx)
    active_holdout = holdout_gate.active_holdout

    if args.opponents == "DEFAULT":
        # Pool refreshed 2026-08-27: rayk_c95/kaito/tran_hh/pilkwang/stevenleehans are
        # the opponents NOT already claimed by a HOLDOUT_ROTATIONS entry (see config.py) --
        # every one of them currently beats the champion, so this is real signal, not a
        # decisively-won-already pool like the one this replaced.
        all_candidates = ["rayk_c95", "kaito", "tran_hh", "pilkwang", "stevenleehans",
                           "selfvar_no_expand", "selfvar_panic_seller"]
        opponent_names = [op for op in all_candidates if op not in active_holdout] + ["champion"]
    else:
        opponent_names = [o.strip() for o in args.opponents.split(",") if o.strip()]

    leaked = sorted(set(opponent_names) & set(active_holdout))
    if leaked and args.holdout_episodes > 0:
        parser.error(
            f"opponents {leaked} are held out for the promotion gate (active holdout) and cannot "
            f"also be in the search pool. Either drop them from --opponents, or pass "
            f"--holdout-episodes 0 to deliberately disable the out-of-sample gate."
        )

    if args.study_name == DEFAULT_STUDY_NAME:
        args.study_name = f"{args.study_name}_rot{rotation_idx}"
    had_champion_before = BEST_CONFIG_PATH.exists()
    champion_config = load_champion_config()

    optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=len(opponent_names) * args.episodes_per_opponent,
        ),
        load_if_exists=True,
    )

    n_jobs = max(1, args.n_jobs)
    per_worker = [args.n_trials // n_jobs] * n_jobs
    for i in range(args.n_trials % n_jobs):
        per_worker[i] += 1

    print(f"Running {args.n_trials} trials across {n_jobs} worker process(es), "
          f"opponents={opponent_names}, episodes_per_opponent={args.episodes_per_opponent}")
    print(f"Active holdout rotation: {rotation_idx} (held out: {active_holdout})")
    print(f"Study name: {args.study_name}")

    if n_jobs == 1:
        _worker(args.study_name, args.storage, args.n_trials, opponent_names, args.episodes_per_opponent,
                champion_config, args.risk_aversion, args.tail_quantile, args.tail_weight, args.win_scale, args.seeded)
    else:
        procs = []
        for n in per_worker:
            if n <= 0:
                continue
            p = multiprocessing.Process(
                target=_worker,
                args=(args.study_name, args.storage, n, opponent_names, args.episodes_per_opponent,
                      champion_config, args.risk_aversion, args.tail_quantile, args.tail_weight,
                      args.win_scale, args.seeded),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = study.best_trial
    print(f"\nSearch's own best trial margin (noisy, {args.episodes_per_opponent} episodes x {len(opponent_names)}-opponent pool): {best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    merged = sample_config(_TolerantFixedTrial(best.params))

    print(f"\nRunning dedicated verification: {args.verification_episodes} episodes, candidate vs champion only...")
    wins, losses, ties, verify_margin = verify_candidate(merged, champion_config, args.verification_episodes, seeded=args.seeded)
    print(f"Verification: {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}")

    beat_champion = wins > losses

    holdout_passed, holdout_detail = True, {}
    if beat_champion and args.holdout_episodes > 0:
        print(f"\nRunning out-of-sample holdout check: {args.holdout_episodes} episodes each vs {active_holdout} (candidate and champion)...")
        holdout_passed, holdout_detail = holdout_gate.verify_holdout(
            merged, champion_config, args.holdout_episodes, args.holdout_tolerance, seeded=args.seeded
        )
        for name, (c_w, c_m, h_w, h_m, regressed) in holdout_detail.items():
            flag = "REGRESSED" if regressed else "ok"
            print(f"  {name:18s} candidate {c_w}W avg {c_m:+10.1f} | champion {h_w}W avg {h_m:+10.1f}  [{flag}]")

    promoted = beat_champion and holdout_passed
    if promoted:
        BEST_CONFIG_PATH.write_text(json.dumps(merged, indent=2))
        print(f"New champion written to {BEST_CONFIG_PATH} -- remember to re-run "
              f"`python src/build_adaptive_submission.py` to update submission/main.py")
    elif not beat_champion:
        print("Candidate did not beat the champion head-to-head (wins <= losses) -- adaptive_best_config.json left unchanged.")
    else:
        regressed_on = [n for n, v in holdout_detail.items() if v[4]]
        print(f"Candidate beat the champion head-to-head but REGRESSED against held-out opponent(s) {regressed_on} "
              f"-- adaptive_best_config.json left unchanged.")

    _log_outcome(study, opponent_names, args.episodes_per_opponent, args.n_trials, n_jobs, had_champion_before,
                 args.verification_episodes, wins, losses, ties, verify_margin, promoted,
                 holdout_gate, holdout_detail, holdout_passed)


if __name__ == "__main__":
    main()
