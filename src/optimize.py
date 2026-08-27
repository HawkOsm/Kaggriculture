"""CLI entry point for `adaptive_agent`'s Optuna search/promotion pipeline
-- the real logic lives in `src/tuning/` (config.py/search_space.py/
simulation.py/scoring.py/verification.py/optimize.py). See
`src/tuning/optimize.py`'s module docstring for the full usage/design notes.

Usage:
    python src/optimize.py --n-trials 150 --n-jobs 8
"""
import multiprocessing

from tuning.optimize import main

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
