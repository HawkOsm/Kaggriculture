"""Compatibility shim -- re-exports from `adaptive_agent/`, this project's
live submission architecture as of 2026-08-27 (see docs/tests/LOG.md). Keeps
`import agent; agent.make_agent(...)` working for anything still using that
pattern. `adaptive_agent` below is a ready-to-run instance built with the
real tuned config (`adaptive_best_config.json`), for `run_match.py`'s
`agent:adaptive_agent` dynamic-load convention."""
import json
from pathlib import Path

from adaptive_agent.agent import make_agent  # noqa: F401

_CONFIG_PATH = Path(__file__).resolve().parent / "adaptive_best_config.json"
_TUNED_CONFIG = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else None
adaptive_agent = make_agent(_TUNED_CONFIG)
