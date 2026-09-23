"""Build submission/main_hanif_plus.py: hanif's public agent + our measured modifications.

Base: hanifnoerrofiq / "A Wonderful Life" (public Code-tab notebook, Apache-2.0 lineage, notices
retained in the file; credited in CREDITS.md). Decoded copy: .claude/scratch/decoded_agents/hanif_current.py.

Our modifications are source-level patches, each asserted to apply exactly once, selected by paired
local sweeps (.claude/scratch/hanif_route/) against prvsiyan's current public agent, a hanif mirror
(proxy for the many hanif forks on the ladder) and guard opponents, then re-validated on fresh seeds.
See PATCHES below for the list and the out-of-sample numbers.

Usage: python src/distill/build_hanif_plus.py [base.py] [out.py]
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
base_p = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / ".claude/scratch/decoded_agents/hanif_current.py"
out_p = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "submission/main_hanif_plus.py"

SET = ("_SETTINGS={'hand_align': True, 'weed_repair': True, 'sell_lead': True, 'budget_guard': False, "
       "'room_guard': False, 'clamp_sells': False, 'dead_stock': False, 'terminal_liquidation': False, "
       "'front_run': False}")


def replace_once(src, old, new):
    assert src.count(old) == 1, f"patch anchor not unique/present: {old[:60]!r}"
    return src.replace(old, new, 1)


def set_const(src, name, value):
    pat = re.compile(r"^" + re.escape(name) + r" *= *[^#\n]*", re.M)
    assert len(pat.findall(src)) == 1, f"constant {name} not unique/present"
    return pat.sub(f"{name} = {value!r}", src, count=1)


# (name, patch fn). Fresh-seed validation (seeds 25000-25029, 100 paired contexts) of all four together:
#   vs prvsiyan  mean margin +618 (better in 27/30), wins 6/30 vs base 1/30
#   vs hanif mirror  28 wins / 2 losses of 30 (base: 2/2, 26 ties)
#   guard (roxy, kawa, 2 top tapes)  39/40 wins, unchanged; mean margin -466
# Second fresh block (seeds 26000-26029): vs prvsiyan +543 (better in 26/30), mirror 23-7, guard 39/40.
# Refinements tested on top and rejected as within noise: race horizon 60/72 (cap raised), feed days 0,
# sale-race from hour 20, _OR2_CAP 20, _SR_MARGIN 8. Also rejected: room_guard (no-op on top),
# budget_guard (never better), terminal_liquidation (byte-identical), TomatoInsteadOfCow (-5.2k/game).
PATCHES = [
    # Clamp each SELL to projected stock while keeping the order slot (hanif ships the layer disabled).
    ("clamp_sells", lambda s: replace_once(s, SET, SET.replace("'clamp_sells': False", "'clamp_sells': True"))),
    # Default market-race reservation horizon 40 -> 48 (= V9_RACE_MAX): reserve planned sales earlier.
    ("race_default_48", lambda s: set_const(s, "V9_RACE_DEFAULT", 48)),
    # Feed-reserve lookahead 2 days -> 1: less stock withheld from sale.
    ("feed_days_1", lambda s: set_const(s, "_CA_FEED_DAYS", 1)),
    # Day-end sale-race hours (22, 23) -> (21, 22, 23): start the pre-drop sale race an hour earlier.
    ("sr_hours_21", lambda s: set_const(s, "_SR_HOURS", (21, 22, 23))),
]

src = base_p.read_text()
for name, fn in PATCHES:
    new = fn(src)
    assert new != src, f"patch {name} had no effect"
    src = new

banner = ("# HawkOsm modifications on top of the public 'A Wonderful Life' agent (hanifnoerrofiq).\n"
          "# Patches (src/distill/build_hanif_plus.py): " + ", ".join(n for n, _ in PATCHES) + ".\n"
          "# All upstream attributions and Apache-2.0 notices below are retained unchanged.\n")
src = banner + src
assert src.rstrip().endswith('agent = globals().pop("agent")') or "agent=globals().pop('agent')" in src.rstrip().splitlines()[-1], \
    "Kaggle loader picks the last callable: file must end by exporting agent"
out_p.write_text(src)
print(f"wrote {out_p} ({len(src)} bytes), patches: {[n for n, _ in PATCHES]}")
