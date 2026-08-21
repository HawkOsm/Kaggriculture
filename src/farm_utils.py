"""Shared movement/geometry helpers for Kaggriculture agents.

These are pure functions with no game-strategy opinions -- deciding *where*
a unit should go and *what* it should do once there is each agent's job
(that's the part that actually differs between opponents/multi_crop.py and
robust_agent.py); this module only answers "which direction is that" and
"am I there yet". Every agent that walks a grid needs the same answer to
those two questions, so it lives here once instead of copy-pasted per file.

opponents/melon_maxxer.py is deliberately NOT wired to this module -- it's kept as a
verbatim copy of the official bovard/kaggriculture-getting-started notebook
(see TUTORIAL.md), so its own inline _step_toward is left alone.
"""


def step_toward(pos, target):
    """One cardinal-direction move from `pos` toward `target`, or None if
    already there. Greedy axis order (x before y) -- fine on an open grid,
    doesn't route around anything since locked tiles are passable anyway."""
    fx, fy = pos
    tx, ty = target
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def closest(pos, coords):
    """Nearest coordinate to `pos` by Manhattan distance, or None if
    `coords` is empty."""
    if not coords:
        return None
    fx, fy = pos
    return min(coords, key=lambda c: abs(c[0] - fx) + abs(c[1] - fy))


def act_or_move(pos, target, act_action):
    """The unit's action for this turn: `act_action` if already standing on
    `target`, otherwise a single step toward it."""
    if pos == target:
        return act_action
    step = step_toward(pos, target)
    return [step] if step else ["PASS"]


def shed_tiles(board_size):
    """The four shed-adjacent tiles, one per quadrant. The shed itself is
    never locked, so all four are always valid pickup/drop targets even
    though three of them start out on locked quadrants."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
