"""Foundation of the distillation router: load a kaggle_environments episode replay,
extract each seat's per-step action sequence, and prove we can REPLAY it bit-exact.
This validates action alignment + the executor before any routing/feature work.

kaggle_environments replay layout: episode['steps'] is a list of 720 entries, each a
list [seat0, seat1], each {action, observation, reward, status}. The action stored at
steps[t][seat]['action'] is the action that seat took while observing steps[t]['observation']
(step 0 holds the initial PASS). To replay, at env step t we submit steps[t][seat]['action'].
We VERIFY the alignment by reconstructing rewards bit-exact rather than assuming it.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "opponents"))


def load_episode(path):
    return json.load(open(path))


def extract_actions(episode, seat, shift=0):
    """Per-step action list for `seat`. shift lets us test the off-by-one:
    action[t] = steps[t+shift][seat]['action']."""
    steps = episode["steps"]
    out = []
    for t in range(len(steps)):
        src = t + shift
        if 0 <= src < len(steps) and seat < len(steps[src]):
            out.append(steps[src][seat].get("action") or {"farmer": ["PASS"], "hands": [], "market": []})
        else:
            out.append({"farmer": ["PASS"], "hands": [], "market": []})
    return out


def make_replay_agent(actions):
    """Agent that returns the recorded action for the current step."""
    state = {"t": 0}
    def agent(obs, config=None):
        t = int(obs["step"]) if "step" in obs else state["t"]
        state["t"] = t + 1
        return actions[t] if 0 <= t < len(actions) else {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def reconstruct(episode, shift=0):
    """Replay both seats' recorded actions in a fresh env at the episode's seed;
    return (final_rewards, expected_rewards, match)."""
    from kaggle_environments import make
    seed = int(episode["info"].get("seed", episode["configuration"].get("seed", 0)))
    a0 = make_replay_agent(extract_actions(episode, 0, shift))
    a1 = make_replay_agent(extract_actions(episode, 1, shift))
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([a0, a1])
    got = [env.steps[-1][0].reward, env.steps[-1][1].reward]
    exp = [float(r) for r in episode["rewards"]]
    return got, exp, (got == exp)


if __name__ == "__main__":
    ep = load_episode(REPO / "input" / "97601003.json")
    print("expected rewards:", ep["rewards"], "seed:", ep["info"].get("seed"))
    for shift in (0, 1, -1):
        try:
            got, exp, ok = reconstruct(ep, shift)
            print(f"shift={shift:+d}: got={got} expected={exp} BIT-EXACT={ok}")
        except Exception as e:
            print(f"shift={shift:+d}: ERROR {e}")
