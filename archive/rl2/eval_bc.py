import sys
import multiprocessing as mp
from pathlib import Path
from kaggle_environments import make

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "opponents"))

from run_match import load_agent

OPPONENTS = [
    "pilkwang_agent",
    "cropdusta_97601003_agent",
    "prvsiyan_frontier_agent",
    "boatlee_v16_agent",
]

def run_match(args):
    seed, opp_name, seat, agent_path = args
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    
    agent = load_agent(agent_path)
    opp = load_agent(opp_name)
    
    agents = [agent, opp] if seat == 0 else [opp, agent]
    env.run(agents)
    
    # wait, load_agent returns a path or string if it is a file. 
    # Actually env.run will call it properly.
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    
    if seat == 0:
        win = r0 > r1
    else:
        win = r1 > r0
    return win

def run():
    agent_path = "src/rl2/bc_agent.py"
    total_wins = 0
    total_games = 0
    
    print(f"{'Opponent':<30} | {'Wins':<10}")
    print("-" * 45)
    
    pool = mp.Pool(18)
    for opp in OPPONENTS:
        args_list = []
        for seed in range(32):
            seat = seed % 2
            args_list.append((seed, opp, seat, agent_path))
            
        results = pool.map(run_match, args_list)
        wins = sum(results)
        total_wins += wins
        total_games += 32
        
        print(f"{opp:<30} | {wins}/32 ({wins/32:.2f})")
        
    print("-" * 45)
    print(f"{'Total':<30} | {total_wins}/{total_games} ({total_wins/total_games:.2f})")

if __name__ == "__main__":
    run()
