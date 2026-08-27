from kaggle_environments import make as make_env


def play_episode(agent_a, agent_b, episode_steps=720, seed=None):
    config = {"episodeSteps": episode_steps}
    if seed is not None:
        config["seed"] = seed
    env = make_env("kaggriculture", configuration=config, debug=True)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    return final[0]["reward"], final[1]["reward"]
