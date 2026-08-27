from .config import get_config
from .state import _role_plan
from .dispatch import _unit_actions
from .market import _market_actions, _schedule_market_adjustment

class PilkwangDispatcher:
    def __init__(self, config=None):
        self.config = config

    def decide(self, obs):
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        if not (0 <= player < len(farms)):
            return {"farmer": ["PASS"], "hands": [], "market": []}
            
        farm = farms[player]
        private = obs.get("private", {}) or {}
        
        roles = _role_plan(obs, self.config, farm)
        field = _unit_actions(obs, self.config, farm, private, roles)
        market = _market_actions(obs, self.config, farm, private, roles, field)
        market = _schedule_market_adjustment(obs, self.config, farm, private, market)
        
        return {
            "farmer": field["farmer"],
            "hands": field["hands"],
            "market": market,
        }

def make_agent(config=None):
    dispatcher = PilkwangDispatcher(config)
    def agent(obs):
        try:
            return dispatcher.decide(obs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            farms = obs.get("farms", []) if hasattr(obs, "get") else []
            player = int(obs.get("player", 0)) if hasattr(obs, "get") else 0
            hand_count = (
                len(farms[player].get("hands", []) or [])
                if 0 <= player < len(farms)
                else 0
            )
            return {
                "farmer": ["PASS"],
                "hands": [["PASS"] for _ in range(hand_count)],
                "market": [],
            }
    return agent

pilkwang_agent = make_agent()
