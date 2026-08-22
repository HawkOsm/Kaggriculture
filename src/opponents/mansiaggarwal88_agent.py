"""mansiaggarwal88_agent: local benchmark opponent.
"""

# Install the latest Kaggle environments package
# !pip install -U kaggle-environments

from kaggle_environments import make

# Create the environment!



def handle_market(private_info, my_farm):
    orders = []
    # If we have no wheat seeds and enough money, buy 1 seed
    if private_info["seeds"].get("WHEAT", 0) == 0 and my_farm["money"] >= 10:
        orders.append(["BUY_SEED", "WHEAT", 1])
    
    # If we have harvested wheat in our shed, sell it!
    wheat_in_shed = private_info["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        orders.append(["SELL", "WHEAT", wheat_in_shed])
        
    return orders



def handle_planting(tile, private_info):
    # If the tile is empty and we have a seed, plant it!
    if tile is None and private_info["seeds"].get("WHEAT", 0) > 0:
        return ["PLANT", "WHEAT"]
    return None



def handle_plant_care(tile, current_day):
    # Check if the tile has a plant
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = current_day - tile["planted_day"]
        
        if crop_age >= 2:  # Wheat is fully grown after 2 days
            return ["HARVEST"]
        if not tile["watered_today"]:
            return ["WATER"]
            
    return None



def my_wheat_farmer(obs):
    p = obs["player"]
    my_farm = obs["farms"][p]
    priv = obs["private"]
    
    fx, fy = my_farm["farmer"]
    tile = my_farm["tiles"][fy][fx]
    
    orders = handle_market(priv, my_farm)
    action = handle_planting(tile, priv)
    
    if action is None:
        action = handle_plant_care(tile, obs["day"])
    if action is None:
        action = ["PASS"]
        
    return {"farmer": action, "hands": [], "market": orders}

mansiaggarwal88_agent = my_wheat_farmer

