"""umutdorukztrk_agent: local benchmark opponent.
"""
# -------------------------------------------------------------------------
# 1. CONSTANTS
# -------------------------------------------------------------------------
SHED_CAPACITY = 100
MAX_MARKET_ORDERS_PER_TURN = 10
BOARD_SIZE = 10

# -------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------------
def bfs(tiles, start):
    dist = {start: 0}
    q, head = [start], 0
    while head < len(q):
        x, y = q[head]
        head += 1
        d = dist[(x, y)] + 1
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in dist and tiles[ny][nx] != "LOCKED":
                dist[(nx, ny)] = d
                q.append((nx, ny))
    return dist

def move_towards(tiles, start, goal):
    if start == goal: return None
    back = bfs(tiles, goal)
    best = None
    dirs = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
    for op, (dx, dy) in dirs.items():
        nx, ny = start[0] + dx, start[1] + dy
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or tiles[ny][nx] == "LOCKED":
            continue
        d = back.get((nx, ny))
        if d is not None and (best is None or d < best[0]):
            best = (d, op)
    return best[1] if best else None

# -------------------------------------------------------------------------
# 3. MAIN AGENT
# -------------------------------------------------------------------------
def agent(obs):
    me = obs["farms"][obs["player"]]
    shed = obs["private"]["shed"]
    seeds = obs["private"]["seeds"]
    invs = obs["private"]["inventories"]
    hour = obs["hour"]
    day = obs["day"]
    tiles = me["tiles"]
    
    # --- PHASE 1: MARKET ARBITRAGE & EXPANSION ---
    market_orders = []
    
    # 1. Hire hands slowly (max 3, one per day)
    if len(me["hands"]) < 3 and me["hires_today"] == 0 and me["money"] > 500:
        market_orders.append(["HIRE"])
        
    # 2. Buy Land (Unlock Northeast Quadrant)
    if "NE" not in me["unlocked_quadrants"] and me["money"] > 1500:
        market_orders.append(["UNLOCK", "NE"])
        
    # 3. Sell logic
    total_items = sum(shed.values())
    for item, amount in shed.items():
        if amount > 0:
            if hour % 4 == 3 or total_items > (SHED_CAPACITY - 20):
                market_orders.append(["SELL", item, amount])
                
    # 4. Dynamic Seed Buying (The Melon Switch)
    # Melons take 12 days. If we have cash and time, buy Melons. Otherwise, buy Wheat.
    if day < 18 and seeds.get("MELON", 0) < 5 and me["money"] > 800:
        market_orders.append(["BUY_SEED", "MELON", 5])
    elif seeds.get("WHEAT", 0) < 10 and me["money"] > 150:
        market_orders.append(["BUY_SEED", "WHEAT", 10])
        
    market_orders = market_orders[:MAX_MARKET_ORDERS_PER_TURN]
    
    # --- PHASE 2: TASK SCHEDULER ---
    tasks = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            t = tiles[y][x]
            if t == "LOCKED": continue
            
            if t is None:
                # Plant highest value seed available
                if seeds.get("MELON", 0) > 0:
                    tasks.append((3, (x, y), ["PLANT", "MELON"]))
                elif seeds.get("WHEAT", 0) > 0:
                    tasks.append((3, (x, y), ["PLANT", "WHEAT"]))
            elif t["kind"] == "WEED":
                tasks.append((4, (x, y), ["DIG"]))
            elif t["kind"] == "PLANT":
                if not t["watered_today"]:
                    tasks.append((1, (x, y), ["WATER"]))
                elif t["yield_units"] > 0:
                    tasks.append((2, (x, y), ["HARVEST"]))
                    
    tasks.sort(key=lambda t: t[0])
    
    # --- PHASE 3: WORKER ASSIGNMENT ---
    units = [tuple(me["farmer"])] + [tuple(u) for u in me["hands"]]
    ops = []
    
    for idx, pos in enumerate(units):
        inv = invs[idx] if idx < len(invs) else {}
        carried = sum(inv.values())
        
        if carried >= 8 or (hour >= 21 and carried > 0):
            if pos == (4, 4):
                ops.append(["DROP"])
            else:
                mv = move_towards(tiles, pos, (4, 4))
                ops.append([mv] if mv else ["PASS"])
            continue

        d = bfs(tiles, pos)
        best_task = None
        
        for prio, tp, top in tasks:
            if tp in d:
                score = d[tp] + (prio * 3) 
                if best_task is None or score < best_task[0]:
                    best_task = (score, tp, top)
                    
        if best_task is not None:
            _, target_pos, target_action = best_task
            tasks = [t for t in tasks if t[1] != target_pos]
            
            if pos == target_pos:
                ops.append(target_action)
            else:
                mv = move_towards(tiles, pos, target_pos)
                ops.append([mv] if mv else ["PASS"])
        else:
            ops.append(["PASS"])

    return {
        "farmer": ops[0],
        "hands": ops[1:],
        "market": market_orders
    }

umutdorukztrk_agent = agent

