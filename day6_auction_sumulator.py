import pandas as pd
import pulp

df = pd.read_csv("player_valuation_market_analysis.csv")

p_names = df["Player"].tolist()
u = dict(zip(p_names, df["utility_score"]))
fmv = dict(zip(p_names, df["fmv"]))
sal = dict(zip(p_names, df["actual_salary"]))
roles = dict(zip(p_names, df["Role"]))
os_flag = dict(zip(p_names, df["is_overseas"]))

def solve_roster(acquired, budget, active_pool):
    prob = pulp.LpProblem("Dynamic_Portfolio", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("select", p_names, cat="Binary")
    
    prob += pulp.lpSum([u[p] * x[p] for p in p_names])
    
    prob += pulp.lpSum([sal[p] * x[p] for p in p_names if p not in acquired]) <= budget
    prob += pulp.lpSum([x[p] for p in p_names]) == 5
    
    prob += pulp.lpSum([x[p] for p in p_names if roles[p] == "Batsman"]) >= 2
    prob += pulp.lpSum([x[p] for p in p_names if roles[p] == "Bowler"]) >= 1
    prob += pulp.lpSum([x[p] for p in p_names if roles[p] == "Wicketkeeper"]) == 1
    prob += pulp.lpSum([x[p] for p in p_names if os_flag[p] == 1]) <= 2
    
    for p in acquired:
        prob += x[p] == 1
    for p in p_names:
        if p not in active_pool and p not in acquired:
            prob += x[p] == 0
            
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    return [p for p in p_names if x[p].varValue == 1]

def calc_bid_ceil(p, budget, active_pool):
    p_role = roles[p]
    p_u = u[p]
    
    alts = [x for x in active_pool if roles[x] == p_role and x != p]
    if not alts:
        return fmv[p]
        
    next_best = max(alts, key=lambda x: u[x])
    u_diff = p_u - u[next_best]
    
    rem_u = sum(u[x] for x in active_pool)
    cost_per_u = budget / (rem_u + 1e-5)
    
    ceil = fmv[p] + (u_diff * cost_per_u)
    return max(fmv[p] * 0.8, min(ceil, budget * 0.7))

print("--- STARTING SEQUENTIAL AUCTION SIMULATION ---")
budget = 30.0
acquired = []
active_pool = list(p_names)

draft_sequence = ["Shubman Gill", "Virat Kohli", "Jasprit Bumrah", "Suryakumar Yadav", "Jos Buttler"]

for step, target in enumerate(draft_sequence, 1):
    print(f"\n>>> STEP {step}: {target} is up for bidding!")
    
    ceil = calc_bid_ceil(target, budget, active_pool)
    actual_cost = sal[target]
    
    print(f"Role: {roles[target]} | Baseline FMV: INR {fmv[target]:.2f} Cr | Dynamic Bid Ceiling: INR {ceil:.2f} Cr")
    print(f"Current Market Bidding Price: INR {actual_cost:.2f} Cr")
    
    if actual_cost <= ceil:
        print(f"DECISION: BID & ACQUIRE! Price (INR {actual_cost:.2f} Cr) is within the ceiling limit.")
        acquired.append(target)
        budget -= actual_cost
    else:
        print("DECISION: WALK AWAY! Bidding exceeds dynamic ceiling value.")
        
    active_pool.remove(target)
    
    target_roster = solve_roster(acquired, budget, active_pool)
    
    print(f"Remaining Budget: INR {budget:.2f} Cr")
    if target_roster:
        print(f"Projected Optimal 5-Player Squad: {target_roster}")
    else:
        print("ALERT: Current state of acquisitions has caused roster infeasibility!")

print("\n--- FINAL SIMULATION RESULTS ---")
print(f"Acquired Roster: {acquired}")
print(f"Remaining Capital: INR {budget:.2f} Cr")
