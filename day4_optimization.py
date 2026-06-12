import pandas as pd
import pulp

df = pd.read_csv("player_valuation_market_analysis.csv")

p_names = df["Player"].tolist()
u = dict(zip(p_names, df["utility_score"]))
sal = dict(zip(p_names, df["actual_salary"]))
roles = dict(zip(p_names, df["Role"]))
os_flag = dict(zip(p_names, df["is_overseas"]))

def solve_roster(acquired=None, budget=30.0):
    if acquired is None:
        acquired = []
        
    prob = pulp.LpProblem("IPL_Portfolio_Optimization", pulp.LpMaximize)
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
        
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] != "Optimal":
        return None, None
        
    selected = [p for p in p_names if x[p].varValue == 1]
    tot_cost = sum(sal[p] for p in selected)
    tot_u = sum(u[p] for p in selected)
    
    return selected, tot_cost, tot_u

print("--- SCENARIO 1: INITIAL UNCONSTRAINED SOLVE (BUDGET = 30.0 Cr) ---")
sel_1, cost_1, util_1 = solve_roster()
if sel_1:
    for p in sel_1:
        print(f"Player: {p:<16} | Role: {roles[p]:<12} | Cost: INR {sal[p]:.2f} Cr | Utility: {u[p]:.2f}")
    print(f"Total Portfolio Cost: INR {cost_1:.2f} Cr | Accumulated Utility: {util_1:.2f}")
else:
    print("Optimization failed to find a valid solution under current constraints.")

print("\n--- SCENARIO 2: SEQUENTIAL SOLVE (LOCKED: Virat Kohli, BUDGET = 30.0 Cr) ---")
sel_2, cost_2, util_2 = solve_roster(acquired=["Virat Kohli"])
if sel_2:
    for p in sel_2:
        is_locked = " [LOCKED]" if p == "Virat Kohli" else ""
        print(f"Player: {p:<16}{is_locked:<9} | Role: {roles[p]:<12} | Cost: INR {sal[p]:.2f} Cr | Utility: {u[p]:.2f}")
    print(f"Total Portfolio Cost: INR {cost_2:.2f} Cr | Accumulated Utility: {util_2:.2f}")
else:
    print("Optimization failed. Budget limit exceeded by locking in premium player.")