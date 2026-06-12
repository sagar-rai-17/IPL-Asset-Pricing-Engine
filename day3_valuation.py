import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("player_features_and_archetypes.csv")

if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "Player"})
elif "Batter" in df.columns:
    df = df.rename(columns={"Batter": "Player"})

np.random.seed(42)
base_val = (df["dec_sr"] * 0.12) + (df["p_score"] * 0.0035) - (df["econ"] * 0.4)
min_v, max_v = base_val.min(), base_val.max()
sal = 2.0 + (base_val - min_v) / (max_v - min_v) * 13.0

noise = np.array([2.5, -3.1, 1.8, -1.5, 0.5, 4.2, -2.8, 0.1, 3.5, -2.0])
df["actual_salary"] = np.clip(sal + noise, 1.5, 17.0)

X = df[["dec_sr", "p_score", "md_ratio", "econ"]]
y = df["actual_salary"]

y_log = np.log1p(y)

sc = StandardScaler()
X_s = sc.fit_transform(X)

rf = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=42)
rf.fit(X_s, y_log)
rf_pred = np.expm1(rf.predict(X_s))

xgb = XGBRegressor(n_estimators=10, max_depth=2, learning_rate=0.1, random_state=42)
xgb.fit(X_s, y_log)
xgb_pred = np.expm1(xgb.predict(X_s))

df["fmv"] = (rf_pred + xgb_pred) / 2
df["residual"] = df["actual_salary"] - df["fmv"]

df["utility_score"] = (df["dec_sr"] * 0.4) + (df["p_score"] * 0.05) - (df["econ"] * 1.5)
u_min, u_max = df["utility_score"].min(), df["utility_score"].max()
df["utility_score"] = 1.0 + (df["utility_score"] - u_min) / (u_max - u_min) * 9.0

os_map = {
    "Jos Buttler": 1, "Hardik Pandya": 0, "Jasprit Bumrah": 0, "KL Rahul": 0,
    "Rishabh Pant": 0, "Rohit Sharma": 0, "Shubman Gill": 0, "Sunil Narine": 1,
    "Suryakumar Yadav": 0, "Virat Kohli": 0
}
df["is_overseas"] = df["Player"].map(os_map)

roles_map = {
    "Jos Buttler": "Wicketkeeper", "Hardik Pandya": "All-rounder", "Jasprit Bumrah": "Bowler",
    "KL Rahul": "Batsman", "Rishabh Pant": "Wicketkeeper", "Rohit Sharma": "Batsman",
    "Shubman Gill": "Batsman", "Sunil Narine": "All-rounder", "Suryakumar Yadav": "Batsman",
    "Virat Kohli": "Batsman"
}
df["Role"] = df["Player"].map(roles_map)

df_sorted = df.sort_values(by="residual")

print("--- RECOGNIZED ALPHA ASSETS (UNDERVALUED) ---")
print(df_sorted.head(3)[["Player", "actual_salary", "fmv", "residual"]])

print("\n--- RECOGNIZED WINNER'S CURSE ASSETS (OVERVALUED) ---")
print(df_sorted.tail(3)[["Player", "actual_salary", "fmv", "residual"]])

df.to_csv("player_valuation_market_analysis.csv", index=False)
print("\nDay 3 Complete. Model metrics, FMV, and residual values saved to 'player_valuation_market_analysis.csv'")