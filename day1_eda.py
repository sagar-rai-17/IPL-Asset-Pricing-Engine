import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fp = os.path.join("data", "IPL_Auction_Valuation_Large_Dataset.xlsx")
df = pd.read_excel(fp, sheet_name="Ball_by_Ball_Data")

print("--- DATA STATUS ---")
print(f"Total Rows: {len(df)}")
print(f"Missing Values:\n{df.isnull().sum()}")

df["Runs_Batter"] = df["Runs_Batter"].fillna(0)
df["Extras"] = df["Extras"].fillna(0)
df["Total_Runs"] = df["Total_Runs"].fillna(0)
df["Wicket"] = df["Wicket"].fillna(0).astype(int)
df["Wicket_Type"] = df["Wicket_Type"].fillna("None")
df["Player_Out"] = df["Player_Out"].fillna("None")

str_cols = ["Batter", "Bowler", "Non_Striker", "Venue", "Batting_Team", "Bowling_Team"]
for c in str_cols:
    df[c] = df[c].astype(str).str.strip()

df["Date"] = pd.to_datetime(df["Date"])

df["Target"] = df["Target"].fillna(0)
df["Required_Run_Rate"] = df["Required_Run_Rate"].fillna(0.0)

matches = df.groupby("Match_ID")
r_per_match = matches["Total_Runs"].sum()
w_per_phase = df.groupby("Phase")["Wicket"].sum()

results = df.drop_duplicates(subset=["Match_ID"], keep="last")
win_pct = results["Result"].value_counts(normalize=True)

print("\n--- DIAGNOSTIC METRICS ---")
print(f"Unique Matches: {df['Match_ID'].nunique()}")
print(f"Unique Batters: {df['Batter'].nunique()}")
print(f"Unique Bowlers: {df['Bowler'].nunique()}")
print(f"Average Runs Per Match: {r_per_match.mean():.1f}")
print("\nWickets Per Phase:")
print(w_per_phase)
print("\nWin/Loss Ratios:")
print(win_pct)

plt.figure(figsize=(10, 5))
sns.histplot(r_per_match, bins=20, kde=True, color="blue")
plt.title("Distribution of Total Runs Per Match")
plt.xlabel("Total Runs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("runs_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=w_per_phase.index, y=w_per_phase.values, palette="viridis")
plt.title("Total Wickets Fallen by Phase")
plt.xlabel("Match Phase")
plt.ylabel("Wickets")
plt.tight_layout()
plt.savefig("wickets_by_phase.png")
plt.close()

df.to_csv("cleaned_ball_by_ball.csv", index=False)
print("\nDay 1 Complete. Cleaned data exported to 'cleaned_ball_by_ball.csv'")