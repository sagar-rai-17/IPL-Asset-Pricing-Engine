import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("cleaned_ball_by_ball.csv")

b_types = {b: ("Spin" if i % 2 == 0 else "Pace") for i, b in enumerate(df["Bowler"].unique())}
df["Bowler_Type"] = df["Bowler"].map(b_types)

max_s = df["Season"].max()
df["w"] = np.exp(-0.05 * (max_s - df["Season"]))

df["run_w"] = df["Runs_Batter"] * df["w"]
df["ball_w"] = df["w"]

df["p_mult"] = df["Required_Run_Rate"] / (11 - df["Current_Wickets"])
df["p_index"] = df["Runs_Batter"] * df["p_mult"]

bat_grp = df.groupby("Batter")
bat_perf = pd.DataFrame({
    "r_sum": bat_grp["Runs_Batter"].sum(),
    "b_faced": bat_grp.size(),
    "dec_runs": bat_grp["run_w"].sum(),
    "dec_balls": bat_grp["ball_w"].sum(),
    "p_score": bat_grp["p_index"].sum()
})

bat_perf["sr"] = (bat_perf["r_sum"] / bat_perf["b_faced"]) * 100
bat_perf["dec_sr"] = (bat_perf["dec_runs"] / bat_perf["dec_balls"]) * 100

vs_spin = df[df["Bowler_Type"] == "Spin"].groupby("Batter")["Runs_Batter"].sum()
vs_pace = df[df["Bowler_Type"] == "Pace"].groupby("Batter")["Runs_Batter"].sum()

bat_perf["r_vs_spin"] = bat_perf.index.map(vs_spin).fillna(0)
bat_perf["r_vs_pace"] = bat_perf.index.map(vs_pace).fillna(0)
bat_perf["md_ratio"] = bat_perf["r_vs_spin"] / (bat_perf["r_vs_pace"] + 1e-5)

bat_features = bat_perf[["dec_sr", "p_score", "md_ratio"]].copy()

bowl_grp = df.groupby("Bowler")
bowl_perf = pd.DataFrame({
    "r_conceded": bowl_grp["Total_Runs"].sum(),
    "b_bowled": bowl_grp.size()
})
bowl_perf["econ"] = (bowl_perf["r_conceded"] / bowl_perf["b_bowled"]) * 6

m_df = pd.merge(bat_features, bowl_perf[["econ"]], left_index=True, right_index=True, how="outer").fillna(0)

scl = StandardScaler()
m_scaled = scl.fit_transform(m_df)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
m_df["Cluster"] = km.fit_predict(m_scaled)

pca = PCA(n_components=2)
coords = pca.fit_transform(m_scaled)
m_df["pca1"] = coords[:, 0]
m_df["pca2"] = coords[:, 1]

plt.figure(figsize=(8, 6))
for cl in sorted(m_df["Cluster"].unique()):
    sub = m_df[m_df["Cluster"] == cl]
    plt.scatter(sub["pca1"], sub["pca2"], label=f"Cluster {cl}", s=100)

for idx, row in m_df.iterrows():
    plt.annotate(idx, (row["pca1"] + 0.05, row["pca2"] + 0.05), fontsize=9)

plt.title("Player Archetype Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("player_archetypes.png")
plt.close()

m_df.to_csv("player_features_and_archetypes.csv")
print("Day 2 Complete. Feature metrics and clusters saved to 'player_features_and_archetypes.csv'")
print("\nGenerated Feature Matrix:")
print(m_df[["dec_sr", "p_score", "md_ratio", "econ", "Cluster"]])