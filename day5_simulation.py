import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

df = pd.read_csv("player_valuation_market_analysis.csv")

port = ["Hardik Pandya", "Jasprit Bumrah", "Rishabh Pant", "Shubman Gill", "Virat Kohli"]
df_port = df[df["Player"].isin(port)].set_index("Player")

means = df_port["utility_score"].to_dict()
vars_dict = {p: means[p] * 0.25 for p in port}

n_sims = 10000
n_players = len(port)

R = np.array([
    [1.0, 0.1, 0.2, 0.3, 0.4],
    [0.1, 1.0, -0.1, -0.1, -0.1],
    [0.2, -0.1, 1.0, 0.25, 0.3],
    [0.3, -0.1, 0.25, 1.0, 0.35],
    [0.4, -0.1, 0.3, 0.35, 1.0]
])

mvn = stats.multivariate_normal(mean=np.zeros(n_players), cov=R)
z = mvn.rvs(size=n_sims)
u = stats.norm.cdf(z)

def gen_perf(u_matrix, drop_player=None, drop_pct=0.0):
    p_matrix = np.zeros_like(u_matrix)
    for i, p in enumerate(port):
        m = means[p]
        v = vars_dict[p]
        
        if p == drop_player:
            m = m * (1.0 - drop_pct)
            
        theta = v / m
        k = m / theta
        
        p_matrix[:, i] = stats.gamma.ppf(u_matrix[:, i], a=k, scale=theta)
    return p_matrix

perf_base = gen_perf(u)
tot_base = perf_base.sum(axis=1)

perf_stress = gen_perf(u, drop_player="Virat Kohli", drop_pct=0.30)
tot_stress = perf_stress.sum(axis=1)

var_5_base = np.percentile(tot_base, 5)
var_5_stress = np.percentile(tot_stress, 5)

m_base = tot_base.mean()
m_stress = tot_stress.mean()
perf_loss = (m_base - m_stress) / m_base * 100

print("--- PORTFOLIO RISK & VOLATILITY METRICS ---")
print(f"Base Portfolio Expected Utility: {m_base:.2f}")
print(f"Base Portfolio Value-at-Risk (5th Percentile VaR): {var_5_base:.2f}")
print(f"Post-Stress Expected Utility (Kohli -30% performance): {m_stress:.2f}")
print(f"Post-Stress Value-at-Risk (5th Percentile VaR): {var_5_stress:.2f}")
print(f"Total Portfolio Utility Loss Under Stress: {perf_loss:.2f}%")

plt.figure(figsize=(10, 6))
plt.hist(tot_base, bins=50, alpha=0.6, label="Base Portfolio Distribution", color="royalblue", density=True)
plt.hist(tot_stress, bins=50, alpha=0.6, label="Stressed Portfolio (Kohli -30%)", color="crimson", density=True)
plt.axvline(var_5_base, color="blue", linestyle="dashed", linewidth=2, label=f"Base 5% VaR ({var_5_base:.2f})")
plt.axvline(var_5_stress, color="red", linestyle="dashed", linewidth=2, label=f"Stressed 5% VaR ({var_5_stress:.2f})")
plt.title("Portfolio Performance Risk Profile: Copula-Based Monte Carlo")
plt.xlabel("Aggregated Portfolio Utility Score")
plt.ylabel("Probability Density")
plt.legend()
plt.tight_layout()
plt.savefig("portfolio_risk_simulation.png")
plt.close()

print("\nDay 5 Complete. Monte Carlo risk analysis plots saved to 'portfolio_risk_simulation.png'")