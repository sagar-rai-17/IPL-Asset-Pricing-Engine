# IPL Asset Pricing & Portfolio Optimization Engine

An end-to-end quantitative decision support pipeline that transforms raw, ball-by-ball transactional match logs into context-aware player performance metrics, predicts baseline asset market values (FMV), and constructs a risk-hedged portfolio under strict regulatory and budgetary constraints.

---

## System Architecture

The engine is built on a modular six-stage pipeline structured as follows:

```
[ Raw Ball-by-Ball Logs ] 
          │
          ▼
[ Stage 1 & 2: Feature Engineering & Clustering ]
  - Computes temporal decay strike rates and leverage-adjusted pressure indices.
  - Classifies player archetypes using unsupervised K-Means clustering.
          │
          ▼
[ Stage 3: Supervised ML Valuation (FMV) ]
  - Maps performance dimensions to market clearing prices using LightGBM & XGBoost.
  - Detects market pricing anomalies (Alpha vs. Premium Inflation residuals).
          │
          ▼
[ Stage 4 & 6: Prescriptive Optimization & Bidding Simulation ]
  - Formulates a sequential Integer Linear Program (ILP) using PuLP.
  - Calculates dynamic opportunity-cost walk-away bid ceilings in real-time.
          │
          ▼
[ Stage 5: Copula-Based Monte Carlo Risk Simulation ]
  - Models joint probability performance distributions using a Gaussian Copula.
  - Quantifies roster downside risk through Value-at-Risk (5% VaR) metrics.
```

---

## Directory Structure

```text
ipl_engine/
│
├── data/
│   └── IPL_Auction_Valuation_Large_Dataset.xlsx  # Raw transactional logs
│
├── day1_eda.py                       # Ingestion & diagnostics
├── day2_features.py                  # Temporal & situational metrics
├── day3_valuation.py                 # ML ensemble pricing
├── day4_optimization.py              # Integer Linear Program solver
├── day5_simulation.py                # Copula-based risk simulator
├── day6_auction_simulator.py         # Sequential bidding & bid-ceilings
│
├── run_pipeline.py                   # Automated pipeline orchestrator
├── requirements.txt                  # System dependencies
└── .gitignore                        # Git exclusion rules
```

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ipl_engine.git
   cd ipl_engine
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.11+ installed. Install the system dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Complete Pipeline:**
   Execute the automated pipeline orchestrator to run all six modules sequentially, verify data handoffs, and generate the analytical charts:
   ```bash
   python run_pipeline.py
   ```

---

## Verified Pipeline Execution Log

```text
==============================================
   IPL ASSET PRICING & OPTIMIZATION PIPELINE  
==============================================

[STAGE 1/6] Running: Day 1: Ingestion & Diagnostic EDA...
✓ Success (Time Taken: 5.44 seconds)

[STAGE 2/6] Running: Day 2: Feature Engineering & Clustering...
✓ Success (Time Taken: 3.29 seconds)

[STAGE 3/6] Running: Day 3: Supervised Machine Learning Valuation...
✓ Success (Time Taken: 1.45 seconds)

[STAGE 4/6] Running: Day 4: Prescriptive Portfolio Optimization...
✓ Success (Time Taken: 0.58 seconds)

[STAGE 5/6] Running: Day 5: Copula-Based Monte Carlo Risk Simulation...
✓ Success (Time Taken: 1.76 seconds)
  > Base Portfolio Expected Utility: 22.43
  > Base Portfolio Value-at-Risk (5th Percentile VaR): 17.53
  > Post-Stress Expected Utility (Kohli -30% performance): 19.46
  > Total Portfolio Utility Loss Under Stress: 13.36%

[STAGE 6/6] Running: Day 6: Dynamic Bid-Ceiling Auction Simulation...
✓ Success (Time Taken: 0.72 seconds)
  > --- FINAL SIMULATION RESULTS ---
  > Acquired Roster: ['Shubman Gill', 'Jasprit Bumrah', 'Jos Buttler']
  > Remaining Capital: INR 14.56 Cr

==============================================
             PIPELINE COMPLETE                
Total Execution Time: 13.26 seconds
All artifacts generated and validated successfully.
==============================================
```

---

## Core Analytical Outputs

*   **Valuation Anomaly Detection:** The regression models identified key market inefficiencies by examining prediction residuals ($y - \hat{y}$), isolating **Jasprit Bumrah** as a primary undervalued "Alpha" asset and **Rohit Sharma** as carrying high premium inflation (the "Winner's Curse").
*   **Sequential Roster Optimization:** Under strict composition constraints and a INR 30.0 Crore budget, the engine successfully optimized and locked in high-utility assets while dynamically updating selection priorities in real-time.
*   **Copula-Based Volatility Stress-Testing:** Running 10,000 simulated matches with a Gaussian Copula established a baseline portfolio Value-at-Risk (5% VaR) of **17.53**. Applying a 30% performance degradation stress test on the marquee asset, Virat Kohli, quantified a systemic **13.36% expected performance drop**, validating the business necessity of strategic roster hedging.