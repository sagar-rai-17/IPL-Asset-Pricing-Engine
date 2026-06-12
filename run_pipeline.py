import sys
import os
import time
import subprocess

stages = [
    ("Day 1: Ingestion & Diagnostic EDA", "day1_eda.py"),
    ("Day 2: Feature Engineering & Clustering", "day2_features.py"),
    ("Day 3: Supervised Machine Learning Valuation", "day3_valuation.py"),
    ("Day 4: Prescriptive Portfolio Optimization", "day4_optimization.py"),
    ("Day 5: Copula-Based Monte Carlo Risk Simulation", "day5_simulation.py"),
    ("Day 6: Dynamic Bid-Ceiling Auction Simulation", "day6_auction_simulator.py")
]

print("==============================================")
print("   IPL ASSET PRICING & OPTIMIZATION PIPELINE  ")
print("==============================================")

st_time = time.time()

for idx, (name, file) in enumerate(stages, 1):
    print(f"\n[STAGE {idx}/{len(stages)}] Running: {name}...")
    
    if not os.path.exists(file):
        print(f"ERROR: Execution file '{file}' not found in current directory.")
        sys.exit(1)
        
    s_start = time.time()
    res = subprocess.run([sys.executable, file], capture_output=True, text=True)
    s_dur = time.time() - s_start
    
    if res.returncode != 0:
        print(f"\n❌ PIPELINE FAILURE AT STAGE {idx} ({name})")
        print("ERROR DETAILS:")
        print(res.stderr)
        sys.exit(1)
        
    print(f"✓ Success (Time Taken: {s_dur:.2f} seconds)")
    
    if idx in [1, 2, 3, 5, 6]:
        lines = res.stdout.strip().split("\n")
        print("Output Snippet:")
        for line in lines[-5:]:
            print(f"  > {line}")

tot_dur = time.time() - st_time
print("\n==============================================")
print("             PIPELINE COMPLETE                ")
print(f"Total Execution Time: {tot_dur:.2f} seconds")
print("All artifacts generated and validated successfully.")
print("==============================================")