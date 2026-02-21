"""Run Fixed-Time baseline on the grid for early results."""
import json
import os
from compare import run_fixed_time, aggregate_results

def main():
    config = "configs/grid_network.sumocfg"
    results = run_fixed_time(config, episodes=3)
    agg = aggregate_results(results)
    
    os.makedirs("results", exist_ok=True)
    with open("results/fixed_baseline.json", "w") as f:
        json.dump(agg, f, indent=2)
    
    print("\nFixed-Time Aggregate Results (3 episodes):")
    print(f"  Avg Wait: {agg['avg_waiting_time']['mean']:.2f}s")
    print(f"  CO2: {agg['co2_kg']['mean']:.2f} kg")
    print(f"  Throughput: {agg['throughput']['mean']:.0f}")

if __name__ == "__main__":
    main()
