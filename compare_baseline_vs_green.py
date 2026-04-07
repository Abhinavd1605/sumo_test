"""
2-Way Comparison: Fixed-Time Baseline vs Green AI DNLight on Grid.

Runs the fixed-time controller and the Green AI model on the 2x2 grid network, 
collects detailed traffic and environmental metrics, and generates 
a head-to-head comparison table and visualization charts.

Usage:
    python compare_baseline_vs_green.py --episodes 5
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent
from dnlight.baselines import FixedTimeController
from dnlight.carbon_tracker import CarbonTracker

import traci

def parse_args():
    p = argparse.ArgumentParser(description="Fixed-Time vs Green AI Comparison")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Grid network config")
    p.add_argument("--green-ckpt", type=str,
                   default="checkpoints_multi_green_ext/dnlight_green_best",
                   help="Extended Green AI grid checkpoint")
    p.add_argument("--episodes", type=int, default=5,
                   help="Evaluation episodes per approach")
    p.add_argument("--output", type=str, default="results_baseline_vs_green",
                   help="Output directory for results")
    return p.parse_args()

class MetricCollector:
    """Collect traffic and emission metrics during an episode."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_wait_time = 0.0
        self.total_time_loss = 0.0
        self.emv_delay = 0.0
        self.emv_count = 0
        self.vehicle_count = 0
        self.arrived = 0
        self.total_co2 = 0.0
        self.total_nox = 0.0
        self.total_fuel = 0.0
        self.total_pmx = 0.0
        self.steps = 0
        self.total_reward = 0.0

    def collect_step(self, info: dict):
        self.steps += 1
        if 'emissions' in info:
            for tls_em in info['emissions'].values():
                if isinstance(tls_em, dict):
                    self.total_co2 += tls_em.get('co2_mg_per_s', 0)
                    self.total_nox += tls_em.get('nox_mg_per_s', 0)
                    self.total_fuel += tls_em.get('fuel_ml_per_s', 0)
                    self.total_pmx += tls_em.get('pmx_mg_per_s', 0)

    def collect_end_of_episode(self):
        try:
            self.arrived = traci.simulation.getArrivedNumber()
            for vid in traci.vehicle.getIDList():
                try:
                    wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                    time_loss = traci.vehicle.getTimeLoss(vid)
                    vtype = traci.vehicle.getTypeID(vid)
                    self.total_wait_time += wait
                    self.total_time_loss += time_loss
                    self.vehicle_count += 1
                    if vtype in ("ambulance", "fire_truck", "police"):
                        self.emv_delay += time_loss
                        self.emv_count += 1
                except: continue
        except: pass

    def get_summary(self) -> dict:
        n = max(self.vehicle_count, 1)
        return {
            'avg_waiting_time': self.total_wait_time / n,
            'avg_time_loss': self.total_time_loss / n,
            'emv_avg_delay': self.emv_delay / max(self.emv_count, 1),
            'throughput': self.arrived,
            'total_vehicles': self.vehicle_count,
            'co2_kg': self.total_co2 / 1e6,
            'nox_g': self.total_nox / 1e3,
            'fuel_liters': self.total_fuel / 1e3,
            'pmx_g': self.total_pmx / 1e3,
            'total_reward': self.total_reward,
        }

def run_evaluation(config, checkpoint, episodes, green=False, seeds=None, use_fixed_time=False):
    mode_name = "Fixed Time Baseline" if use_fixed_time else ("Green AI" if green else "Standard AI")
    print(f"\nEvaluating {mode_name} for {episodes} episodes (Deterministic Traffic)...")
    
    env = MultiIntersectionEnv(sumocfg_path=config, use_gui=False, use_green_reward=green)
    
    carbon = CarbonTracker()
    results = []
    
    agent = None
    if not use_fixed_time:
        agent = DNLightAgent(env.get_state_dim(), env.get_action_dim(), use_attention=True)
        agent.load(checkpoint)

    for ep in range(1, episodes + 1):
        seed = seeds[ep-1] if seeds else None
        states = env.reset(seed=seed)
        
        agents = {}
        if use_fixed_time:
            # Fixed time controller per intersection, uses 50s green duration
            agents = {tid: FixedTimeController(num_phases=4, green_duration=50, yellow_duration=env.yellow_duration) for tid in env.tls_ids}
        else:
            agents = {tid: agent for tid in env.tls_ids}

        mc = MetricCollector()
        done = False
        while not done:
            carbon.start_step()
            
            if use_fixed_time:
                actions = {tid: agents[tid].select_action() for tid in env.tls_ids}
            else:
                actions = {tid: agents[tid].select_action(states[tid], training=False) for tid in env.tls_ids}
                
            next_states, rewards, done, info = env.step(actions)
            carbon.end_step()
            
            if use_fixed_time:
                for tid in env.tls_ids:
                    agents[tid].update(step_duration=env.step_duration)

            mc.collect_step(info)
            mc.total_reward += sum(rewards.values())
            states = next_states
            
        mc.collect_end_of_episode()
        summary = mc.get_summary()
        summary['compute_carbon_mg'] = carbon.reset_episode() * 1000
        results.append(summary)
        print(f"  Ep {ep} (Seed {seed}): Wait={summary['avg_waiting_time']:.1f}s, CO2={summary['co2_kg']:.2f}kg")

    env.close()
    return results

def aggregate(results):
    return {k: {'mean': float(np.mean([r[k] for r in results])), 'std': float(np.std([r[k] for r in results]))} 
            for k in results[0].keys()}

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Generate fixed set of seeds for all models
    eval_seeds = [44 + i for i in range(args.episodes)]

    # Run Fixed-Time Baseline
    ft_raw = run_evaluation(args.config, None, args.episodes, green=False, seeds=eval_seeds, use_fixed_time=True)
    
    # Run Green AI
    grn_raw = run_evaluation(args.config, args.green_ckpt, args.episodes, green=True, seeds=eval_seeds, use_fixed_time=False)

    ft, grn = aggregate(ft_raw), aggregate(grn_raw)
    
    # Save JSON
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump({'fixed_time': ft, 'green': grn}, f, indent=2)

    # Plotting
    metrics_to_plot = [
        ('avg_waiting_time', 'Avg Wait (s)'),
        ('co2_kg', 'CO2 (kg)'),
        ('nox_g', 'NOx (g)'),
        ('fuel_liters', 'Fuel (L)'),
        ('total_reward', 'Reward'),
        ('compute_carbon_mg', 'Compute CO2 (mg)')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fixed-Time Baseline vs Green AI Comparison', fontsize=16)
    
    labels = ['Fixed-Time', 'Green AI']
    for i, (m, label) in enumerate(metrics_to_plot):
        ax = axes[i//3, i%3]
        ft_val, grn_val = ft[m]['mean'], grn[m]['mean']
        ft_std, grn_std = ft[m]['std'], grn[m]['std']
        
        bars = ax.bar(labels, [ft_val, grn_val], 
                       yerr=[ft_std, grn_std], color=['#e74c3c', '#2ecc71'], capsize=10)
        ax.set_title(label)
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(args.output, 'comparison_charts.png')
    plt.savefig(plot_path)
    print(f"\nResults saved to {args.output}")
    print(f"Charts saved to {plot_path}")

    # Print Table
    print("\n" + "="*70)
    print(f"{'Metric':<20} | {'Fixed-Time Baseline':>20} | {'Green AI':>20}")
    print("-" * 70)
    for m, label in metrics_to_plot:
        print(f"{label:<20} | {ft[m]['mean']:>20.2f} | {grn[m]['mean']:>20.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
