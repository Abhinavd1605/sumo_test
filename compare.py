"""
3-Way Comparison: Fixed-Time vs DNLight vs DNLight+GreenAI on Grid.

Runs all three approaches on the 2x2 grid network, collects
traffic performance and emission metrics, and generates a
comparison table + charts.

Usage:
    python compare.py --dnlight-ckpt checkpoints_multi/dnlight_standard_best \
                      --green-ckpt checkpoints_multi/dnlight_green_best \
                      --episodes 5
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent
from dnlight.baselines import FixedTimeController
from dnlight.carbon_tracker import CarbonTracker

import traci


def parse_args():
    p = argparse.ArgumentParser(description="DNLight 3-Way Comparison on Grid")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Grid network config")
    p.add_argument("--dnlight-ckpt", type=str,
                   default="checkpoints_multi_std_ext/dnlight_green_best",
                   help="Extended Standard DNLight grid checkpoint")
    p.add_argument("--green-ckpt", type=str,
                   default="checkpoints_multi_green_ext/dnlight_green_best",
                   help="Extended Green AI grid checkpoint")
    p.add_argument("--episodes", type=int, default=5,
                   help="Evaluation episodes per approach")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory for results")
    return p.parse_args()


class MetricCollector:
    """Collect traffic and emission metrics during an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_wait_time = 0.0
        self.total_travel_time = 0.0
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
        """Collect metrics from one step's info dict."""
        self.steps += 1

        # Emissions from info (Multi-intersection aggregate)
        if 'emissions' in info:
            em = info['emissions']
            for tls_em in em.values():
                if isinstance(tls_em, dict):
                    self.total_co2 += tls_em.get('co2_mg_per_s', 0)
                    self.total_nox += tls_em.get('nox_mg_per_s', 0)
                    self.total_fuel += tls_em.get('fuel_ml_per_s', 0)
                    self.total_pmx += tls_em.get('pmx_mg_per_s', 0)

    def collect_end_of_episode(self):
        """Collect final per-vehicle stats at end of episode."""
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
                except Exception:
                    continue
        except Exception:
            pass

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
            'steps': self.steps,
        }


def run_fixed_time(config: str, episodes: int) -> list:
    """Run fixed-time baseline evaluation on the grid."""
    print("\n" + "="*50)
    print("Running Fixed-Time Baseline (Grid)")
    print("="*50)

    env = MultiIntersectionEnv(sumocfg_path=config, use_gui=False)
    results = []

    for ep in range(1, episodes + 1):
        env.reset()
        controllers = {tid: FixedTimeController(num_phases=4, green_duration=30)
                       for tid in env.tls_ids}
        mc = MetricCollector()
        done = False

        while not done:
            actions = {tid: c.select_action() for tid, c in controllers.items()}
            next_states, rewards, done, info = env.step(actions)
            mc.collect_step(info)
            mc.total_reward += sum(rewards.values())
            for c in controllers.values(): c.update()

        mc.collect_end_of_episode()
        summary = mc.get_summary()
        results.append(summary)
        print(f"  Episode {ep}/{episodes}: "
              f"Reward={summary['total_reward']:.1f}, "
              f"CO2={summary['co2_kg']:.2f} kg, "
              f"Avg Wait={summary['avg_waiting_time']:.1f}s")

    env.close()
    return results


def run_dnlight_multi(config: str, checkpoint: str, episodes: int, 
                      green: bool = False) -> list:
    """Run multi-intersection agent evaluation (Standard or Green)."""
    mode_name = "Green AI" if green else "Standard DNLight"
    print("\n" + "="*50)
    print(f"Running {mode_name} (Grid)")
    print("="*50)

    env = MultiIntersectionEnv(
        sumocfg_path=config, use_gui=False,
        use_green_reward=green,
    )
    states = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    env.close()

    agent = DNLightAgent(state_dim, action_dim, use_attention=True)
    if os.path.exists(checkpoint + ".index") or os.path.exists(checkpoint):
        agent.load(checkpoint)
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  WARNING: Checkpoint {checkpoint} not found. Running untrained.")

    carbon = CarbonTracker()
    results = []

    for ep in range(1, episodes + 1):
        states = env.reset()
        mc = MetricCollector()
        done = False

        while not done:
            carbon.start_step()
            actions = {tid: agent.select_action(states[tid], training=False)
                       for tid in env.tls_ids}

            next_states, rewards, done, info = env.step(actions)
            carbon.end_step()
            mc.collect_step(info)
            mc.total_reward += sum(rewards.values())
            states = next_states

        mc.collect_end_of_episode()
        summary = mc.get_summary()
        summary['compute_carbon_gco2'] = carbon.reset_episode() * 1000
        results.append(summary)
        print(f"  Episode {ep}/{episodes}: "
              f"Reward={summary['total_reward']:.1f}, "
              f"CO2={summary['co2_kg']:.2f} kg, "
              f"Avg Wait={summary['avg_waiting_time']:.1f}s")

    env.close()
    return results


def aggregate_results(results: list) -> dict:
    if not results: return {}
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return agg


def print_comparison_table(fixed: dict, dnlight: dict, green: dict):
    print("\n" + "="*80)
    print("                    3-WAY COMPARISON RESULTS (GRID)")
    print("="*80)
    metrics = [
        ('Avg Waiting Time (s)', 'avg_waiting_time'),
        ('Avg Time Loss (s)', 'avg_time_loss'),
        ('EMV Avg Delay (s)', 'emv_avg_delay'),
        ('Throughput (vehicles)', 'throughput'),
        ('CO2 Emissions (kg)', 'co2_kg'),
        ('NOx Emissions (g)', 'nox_g'),
        ('Fuel Consumption (L)', 'fuel_liters'),
        ('Total Reward', 'total_reward'),
        ('Compute Carbon (mg)', 'compute_carbon_gco2'),
    ]
    header = f"{'Metric':<25} | {'Fixed-Time':>15} | {'DNLight Std':>15} | {'DNLight Green':>15}"
    print(header)
    print("-" * len(header))
    for label, key in metrics:
        def fmt(d, k):
            if k in d: return f"{d[k]['mean']:>8.1f}"
            return "N/A"
        print(f"{label:<25} | {fmt(fixed, key):>15} | {fmt(dnlight, key):>15} | {fmt(green, key):>15}")
    print("="*80)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    f_res = run_fixed_time(args.config, args.episodes)
    d_res = run_dnlight_multi(args.config, args.dnlight_ckpt, args.episodes, green=False)
    g_res = run_dnlight_multi(args.config, args.green_ckpt, args.episodes, green=True)

    f_agg, d_agg, g_agg = aggregate_results(f_res), aggregate_results(d_res), aggregate_results(g_res)
    print_comparison_table(f_agg, d_agg, g_agg)

    with open(os.path.join(args.output, 'comparison_results.json'), 'w') as f:
        json.dump({'fixed': f_agg, 'standard': d_agg, 'green': g_agg}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Green AI 3-Way Comparison', fontsize=16)
        chart_metrics = [('avg_waiting_time', 'Wait Time'), ('emv_avg_delay', 'EMV Delay'), 
                         ('co2_kg', 'CO2 (kg)'), ('fuel_liters', 'Fuel (L)'), 
                         ('compute_carbon_gco2', 'Compute CO2 (mg)'), ('total_reward', 'Reward')]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        labels = ['Fixed', 'Std', 'Green']
        for idx, (key, title) in enumerate(chart_metrics):
            ax = axes[idx // 3][idx % 3]
            vals = [f_agg.get(key, {}).get('mean', 0), d_agg.get(key, {}).get('mean', 0), g_agg.get(key, {}).get('mean', 0)]
            ax.bar(labels, vals, color=colors)
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'comparison_chart.png'))
    except Exception as e:
        print(f"Chart error: {e}")

if __name__ == "__main__":
    main()
