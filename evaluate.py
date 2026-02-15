"""
DNLight Evaluation Script.

Loads a trained checkpoint and runs evaluation episodes
with no exploration (greedy policy).

Usage:
    python evaluate.py --checkpoint checkpoints/dnlight_best --episodes 5
    python evaluate.py --checkpoint checkpoints/dnlight_best --gui
"""
import os
import argparse
import numpy as np
import time

from dnlight.environment import SumoEnvironment
from dnlight.agent import DNLightAgent


def parse_args():
    p = argparse.ArgumentParser(description="DNLight Evaluation")
    p.add_argument("--config", type=str,
                   default="configs/single_intersection.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of evaluation episodes")
    p.add_argument("--gui", action="store_true",
                   help="Use SUMO GUI")
    return p.parse_args()


def main():
    args = parse_args()

    # Initialize environment
    env = SumoEnvironment(
        sumocfg_path=args.config,
        use_gui=args.gui,
    )

    # Discover dims
    initial_state = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    env.close()

    # Initialize and load agent
    agent = DNLightAgent(state_dim, action_dim)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # ── Evaluation loop ──────────────────────────────────────────
    all_rewards = []
    all_steps = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)
        print(f"Episode {episode}: Reward = {ep_reward:.2f}, "
              f"Steps = {ep_steps}")

    env.close()

    # Summary
    print(f"\n{'='*40}")
    print(f"Evaluation Summary ({args.episodes} episodes)")
    print(f"{'='*40}")
    print(f"Mean Reward: {np.mean(all_rewards):.2f} "
          f"+/- {np.std(all_rewards):.2f}")
    print(f"Mean Steps:  {np.mean(all_steps):.1f}")
    print(f"Best Reward: {np.max(all_rewards):.2f}")


if __name__ == "__main__":
    main()
