"""
DNLight Multi-Intersection Evaluation Script (GUI version).

Usage:
    python evaluate_multi.py --checkpoint checkpoints_multi_std_ext/dnlight_std_best --gui
    python evaluate_multi.py --checkpoint checkpoints_multi_green_ext/dnlight_green_best --gui --green
"""
import os
import argparse
import numpy as np
import time

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent

def parse_args():
    p = argparse.ArgumentParser(description="DNLight Multi-Intersection Evaluation")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of evaluation episodes")
    p.add_argument("--gui", action="store_true",
                   help="Use SUMO GUI")
    p.add_argument("--green", action="store_true",
                   help="Use Green reward / Multi-agent structure")
    return p.parse_args()

def main():
    args = parse_args()

    # Initialize environment
    env = MultiIntersectionEnv(
        sumocfg_path=args.config,
        use_gui=args.gui,
        label="eval_gui"
    )

    # Discover dims
    states = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize and load agent
    agent = DNLightAgent(state_dim, action_dim, use_attention=True)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluation loop ──────────────────────────────────────────
    for episode in range(1, args.episodes + 1):
        states = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        print(f"Starting Episode {episode}...")
        while not done:
            # Multi-agent action selection
            actions = {tid: agent.select_action(s, training=False) 
                       for tid, s in states.items()}
            
            next_states, rewards, done, info = env.step(actions)
            ep_reward += sum(rewards.values())
            ep_steps += 1
            states = next_states

            # Optional: slow down for observation if GUI
            if args.gui:
                time.sleep(0.05)

        print(f"Episode {episode} Complete | Reward: {ep_reward:.2f} | Steps: {ep_steps}")

    env.close()

if __name__ == "__main__":
    main()
