"""
Multi-Agent Training Script for DNLight + Green AI.

Trains a shared DNLight agent across all intersections in a multi-
intersection grid, using attention-based communication and the
Green AI reward function with CO2 penalties.

Usage:
    python train_multi.py --episodes 100 --no-gui
    python train_multi.py --episodes 5 --gui
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent, BATCH_SIZE
from dnlight.carbon_tracker import CarbonTracker


def parse_args():
    p = argparse.ArgumentParser(description="DNLight Multi-Agent Training")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Path to grid SUMO .sumocfg file")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--no-gui", dest="gui", action="store_false")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_multi")
    p.add_argument("--log-dir", type=str, default="logs_multi")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--green", action="store_true", default=True,
                   help="Use green reward with CO2 penalty")
    p.add_argument("--no-green", dest="green", action="store_false",
                   help="Use standard DNLight reward")
    p.add_argument("--label", type=str, default="multi",
                   help="Unique label for TraCI connection")
    p.add_argument("--alpha-co2", type=float, default=0.3,
                   help="CO2 penalty weight in green reward")
    p.set_defaults(gui=False)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    tb_writer = tf.summary.create_file_writer(args.log_dir)

    # Initialize multi-intersection environment
    print(f"Initializing multi-intersection environment: {args.config}")
    env = MultiIntersectionEnv(
        sumocfg_path=args.config,
        use_gui=args.gui,
        label=args.label,
        use_green_reward=args.green,
        alpha_co2=args.alpha_co2,
    )

    # First reset to discover dimensions
    initial_states = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    num_intersections = env.num_intersections
    env.close()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Number of intersections: {num_intersections}")
    print(f"TLS IDs: {env.tls_ids}")

    # Single shared agent (parameter sharing across intersections)
    agent = DNLightAgent(state_dim, action_dim, use_attention=True)
    print(f"Online network parameters: {agent.online_net.count_params():,}")

    # Carbon tracker
    carbon = CarbonTracker()

    # Resume
    start_episode = 0
    best_reward = -float('inf')

    if args.resume:
        print(f"Resuming from: {args.resume}")
        agent.load(args.resume)
        state_path = os.path.join(args.checkpoint_dir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                ts = json.load(f)
            start_episode = ts.get('episode', 0)
            best_reward = ts.get('best_reward', -float('inf'))
            print(f"  Resuming from episode {start_episode}, "
                  f"best_reward={best_reward:.2f}")

    # Training loop
    total_episodes = start_episode + args.episodes
    print(f"\n{'='*60}")
    print(f"Training episodes {start_episode+1} to {total_episodes}")
    print(f"Green AI mode: CO2 penalty alpha={args.alpha_co2}")
    print(f"{'='*60}\n")

    for episode in range(start_episode + 1, total_episodes + 1):
        ep_start = time.time()
        states = env.reset()

        ep_reward = 0.0
        ep_steps = 0
        ep_losses = []
        ep_td_errors = []
        ep_co2 = 0.0
        ep_fuel = 0.0

        done = False
        while not done:
            # Select actions for all intersections
            carbon.start_step()
            actions = {}
            for tls_id in env.tls_ids:
                actions[tls_id] = agent.select_action(
                    states[tls_id], training=True
                )

            # Step environment
            next_states, rewards, done, info = env.step(actions)
            carbon.end_step()

            # Store transitions and train (for each intersection)
            for tls_id in env.tls_ids:
                agent.store_transition(
                    states[tls_id], actions[tls_id],
                    rewards[tls_id], next_states[tls_id], done
                )

            # Train (once per step, shared agent)
            train_info = agent.train()
            if train_info:
                ep_losses.append(train_info['loss'])
                ep_td_errors.append(train_info['td_error_mean'])

            states = next_states
            ep_reward += sum(rewards.values())
            ep_steps += 1

            # Track emissions
            if 'emissions' in info:
                for em_data in info['emissions'].values():
                    ep_co2 += em_data.get('co2_mg_per_s', 0)
                    ep_fuel += em_data.get('fuel_ml_per_s', 0)

        ep_time = time.time() - ep_start
        ep_carbon = carbon.reset_episode()

        # Logging
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        avg_td = np.mean(ep_td_errors) if ep_td_errors else 0.0
        co2_kg = ep_co2 / 1e6  # mg -> kg

        print(f"Episode {episode:4d}/{total_episodes} | "
              f"Steps: {ep_steps:4d} | "
              f"Reward: {ep_reward:10.2f} | "
              f"Loss: {avg_loss:8.4f} | "
              f"CO2: {co2_kg:6.2f} kg | "
              f"Fuel: {ep_fuel/1000:.1f} L | "
              f"Compute CO2: {ep_carbon*1000:.3f} mg | "
              f"Eps: {agent.epsilon:.4f} | "
              f"Time: {ep_time:.1f}s")

        # TensorBoard
        with tb_writer.as_default():
            tf.summary.scalar("episode/reward", ep_reward, step=episode)
            tf.summary.scalar("episode/steps", ep_steps, step=episode)
            tf.summary.scalar("episode/avg_loss", avg_loss, step=episode)
            tf.summary.scalar("episode/avg_td_error", avg_td, step=episode)
            tf.summary.scalar("episode/epsilon", agent.epsilon, step=episode)
            tf.summary.scalar("episode/co2_kg", co2_kg, step=episode)
            tf.summary.scalar("episode/fuel_liters", ep_fuel/1000,
                              step=episode)
            tf.summary.scalar("episode/compute_carbon_mg",
                              ep_carbon * 1000, step=episode)

        # Checkpoint
        if episode % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"dnlight_green_ep{episode}"
            )
            agent.save(ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(args.checkpoint_dir, "dnlight_green_best"))

        # Save training state
        state_path = os.path.join(args.checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump({
                'episode': episode,
                'best_reward': float(best_reward)
            }, f)

    # Summary
    env.close()
    c_summary = carbon.get_summary()
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    print(f"Total compute carbon: {c_summary['total_carbon_kg']:.6f} kg CO2")
    print(f"Checkpoints in: {args.checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
