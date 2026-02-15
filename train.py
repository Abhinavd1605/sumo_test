"""
DNLight Training Script.

Runs the DNLight agent on a SUMO intersection for multiple episodes,
with logging, checkpointing, and TensorBoard support.

Usage:
    python train.py --episodes 100 --no-gui
    python train.py --episodes 5 --gui   # visual mode
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf

from dnlight.environment import SumoEnvironment
from dnlight.agent import DNLightAgent, BATCH_SIZE


def parse_args():
    p = argparse.ArgumentParser(description="DNLight Training")
    p.add_argument("--config", type=str,
                   default="configs/single_intersection.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of training episodes")
    p.add_argument("--gui", action="store_true",
                   help="Use SUMO GUI for visualization")
    p.add_argument("--no-gui", dest="gui", action="store_false")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for model checkpoints")
    p.add_argument("--log-dir", type=str, default="logs",
                   help="Directory for TensorBoard logs")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save checkpoint every N episodes")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.set_defaults(gui=False)
    return p.parse_args()


def main():
    args = parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # TensorBoard writer
    tb_writer = tf.summary.create_file_writer(args.log_dir)

    # Initialize environment
    print(f"Initializing SUMO environment from: {args.config}")
    env = SumoEnvironment(
        sumocfg_path=args.config,
        use_gui=args.gui,
    )

    # First reset to discover state/action dims
    initial_state = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    env.close()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Incoming lanes: {len(env.incoming_lanes)}")

    # Initialize agent
    agent = DNLightAgent(state_dim, action_dim)
    print(f"Online network parameters: "
          f"{agent.online_net.count_params():,}")

    # Resume from checkpoint if specified
    start_episode = 0
    best_reward = -float('inf')

    if args.resume:
        print(f"Resuming from: {args.resume}")
        agent.load(args.resume)
        # Load training state
        state_path = os.path.join(args.checkpoint_dir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                ts = json.load(f)
            start_episode = ts.get('episode', 0)
            best_reward = ts.get('best_reward', -float('inf'))
            print(f"  Resuming from episode {start_episode}, "
                  f"best_reward={best_reward:.2f}")

    # ── Training loop ────────────────────────────────────────────
    total_episodes = start_episode + args.episodes
    print(f"\n{'='*60}")
    print(f"Training episodes {start_episode+1} to {total_episodes}")
    print(f"{'='*60}\n")

    for episode in range(start_episode + 1, total_episodes + 1):
        ep_start = time.time()
        state = env.reset()

        ep_reward = 0.0
        ep_steps = 0
        ep_losses = []
        ep_td_errors = []

        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            train_info = agent.train()
            if train_info:
                ep_losses.append(train_info['loss'])
                ep_td_errors.append(train_info['td_error_mean'])

            state = next_state
            ep_reward += reward
            ep_steps += 1

        ep_time = time.time() - ep_start

        # ── Logging ──────────────────────────────────────────────
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        avg_td = np.mean(ep_td_errors) if ep_td_errors else 0.0

        print(f"Episode {episode:4d}/{total_episodes} | "
              f"Steps: {ep_steps:4d} | "
              f"Reward: {ep_reward:10.2f} | "
              f"Loss: {avg_loss:8.4f} | "
              f"TD: {avg_td:8.4f} | "
              f"Eps: {agent.epsilon:.4f} | "
              f"Noise: {agent.noise_scale:.4f} | "
              f"Buffer: {len(agent.buffer):,} | "
              f"Time: {ep_time:.1f}s")

        # TensorBoard
        with tb_writer.as_default():
            tf.summary.scalar("episode/reward", ep_reward, step=episode)
            tf.summary.scalar("episode/steps", ep_steps, step=episode)
            tf.summary.scalar("episode/avg_loss", avg_loss, step=episode)
            tf.summary.scalar("episode/avg_td_error", avg_td, step=episode)
            tf.summary.scalar("episode/epsilon", agent.epsilon, step=episode)
            tf.summary.scalar("episode/noise_scale",
                              agent.noise_scale, step=episode)
            tf.summary.scalar("episode/buffer_size",
                              len(agent.buffer), step=episode)

        # ── Checkpoint ───────────────────────────────────────────
        if episode % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"dnlight_ep{episode}"
            )
            agent.save(ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = os.path.join(args.checkpoint_dir, "dnlight_best")
            agent.save(best_path)

        # Save training state for resume
        state_path = os.path.join(args.checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump({'episode': episode, 'best_reward': float(best_reward)}, f)

    # ── Cleanup ──────────────────────────────────────────────────
    env.close()
    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    print(f"Checkpoints in: {args.checkpoint_dir}")
    print(f"TensorBoard logs in: {args.log_dir}")
    print(f"  Run: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
