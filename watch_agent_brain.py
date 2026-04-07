import os
import argparse
import numpy as np
import tensorflow as tf
import time

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent

def parse_args():
    p = argparse.ArgumentParser(description="Watch the Agent's Brain (Q-Values) in Real-Time")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--gui", action="store_true",
                   help="Use SUMO GUI alongside the terminal output")
    p.add_argument("--green", action="store_true",
                   help="Use Green AI multi-agent structure")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to pause terminal output between steps so you can read it")
    return p.parse_args()

def main():
    args = parse_args()

    # Initialize environment
    env = MultiIntersectionEnv(
        sumocfg_path=args.config,
        use_gui=args.gui,
        use_green_reward=args.green,
        label="watch_brain"
    )

    states = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize and load agent
    agent = DNLightAgent(state_dim, action_dim, use_attention=True)
    agent.load(args.checkpoint)
    print(f"\nLoaded neural network checkpoint: {args.checkpoint}")
    time.sleep(2)  # Give user a chance to read the load confirmation

    phase_names = [
        "Phase 0: North-South Straight Green",
        "Phase 1: North-South Left Turn Green ",
        "Phase 2: East-West Straight Green  ",
        "Phase 3: East-West Left Turn Green   "
    ]

    done = False
    ep_steps = 0

    while not done:
        ep_steps += 1
        actions = {}
        
        # Clear screen for a live-updating dashboard feel
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*65)
        print(f"=== SIMULATION STEP: {ep_steps * env.step_duration} SECONDS ===")
        print(f"Agent Model: {'Green AI' if args.green else 'Standard AI'}")
        print("="*65)

        for tid, s in states.items():
            # Deconstruct the State Vector to show the panel exactly what the AI "sees"
            # The state vector has 7 features per lane. 
            total_queue = sum(s[1::7]) * 250.0  # Denormalized queue length (Detection Range = 250)
            max_wait = max(s[2::7]) * 100.0     # Denormalized wait time
            emv_present = sum(s[3::7]) > 0
            
            print(f"\n[INTERSECTION: {tid}]")
            print("  [STATE OBSERVATION]")
            print(f"  > Total Visible Queue: {total_queue:.1f} meters")
            print(f"  > Maximum Wait Time:   {max_wait:.1f} seconds")
            print(f"  > EMV Detected:        {'YES 🚑' if emv_present else 'No'}")
            
            # Intercept the Q-Values straight from the neural network
            state_t = tf.constant(s.reshape(1, -1), dtype=tf.float32)
            q_values = agent.online_net(state_t, training=False)[0].numpy()
            chosen_action = int(np.argmax(q_values))
            actions[tid] = chosen_action
            
            print("\n  [NEURAL NETWORK Q-TABLE EVALUATION]")
            for i, q in enumerate(q_values):
                marker = "<-- ✨ AI CHOSE THIS PHASE ✨" if i == chosen_action else ""
                print(f"  {phase_names[i]} | Q-Value: {q:>8.2f}    {marker}")
                
        print("\n" + "="*65)
        print("Continuing to next simulation step...")
        
        next_states, rewards, done, info = env.step(actions)
        states = next_states

        # Pause script execution slightly so the human can read the live terminal table
        time.sleep(args.delay)

    env.close()
    print("\nSimulation Complete.")

if __name__ == "__main__":
    main()
