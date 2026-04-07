import os
import argparse
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from dnlight.environment import MultiIntersectionEnv
from dnlight.agent import DNLightAgent

def parse_args():
    p = argparse.ArgumentParser(description="Live Matplotlib Visualization of DRL Q-Table")
    p.add_argument("--config", type=str,
                   default="configs/grid_network.sumocfg",
                   help="Path to SUMO .sumocfg file")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--gui", action="store_true",
                   help="Use SUMO GUI alongside the graph")
    p.add_argument("--green", action="store_true",
                   help="Use Green AI multi-agent structure")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds to pause chart updating per step")
    p.add_argument("--focus-tls", type=str, default="C01",
                   help="Which intersection's brain to graph (e.g., C00, C01, C10, C11)")
    return p.parse_args()

def main():
    args = parse_args()

    # Initialize environment
    env = MultiIntersectionEnv(
        sumocfg_path=args.config,
        use_gui=args.gui,
        use_green_reward=args.green,
        label="visualize_gui_q"
    )

    states = env.reset()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Initialize and load agent
    agent = DNLightAgent(state_dim, action_dim, use_attention=True)
    agent.load(args.checkpoint)
    print(f"\nLoaded neural network checkpoint: {args.checkpoint}")

    phase_names = [
        "Phase 0\n(NS Straight)",
        "Phase 1\n(NS Left)",
        "Phase 2\n(EW Straight)",
        "Phase 3\n(EW Left)"
    ]

    # Matplotlib Setup for Live Updating
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title("DNLight - Live Q-Table Agent Brain")

    target_tls = args.focus_tls
    if target_tls not in states:
        print(f"Warning: Intersection '{target_tls}' not found. Defaulting to first available.")
        target_tls = list(states.keys())[0]

    done = False
    ep_steps = 0
    last_info = None

    # Open log file to record all Q-table reports
    log_path = os.path.join(os.getcwd(), 'q_table_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"DNLight Q-Table Simulation Log\n")
    log_file.write(f"Checkpoint: {args.checkpoint}\n")
    log_file.write(f"Mode: {'Green AI' if args.green else 'Standard AI'}\n")
    log_file.write(f"Focus Intersection: {target_tls}\n")
    log_file.write("=" * 70 + "\n\n")
    print(f"Logging Q-tables to: {log_path}")

    while not done:
        ep_steps += 1
        actions = {}
        
        target_q_values = None
        target_state_info = None
        target_action = None

        for tid, s in states.items():
            state_t = tf.constant(s.reshape(1, -1), dtype=tf.float32)
            q_values = agent.online_net(state_t, training=False)[0].numpy()
            chosen_action = int(np.argmax(q_values))
            actions[tid] = chosen_action
            
            # Extract data for the targeted intersection to plot
            if tid == target_tls:
                target_q_values = q_values
                target_action = chosen_action
                
                # Deconstruct State Vector per-direction
                # 8 lanes total, 7 features each. Lanes sorted alphabetically.
                # In SUMO grid: lanes group as 2 per approach (N, E, S, W)
                n_lanes = len(s) // 7
                lanes_per_dir = max(n_lanes // 4, 1)
                dir_names = ['North', 'East', 'South', 'West']
                dir_queues = {}
                for d_idx, d_name in enumerate(dir_names):
                    start = d_idx * lanes_per_dir
                    end = min(start + lanes_per_dir, n_lanes)
                    q_sum = sum(s[l*7 + 1] for l in range(start, end)) * 250.0
                    dir_queues[d_name] = q_sum

                total_queue = sum(dir_queues.values())
                max_wait = max(s[2::7]) * 100.0     
                emv_present = sum(s[3::7]) > 0
                
                target_state_info = {
                    'queue': total_queue,
                    'dir_queues': dir_queues,
                    'wait': max_wait,
                    'emv': emv_present,
                    'co2': last_info['emissions'][tid].get('co2_mg_per_s', 0) / 1000.0 if last_info and 'emissions' in last_info else 0.0
                }

        # Render the Plot
        if target_q_values is not None:
            ax.clear()
            
            # Distinct colors: Gray for rejected options, Green/Red for chosen
            colors = ['#bdc3c7'] * 4 
            if target_state_info['emv']:
                colors[target_action] = '#e74c3c'  # Red for Emergency chosen
            else:
                colors[target_action] = '#2ecc71'  # Green for standard chosen
                
            bars = ax.bar(phase_names, target_q_values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Decorate the bars with their exact numerical Q-Value
            for bar in bars:
                height = bar.get_height()
                y_offset = height + (100 if height >= 0 else -300)
                ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                        f'{height:.1f}',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold', fontsize=11)

            # Titles & State Data Overlay
            emv_status = "DETECTED (Priority Active)" if target_state_info['emv'] else "None"
            dq = target_state_info['dir_queues']
            title = (
                f"Intersection Brain: {target_tls}   |   Simulation Step: {ep_steps * env.step_duration}s\n"
                f"Queue North: {dq['North']:.0f}m | Queue East: {dq['East']:.0f}m | Queue South: {dq['South']:.0f}m | Queue West: {dq['West']:.0f}m\n"
                f"Max Wait: {target_state_info['wait']:.1f}s   |   EMV: {emv_status}   |   CO2: {target_state_info['co2']:.1f} g/s"
            )
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel("Neural Network Assessed Q-Value", fontsize=12, fontweight='bold')
            ax.set_xlabel("Signal Phase Actions", fontsize=12, fontweight='bold')
            
            # Dynamic Y-Axis scaling
            min_q = min(target_q_values)
            max_q = max(target_q_values)
            margin = max(abs(max_q - min_q) * 0.3, 1000.0)
            ax.set_ylim(min_q - margin, max_q + margin)

            # A baseline to show 0
            ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.draw()
            # Wait 'delay' seconds, rendering the GUI event loop
            plt.pause(args.delay)

        # Print Text Q-Table to Terminal every 100 simulation seconds
        current_time = ep_steps * env.step_duration
        if current_time % 100 == 0:
            print("\n" + "="*70)
            print(f"=== PERIODIC Q-TABLE REPORT (STEP: {current_time} SECONDS) ===")
            print("="*70)
            for tid, s in states.items():
                state_t = tf.constant(s.reshape(1, -1), dtype=tf.float32)
                q_vals = agent.online_net(state_t, training=False)[0].numpy()
                chosen = int(np.argmax(q_vals))
                
                n_lanes = len(s) // 7
                lanes_per_dir = max(n_lanes // 4, 1)
                dir_names = ['North', 'East', 'South', 'West']
                dir_q_strs = []
                for d_idx, d_name in enumerate(dir_names):
                    start = d_idx * lanes_per_dir
                    end = min(start + lanes_per_dir, n_lanes)
                    q_sum = sum(s[l*7 + 1] for l in range(start, end)) * 250.0
                    dir_q_strs.append(f"Queue {d_name}: {q_sum:.0f}m")
                max_w = max(s[2::7]) * 100.0     
                emv_p = sum(s[3::7]) > 0
                co2_val = last_info['emissions'][tid].get('co2_mg_per_s', 0) / 1000.0 if last_info and 'emissions' in last_info else 0.0
                
                print(f"\n[INTERSECTION: {tid}]")
                print(f"  State -> {' | '.join(dir_q_strs)}")
                print(f"  Max Wait: {max_w:.1f}s | EMV: {'YES' if emv_p else 'No'} | CO2: {co2_val:.1f} g/s")
                print("  [Q-TABLE EVALUATION]")
                for i, q in enumerate(q_vals):
                    clean_name = phase_names[i].replace('\n', ' ')
                    marker = "<-- ✨ AI CHOSE THIS PHASE ✨" if i == chosen else ""
                    print(f"    {clean_name:<30} | Q-Value: {q:>8.2f}    {marker}")
            print("\n" + "="*70 + "\n")
            # Also write to log file
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"=== Q-TABLE REPORT (STEP: {current_time} SECONDS) ===\n")
            log_file.write(f"{'='*70}\n")
            for tid, s in states.items():
                state_t2 = tf.constant(s.reshape(1, -1), dtype=tf.float32)
                q_vals2 = agent.online_net(state_t2, training=False)[0].numpy()
                chosen2 = int(np.argmax(q_vals2))
                n_l = len(s) // 7
                lpd = max(n_l // 4, 1)
                dn = ['North', 'East', 'South', 'West']
                dqs = []
                for di, dname in enumerate(dn):
                    st = di * lpd
                    en = min(st + lpd, n_l)
                    qs = sum(s[l*7+1] for l in range(st, en)) * 250.0
                    dqs.append(f"Queue {dname}: {qs:.0f}m")
                mw = max(s[2::7]) * 100.0
                ep = sum(s[3::7]) > 0
                cv = last_info['emissions'][tid].get('co2_mg_per_s', 0) / 1000.0 if last_info and 'emissions' in last_info else 0.0
                log_file.write(f"\n[INTERSECTION: {tid}]\n")
                log_file.write(f"  State -> {' | '.join(dqs)}\n")
                log_file.write(f"  Max Wait: {mw:.1f}s | EMV: {'YES' if ep else 'No'} | CO2: {cv:.1f} g/s\n")
                log_file.write(f"  [Q-TABLE EVALUATION]\n")
                for i, q in enumerate(q_vals2):
                    cn = phase_names[i].replace('\n', ' ')
                    mk = "<-- AI CHOSE THIS PHASE" if i == chosen2 else ""
                    log_file.write(f"    {cn:<30} | Q-Value: {q:>8.2f}    {mk}\n")
            log_file.write(f"\n{'='*70}\n\n")
            log_file.flush()

        # Step Environment Forward
        next_states, rewards, done, last_info = env.step(actions)
        states = next_states

    plt.ioff()
    plt.show()
    log_file.close()
    print(f"\nComplete Q-Table log saved to: {log_path}")
    env.close()

if __name__ == "__main__":
    main()
