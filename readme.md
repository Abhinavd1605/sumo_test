# Project Setup & Reproduction Guide
**Green AI Driven Multi-Agent Traffic Signal Control for Emission Reduction and Emergency Vehicle Prioritisation**

This guide provides the exact steps required to install, configure, and run the Deep Reinforcement Learning (DRL) traffic control system.

---

## 1. Prerequisites

Before setting up the Python environment, ensure the following are installed:

1.  **Python 3.9 or 3.10**: Recommended for compatibility with TensorFlow 2.10.
2.  **Eclipse SUMO (Simulation of Urban MObility)**:
    *   [Download SUMO](https://www.eclipse.org/sumo/download/)
    *   **IMPORTANT**: After installation, you must add a system environment variable named `SUMO_HOME` pointing to your SUMO installation directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).
3.  **Conda (Optional but Recommended)**: Use Miniconda or Anaconda to manage dependencies.

---

## 2. Installation

Follow these steps to set up the local environment:

```bash
# 1. Create a dedicated virtual environment
conda create -n sumo_drl python=3.9
conda activate sumo_drl

# 2. Install core dependencies
pip install tensorflow>=2.10.0 numpy matplotlib traci

# 3. Verify SUMO is accessible
# Run this in your terminal; it should print the SUMO version
sumo
```

---

## 3. Project Structure Overview

*   `dnlight/`: Core logic (Agent architecture, Reward functions, SUMO Environment wrappers).
*   `configs/`: Traffic network files (`.sumocfg`, `.net.xml`, `.rou.xml`).
*   `checkpoints/`: Pre-trained model weights for both Standard and Green AI agents.
*   `train_multi.py`: Main script for training agents on the grid network.
*   `evaluate_multi.py`: Script for running a trained model with GUI to observe behavior.
*   `compare_two_ai.py`: Generates head-to-head comparison charts between Standard and Green AI.
*   `visualize_q_table.py`: Live graphical dashboard showing the AI's internal "brain" decisions.

---

## 4. Run Commands Cheat Sheet

### 4.1 Training
To train the agents from scratch:

**Standard AI (Delay-based only):**
```bash
python train_multi.py --episodes 400 --no-green --label std_run
```

**Green AI (CO2 + EMV Prioritisation):**
```bash
python train_multi.py --episodes 400 --green --label green_run
```

---

### 4.2 Visualizing AI Logic (Presentation Tools)
These scripts are designed for live demonstrations and defense presentations.

**Live Q-Table Dashboard:**
Shows a real-time bar chart of the AI's Q-values and current state (Queue, Wait Time, CO2, EMV status).
```bash
python visualize_q_table.py --checkpoint checkpoints_multi_green_ext/dnlight_green_best --green --gui --delay 0.5 --focus-tls C11
```

**Step-by-Step Brain Trace:**
Prints the exact numerical Q-table to the terminal at every step.
```bash
python watch_agent_brain.py --checkpoint checkpoints_multi_green_ext/dnlight_green_best --green --delay 1.0
```

---

### 4.3 Evaluation & Comparison
To generate performance metrics and comparison data.

**Evaluate a Single Model (Manual GUI Inspection):**
```bash
python evaluate_multi.py --checkpoint checkpoints_multi_green_ext/dnlight_green_best --gui --green
```

**Automatic AI-to-AI Comparison:**
Runs 5 episodes of each model and generates a comparison chart in `results_ai_comparison/`.
```bash
python compare_two_ai.py --episodes 5
```

**Baseline vs. Green AI Comparison:**
Compares the Green AI against a Fixed-Time baseline and generates a comparative report.
```bash
python compare_baseline_vs_green.py --episodes 5
```

---

## 6. Running on Non-GPU (CPU-only) Setups

This project is fully compatible with machines that do not have a dedicated NVIDIA GPU. 

1.  **Automatic Fallback**: TensorFlow is designed to automatically detect your hardware. If no compatible GPU (CUDA) is found, it will seamlessly switch to your CPU for all neural network calculations.
2.  **Performance**: Since the traffic simulation (SUMO) is the most resource-intensive part, a modern CPU is more than sufficient for running evaluations and small-scale training.
3.  **Forcing CPU Mode**: If you have a GPU but want to force the project to use the CPU (e.g., to save memory), you can set an environment variable before running your command:
    *   **Windows (PowerShell)**: `$env:CUDA_VISIBLE_DEVICES="-1"; python evaluate_multi.py ...`
    *   **Windows (Command Prompt)**: `set CUDA_VISIBLE_DEVICES=-1 && python evaluate_multi.py ...`
    *   **Linux/Mac**: `CUDA_VISIBLE_DEVICES=-1 python evaluate_multi.py ...`

