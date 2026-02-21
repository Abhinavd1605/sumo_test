# Green AI Traffic Optimization with Multi-Intersection DNLight

This repository contains an implementation of the **DNLight** algorithm (Double Dueling DQN + Attention) extended with **Green AI** features for multi-intersection traffic optimization and environmental impact reduction.

## 🚀 Key Features

- **Multi-Intersection Control**: Orchestrates a 2x2 grid of traffic lights using an attention-based communication mechanism between fog nodes.
- **Green AI Reward**: A multi-objective reward function that balances traffic delay, emergency vehicle (EMV) priority, and **CO₂ emission reduction**.
- **Computational Carbon Tracking**: Real-time tracking of AI inference carbon footprint using the paper's formula: $\Gamma_j(t) = CI_j(t) \cdot E_{j,fog}(t) / 3.6e6$.
- **Advanced DRL Architecture**: Double Dueling DQN with Noisy Networks (structured exploration) and Prioritized Experience Replay (PER).

## 📊 Performance Results

The system was evaluated against a traditional Fixed-Time baseline on a stochastic 2x2 grid network.

| Metric | Fixed-Time (Baseline) | DNLight Green (Ep 172) |
| :--- | :--- | :--- |
| **CO₂ Emissions** | 26.8 ± 0.1 kg | **17.9 ± 9.9 kg** (↓ 33%) |
| **NOx Emissions** | 11.7 ± 0.1 g | **7.6 ± 4.6 g** (↓ 35%) |
| **EMV Avg Delay** | 0.0s | **0.0s** (Priority Maintained) |
| **Compute Carbon** | N/A | 23.7 mg (Insignificant vs Savings) |

*The Green AI model achieves significant environmental savings while maintaining superior flow stability compared to unconstrained RL models.*

## 🛠️ Installation

1. **Prerequisites**:
   - [SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility)
   - Python 3.9+
   - TensorFlow 2.x

2. **Setup Env**:
   ```bash
   conda create -n sumo_drl python=3.9
   conda activate sumo_drl
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Training
To train the multi-intersection Green AI model:
```bash
python train_multi.py --green --episodes 400 --checkpoint-dir checkpoints_multi_green
```

### 2. Evaluation (GUI)
Visualize a single episode in SUMO-GUI:
```bash
python evaluate_multi.py --checkpoint checkpoints_multi_green_ext/dnlight_green_best --gui --green
```

### 3. Comparison
Run a 3-way performance comparison between Fixed-Time, Standard DNLight, and Green AI:
```bash
python compare.py --episodes 10 --output results_final
```

## 🏗️ Project Structure

- `dnlight/`: Core implementation (environment, agent, reward, carbon tracker).
- `configs/`: SUMO network files and grid generation scripts.
- `train_multi.py`: Multi-agent training pipeline.
- `compare.py`: Benchmark evaluation framework.
- `evaluate_multi.py`: Visualization and single-episode testing.

---
*Based on the research paper "DNLight: Deep Reinforcement Learning-Driven Traffic Signal Control with Green AI Extensions".*