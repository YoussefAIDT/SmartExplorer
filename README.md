# SmartExplorer: Autonomous RL-Based Environmental Mapping

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**SmartExplorer** is an advanced Reinforcement Learning (RL) platform designed to simulate and compare various autonomous exploration agents in procedurally generated 2D environments. The project focuses on solving the **Partial Observability** problem (POMDP) where an agent must achieve 100% map coverage while navigating obstacles and collecting valuable resources.

## 🚀 Key Features

- **Procedural World Generation**: Dynamic generation of complex Mazes (DFS-based) and Open World environments.
- **Advanced RL Suite**: 6 custom-implemented algorithms using PyTorch (no high-level wrappers like Stable Baselines3).
- **Scientific Comparison System**: Fairness-driven benchmarking where all models are tested on the exact same set of pre-generated maps.
- **Advanced Metrics Tracking**: Real-time evaluation of:
  - **Map Coverage**: Percentage of the explorable area visited.
  - **Exploration Efficiency**: Discovered cells per step.
  - **Path Redundancy**: Overlap and revisit analysis.
  - **Resource Collection Rate**: Efficiency in gathering GOLD and DIAMOND items.
- **Dynamic Visualization**: Real-time 2D rendering with Pygame, including custom overlays and Fog of War mechanics.

---

## 🧠 Implemented Algorithms

Each algorithm is implemented from scratch in the `strategies/` directory to demonstrate deep understanding of RL mechanics.

| Algorithm | Type | Key Feature |
| :--- | :--- | :--- |
| **DQN** | Value-Based | Standard Deep Q-Network with Target Network and Replay Buffer. |
| **PPO** | Policy-Based | Proximal Policy Optimization for stable, high-performance exploration. |
| **A2C** | Actor-Critic | Synchronous Advantage Actor-Critic for balanced learning. |
| **SAC** | Actor-Critic | Soft Actor-Critic maximizing entropy for robust exploration. |
| **Rainbow DQN** | Hybrid | Combined improvements (Dueling, Double) for enhanced stability. |
| **Curiosity** | Intrinsic | ICM (Intrinsic Curiosity Module) rewarding agents for exploring "surprising" areas. |

---

## 🛠️ Project Structure

```text
├── agent/               # Autonomous explorer logic
├── environment/         # GridWorld, Map Generation, and Rendering
├── strategies/          # Custom PyTorch RL implementations (DQN, PPO, etc.)
├── utils/               # Metrics, Logging, and Comparison tools
├── saved_maps/          # Screenshots of explored environments
├── comparison_plots/    # Statistical performance graphs
├── config.py            # Global simulation & algorithm parameters
├── main.py              # Individual simulation entry point
└── main_advanced.py     # Multi-algorithm comparison system
```

---

## 🏁 Getting Started

### Prerequisites
- Python 3.8+
- [Pygame](https://www.pygame.org/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

### Installation
```bash
git clone https://github.com/yourusername/SmartExplorer.git
cd SmartExplorer
pip install -r requirements.txt
```

### Usage
**Run a single simulation (DQN by default):**
```bash
python main.py
```

**Run a scientific comparison across all models:**
```bash
python main_advanced.py
```
*Note: This will train each of the 6 models for 30 episodes on the same set of maps and output comparison plots.*

---

## 📊 Evaluation Results

Initial benchmarks show that **PPO** and **Curiosity-DQN** excel in complex maze environments by maintaining high exploratory pressure even when rewards are sparse. **SAC**, while powerful in continuous spaces, requires fine-tuning for this discrete grid world to avoid high path redundancy.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- **[Your Name/Team Name]** - *Initial Work & Architecture*
