# PokerTrainer (Aequus)

> **GPU-Native Poker AI: Real-time, high-performance poker bot and trainer using JAX, Cython, and modern CFR algorithms.**

---

## 🚀 Overview
PokerTrainer (Aequus) is a state-of-the-art, GPU-accelerated poker AI framework for training and playing No-Limit Texas Hold'em. It leverages JAX for fast vectorized computation, Cython for ultra-fast hashing, and advanced Counterfactual Regret Minimization (CFR) algorithms—including MCCFR, PDCFR+, and hybrid approaches. Designed for both research and real-world play, it supports multi-GPU, dynamic memory management, and real-time decision making.

---

## ✨ Features
- **GPU-accelerated training** with JAX and XLA
- **Modern CFR algorithms**: MCCFR, PDCFR+, Hybrid, and more
- **Ultra-fast hand evaluation** (400M+ hands/sec) via `phevaluator` and JAX
- **Cython-optimized hashing** for large-scale info set management
- **Realistic poker engine**: NLHE rules, multi-player, stack/blind config
- **CLI for training, playing, benchmarking, and evaluation**
- **Memory-efficient**: Adaptive batch sizes, gradient checkpointing, and smart caching
- **Easy extensibility** for new algorithms and research

---

## 🗂️ Project Structure
```
Aequus/
├── poker_bot/
│   ├── core/           # Core training, simulation, hashing (Cython)
│   ├── bot.py          # Main PokerBot class (AI agent)
│   ├── cli.py          # Command-line interface (training, play, etc.)
│   ├── evaluator.py    # Ultra-fast hand evaluator
│   ├── memory.py       # Memory management utilities
│   ├── gpu_config.py   # GPU/XLA configuration
│   └── ...
├── models/             # Trained models (not versioned)
├── config/             # Training and game configuration YAMLs
├── scripts/            # Setup and utility scripts
├── requirements.txt    # Python dependencies
├── setup.py            # Install script (pip/venv)
└── README.md           # This file
```

---

## ⚡ Installation
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for full acceleration)
- [JAX with CUDA support](https://github.com/google/jax#installation)

### Quick Install
```bash
# Clone the repo
$ git clone https://github.com/Sn0wfly/Aequus.git
$ cd Aequus

# (Recommended) Create a virtual environment
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies and compile Cython extensions
$ pip install -r requirements.txt
$ pip install .
```

> **Note:** For GPU support, install JAX as per [official instructions](https://github.com/google/jax#installation) for your CUDA version.

---

## 🕹️ Usage
### Command-Line Interface (CLI)
All features are accessible via the `poker-bot` command:

#### Train a model
```bash
poker-bot train --iterations 100000 --batch-size 8192 --players 2 --learning-rate 0.01 --save-path models/mccfr_model.pkl
```

#### Play against the bot
```bash
poker-bot play --model models/mccfr_model.pkl --hands 100 --opponents 1 --stack 100 --aggressive
```

#### Evaluate components
```bash
poker-bot evaluate
```

#### See all commands and options
```bash
poker-bot --help
```

---

## 🏗️ Configuration
- **Training:** Edit `config/training_config.yaml` to customize iterations, batch size, learning rate, abstraction, and more.
- **GPU/Memory:** `poker_bot/gpu_config.py` and `poker_bot/memory.py` provide advanced tuning for XLA flags, mixed precision, and adaptive batching.
- **Cython:** The hasher is auto-compiled via `setup.py` for maximum speed.

---

## 🧠 Algorithms & Architecture
- **CFR Variants:** MCCFR, PDCFR+, Hybrid, Parallel, and more (see CLI options)
- **Hand Evaluation:** Uses `phevaluator` and JAX for batch and single-hand evaluation
- **Trainer:** Vectorized, JIT-compiled, and supports dynamic info set growth
- **Bot:** Real-time decision making, configurable aggression/bluff/randomization

---

## 📝 Example: Vast.ai Setup
For cloud GPU training:
```bash
bash scripts/setup_vast_ai.sh
# Then inside the environment:
source /opt/poker_env/bin/activate
poker-bot train --iterations 10000
```

---

## 🤝 Contributing
Pull requests, issues, and research collaborations are welcome! Please open an issue or PR on GitHub.

---

## 👤 Credits
- PokerTrainer Team
- Based on open-source research in poker AI and reinforcement learning

---

## 📄 License
MIT License. See `LICENSE` file for details.

---

## 💡 References
- [JAX](https://github.com/google/jax)
- [phevaluator](https://github.com/andyhugh/phevaluator)
- [Counterfactual Regret Minimization](https://en.wikipedia.org/wiki/Counterfactual_regret_minimization)

---

> For questions or support, open an issue on [GitHub](https://github.com/Sn0wfly/Aequus). 