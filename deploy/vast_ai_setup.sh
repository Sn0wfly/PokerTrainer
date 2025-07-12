#!/bin/bash

# 🚀 PokerBot Deployment Script for Vast.ai
# This script sets up the complete environment for training on vast.ai

set -e  # Exit on any error

echo "🎯 Starting PokerBot deployment on vast.ai..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv /opt/poker_env
source /opt/poker_env/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install CUDA-specific JAX first (crucial for GPU training)
echo "🔥 Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX GPU installation
echo "✅ Verifying JAX GPU installation..."
python3 -c "import jax; print('JAX devices:', jax.devices()); print('GPU available:', len(jax.devices('gpu')) > 0)"

# Install project dependencies
echo "📚 Installing PokerBot dependencies..."
pip install -r requirements.txt

# Install the poker bot package
echo "🤖 Installing PokerBot package..."
pip install -e .

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models/
mkdir -p logs/
mkdir -p checkpoints/
mkdir -p data/

# Set up environment variables
echo "🌍 Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Create environment file
cat > /opt/poker_env/environment.sh << 'EOF'
#!/bin/bash
# PokerBot Environment Variables
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_PLATFORM_NAME=gpu

# Activate virtual environment
source /opt/poker_env/bin/activate
EOF

# Make environment file executable
chmod +x /opt/poker_env/environment.sh

# Create training script
cat > /opt/poker_env/train_poker.sh << 'EOF'
#!/bin/bash
# Quick training script for PokerBot

# Load environment
source /opt/poker_env/environment.sh

# Run training with default parameters
echo "🚀 Starting PokerBot training..."
poker-train --iterations 100000 --batch-size 1024 --players 2 --gpu

echo "✅ Training completed!"
EOF

# Make training script executable
chmod +x /opt/poker_env/train_poker.sh

# Create quick test script
cat > /opt/poker_env/test_setup.sh << 'EOF'
#!/bin/bash
# Test script to verify installation

# Load environment
source /opt/poker_env/environment.sh

echo "🔍 Testing PokerBot installation..."

# Test basic imports
python3 -c "
import jax
import jax.numpy as jnp
print('✅ JAX imported successfully')
print('GPU devices:', jax.devices('gpu'))

try:
    import poker_bot
    print('✅ PokerBot imported successfully')
    
    from poker_bot.evaluator import HandEvaluator
    evaluator = HandEvaluator()
    print('✅ Hand evaluator initialized')
    
    from poker_bot.engine import PokerEngine
    engine = PokerEngine()
    print('✅ Poker engine initialized')
    
    print('🎉 All components working!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

# Test CLI
echo "🎮 Testing CLI..."
poker-bot --help

echo "✅ Setup verification completed!"
EOF

# Make test script executable
chmod +x /opt/poker_env/test_setup.sh

# Run initial test
echo "🧪 Running setup verification..."
source /opt/poker_env/environment.sh
/opt/poker_env/test_setup.sh

# Create tmux session for training
echo "🖥️  Setting up tmux session..."
tmux new-session -d -s poker_training

# Display GPU info
echo "🔥 GPU Information:"
nvidia-smi

# Display completion message
echo "
🎉 PokerBot deployment completed successfully!

📁 Project structure:
   /opt/poker_env/          - Virtual environment
   /opt/poker_env/models/   - Trained models
   /opt/poker_env/logs/     - Training logs
   /opt/poker_env/checkpoints/ - Training checkpoints

🚀 Quick start commands:
   source /opt/poker_env/environment.sh  # Load environment
   /opt/poker_env/train_poker.sh         # Start training
   /opt/poker_env/test_setup.sh          # Test installation
   
🎮 CLI commands:
   poker-train --help        # Training options
   poker-play --help         # Play poker
   poker-bot --help          # Main help

🖥️  Training in background:
   tmux attach -t poker_training    # Attach to training session
   
📊 Monitor training:
   watch -n 1 nvidia-smi           # Monitor GPU usage
   htop                            # Monitor CPU/memory
   
✅ Ready for training on $(nvidia-smi -L | wc -l) GPU(s)!
"

echo "🔥 Deployment completed successfully!" 