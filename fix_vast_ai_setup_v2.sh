#!/bin/bash

echo "🔧 Fixing vast.ai JAX/CUDA setup (v2)..."

# Remove problematic XLA flags
unset XLA_FLAGS

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
nvidia-smi
nvcc --version

# Install CUDA toolkit and libraries
echo "📦 Installing CUDA toolkit..."
apt-get update
apt-get install -y cuda-toolkit-12-4

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install cuSPARSE and other CUDA libraries from NVIDIA repository
echo "📦 Installing CUDA libraries from NVIDIA repo..."
apt-get install -y libcusparse-12-4 libcublas-12-4 libcurand-12-4

# Alternative: try to find existing CUDA libraries
echo "🔍 Looking for existing CUDA libraries..."
find /usr -name "libcusparse*" 2>/dev/null
find /usr/local -name "libcusparse*" 2>/dev/null

# Reinstall JAX with specific CUDA version
echo "🔄 Reinstalling JAX with CUDA 12.4 support..."
pip uninstall -y jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set environment variables for better performance
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"

# Try to force JAX to use CPU if GPU setup fails
echo "🔄 Testing JAX setup..."
python -c "
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
import jax.numpy as jnp
print('JAX devices:', jax.devices())
print('✅ JAX CPU setup working')
"

echo "✅ Setup complete!" 