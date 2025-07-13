#!/bin/bash

echo "🔧 Fixing vast.ai JAX/CUDA setup..."

# Remove problematic XLA flags
unset XLA_FLAGS

# Install cuSPARSE and other CUDA libraries
echo "📦 Installing CUDA libraries..."
apt-get update
apt-get install -y libcusparse-dev libcublas-dev libcurand-dev

# Reinstall JAX with CUDA support
echo "🔄 Reinstalling JAX with CUDA support..."
pip uninstall -y jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set environment variables for better performance
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "✅ Setup complete! Testing JAX..."
python -c "import jax; print('JAX devices:', jax.devices()); print('GPU devices:', jax.devices('gpu'))" 