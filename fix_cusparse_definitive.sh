#!/bin/bash

echo "🔧 FIXING cuSPARSE DEFINITIVELY..."

# Remove problematic XLA flags
unset XLA_FLAGS

# Check current CUDA installation
echo "🔍 Current CUDA setup:"
nvidia-smi
nvcc --version

# Install CUDA libraries from the correct repository
echo "📦 Installing CUDA libraries from NVIDIA repo..."
apt-get update

# Try different package names for different Ubuntu versions
apt-get install -y libcusparse-dev || echo "libcusparse-dev not found"
apt-get install -y libcusparse-12-8 || echo "libcusparse-12-8 not found"
apt-get install -y libcusparse-12-4 || echo "libcusparse-12-4 not found"

# Try installing from CUDA toolkit
apt-get install -y cuda-toolkit-12-8 || echo "cuda-toolkit-12-8 not found"
apt-get install -y cuda-toolkit-12-4 || echo "cuda-toolkit-12-4 not found"

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Find existing CUDA libraries
echo "🔍 Looking for existing CUDA libraries..."
find /usr -name "libcusparse*" 2>/dev/null
find /usr/local -name "libcusparse*" 2>/dev/null

# Create symbolic links if needed
if [ -f "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so" ]; then
    echo "🔗 Creating symbolic links..."
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so /usr/lib/libcusparse.so
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so /usr/lib/libcublas.so
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so /usr/lib/libcurand.so
fi

# Reinstall JAX with specific CUDA version
echo "🔄 Reinstalling JAX with CUDA support..."
pip uninstall -y jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set environment variables for better performance
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Test JAX setup
echo "🔄 Testing JAX setup..."
python -c "
import os
import jax
import jax.numpy as jnp

print('JAX devices:', jax.devices())
try:
    gpu_devices = jax.devices('gpu')
    print('GPU devices:', gpu_devices)
    if len(gpu_devices) > 0:
        print('✅ GPU is available!')
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = x + y
            print('✅ GPU operations work:', z)
    else:
        print('⚠️  No GPU devices found')
except Exception as e:
    print('❌ GPU test failed:', e)
    print('💻 Falling back to CPU')
"

echo "✅ Setup complete!" 