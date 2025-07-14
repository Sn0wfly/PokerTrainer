#!/bin/bash

echo "🔧 INSTALLING CORRECT JAX WITH CUDA SUPPORT"
echo "============================================="

# Check current GPU status
echo "🔍 Current GPU status:"
nvidia-smi

# Remove current JAX installation
echo "🧹 Removing current JAX installation..."
pip uninstall -y jax jaxlib

# Install the correct CUDA-enabled version
echo "📦 Installing JAX with CUDA support..."
pip install --upgrade --no-deps jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade jax==0.4.29

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

# Test the installation
echo "🧪 Testing JAX GPU setup..."
python -c "
import jax
import jax.numpy as jnp

print('JAX version:', jax.__version__)
print('Available devices:', jax.devices())

try:
    gpu_devices = jax.devices('gpu')
    if len(gpu_devices) > 0:
        print('✅ GPU devices found:', gpu_devices)
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = x + y
            print('✅ GPU operations work:', z)
        print('🎉 GPU is working!')
    else:
        print('❌ No GPU devices found')
        
except Exception as e:
    print('❌ GPU test failed:', e)
"

echo "✅ Fix complete! Try running your poker training now." 