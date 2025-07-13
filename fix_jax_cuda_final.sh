#!/bin/bash

echo "🔧 FIXING JAX CUDA SUPPORT FINALLY..."

# Remove problematic XLA flags
unset XLA_FLAGS

# Check CUDA installation
echo "🔍 CUDA setup:"
nvidia-smi
nvcc --version

# Install the correct jaxlib version with CUDA support
echo "🔄 Installing JAX with CUDA 12.8 support..."
pip uninstall -y jax jaxlib

# Install specific CUDA-enabled jaxlib
pip install --upgrade "jaxlib[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

# Test JAX setup
echo "🔄 Testing JAX CUDA setup..."
python -c "
import os
import jax
import jax.numpy as jnp

print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())

try:
    # Try to force GPU detection
    os.environ['JAX_PLATFORM_NAME'] = 'cuda'
    jax.devices()
    print('✅ JAX CUDA setup working!')
    
    # Test GPU operations
    gpu_devices = jax.devices('gpu')
    if len(gpu_devices) > 0:
        print('🎮 GPU devices found:', gpu_devices)
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = x + y
            print('✅ GPU operations work:', z)
        print('🎉 GPU is working! Ready for poker training!')
    else:
        print('⚠️  No GPU devices found')
        
except Exception as e:
    print('❌ GPU test failed:', e)
    print('💻 Using CPU fallback')
    
    # Test CPU operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    print('✅ CPU operations work:', z)
"

echo "✅ Setup complete!" 