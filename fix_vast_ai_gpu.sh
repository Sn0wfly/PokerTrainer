#!/bin/bash

echo "🔧 VAST.AI GPU QUICK FIX"
echo "=========================="

# Check current GPU status
echo "🔍 Current GPU status:"
nvidia-smi

# Check current JAX installation
echo -e "\n🔍 Current JAX installation:"
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Unset any problematic environment variables
echo -e "\n🧹 Cleaning environment variables..."
unset XLA_FLAGS
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

# Set correct environment variables
echo -e "\n⚙️  Setting correct environment variables..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

# Reinstall JAX with CUDA support
echo -e "\n📦 Reinstalling JAX with CUDA support..."
pip uninstall -y jax jaxlib

# Install the working CUDA version
pip install --upgrade --no-deps jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade jax==0.4.29

# Test the installation
echo -e "\n🧪 Testing JAX GPU setup..."
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

echo -e "\n✅ Fix complete! Try running your poker training now."
echo "If it still doesn't work, run: python vast_ai_gpu_diagnostic.py" 