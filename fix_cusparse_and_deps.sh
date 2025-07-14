#!/bin/bash

echo "🔧 FIXING CUSPARSE AND DEPENDENCIES"
echo "===================================="

# Install cuSPARSE
echo "📦 Installing cuSPARSE..."
apt-get update
apt-get install -y libcusparse-dev

# Check if cuSPARSE is now available
echo "🔍 Checking cuSPARSE installation..."
ldconfig -p | grep cusparse

# Fix JAX version conflicts
echo "📦 Fixing JAX version conflicts..."
pip install --upgrade jax==0.6.2 jaxlib==0.6.2

# Install compatible versions of Flax and Orbax
echo "📦 Installing compatible Flax and Orbax..."
pip install --upgrade flax orbax-checkpoint

# Test the fix
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
        print('🎉 GPU is working without cuSPARSE errors!')
    else:
        print('❌ No GPU devices found')
        
except Exception as e:
    print('❌ GPU test failed:', e)
"

echo "✅ Fix complete!" 