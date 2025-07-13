#!/usr/bin/env python3
"""
Test script to verify JAX and NLHE engine work on vast.ai
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

def test_jax_setup():
    """Test basic JAX functionality"""
    print("🧪 Testing JAX setup...")
    
    # Check devices
    devices = jax.devices()
    gpu_devices = jax.devices('gpu')
    
    print(f"📱 Available devices: {devices}")
    print(f"🎮 GPU devices: {gpu_devices}")
    
    # Test basic operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    
    print(f"✅ Basic JAX operations work: {z}")
    
    # Test GPU operations if available
    if gpu_devices:
        print("🎮 Testing GPU operations...")
        with jax.default_device(gpu_devices[0]):
            gpu_x = jnp.array([1.0, 2.0, 3.0])
            gpu_y = jnp.array([4.0, 5.0, 6.0])
            gpu_z = gpu_x + gpu_y
            print(f"✅ GPU operations work: {gpu_z}")
    
    return True

def test_nlhe_engine():
    """Test NLHE engine without strings"""
    print("\n🎰 Testing NLHE engine...")
    
    try:
        from poker_bot.nlhe_real_engine import nlhe_6player_batch
        
        # Create test data
        batch_size = 10
        rng_key = jr.PRNGKey(42)
        rng_keys = jr.split(rng_key, batch_size)
        
        print(f"🔄 Running {batch_size} games...")
        start_time = time.time()
        
        results = nlhe_6player_batch(rng_keys)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ NLHE engine works!")
        print(f"⏱️  Time: {duration:.4f}s")
        print(f"🎮 Games per second: {batch_size/duration:.1f}")
        print(f"📊 Results shape: {results['hole_cards'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ NLHE engine failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing vast.ai JAX fixes...")
    
    # Test JAX setup
    jax_ok = test_jax_setup()
    
    # Test NLHE engine
    nlhe_ok = test_nlhe_engine()
    
    if jax_ok and nlhe_ok:
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n❌ Some tests failed. Check the setup.") 