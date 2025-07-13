#!/usr/bin/env python3
"""
Test script to verify JAX and NLHE engine work on vast.ai (v2 with CPU fallback)
"""

import os
import jax
import jax.numpy as jnp
import jax.random as jr
import time

def test_jax_setup():
    """Test basic JAX functionality with CPU fallback"""
    print("🧪 Testing JAX setup...")
    
    # Force CPU if GPU setup fails
    try:
        devices = jax.devices()
        print(f"📱 Available devices: {devices}")
        
        # Try GPU first
        try:
            gpu_devices = jax.devices('gpu')
            print(f"🎮 GPU devices: {gpu_devices}")
            has_gpu = len(gpu_devices) > 0
        except:
            print("⚠️  GPU not available, using CPU")
            has_gpu = False
        
        # Test basic operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        
        print(f"✅ Basic JAX operations work: {z}")
        
        # Test GPU operations if available
        if has_gpu:
            print("🎮 Testing GPU operations...")
            with jax.default_device(gpu_devices[0]):
                gpu_x = jnp.array([1.0, 2.0, 3.0])
                gpu_y = jnp.array([4.0, 5.0, 6.0])
                gpu_z = gpu_x + gpu_y
                print(f"✅ GPU operations work: {gpu_z}")
        else:
            print("💻 Using CPU for operations")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX setup failed: {e}")
        return False

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
        import traceback
        traceback.print_exc()
        return False

def test_cfr_trainer():
    """Test CFR trainer"""
    print("\n🧠 Testing CFR trainer...")
    
    try:
        from poker_bot.nlhe_cfr_trainer import NLHE6PlayerCFRTrainer
        
        # Create trainer with small batch size
        trainer = NLHE6PlayerCFRTrainer(batch_size=50)
        
        print("🔄 Running 5 training iterations...")
        start_time = time.time()
        
        results = trainer.train(num_iterations=5, save_interval=2)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ CFR trainer works!")
        print(f"⏱️  Time: {duration:.4f}s")
        print(f"📊 Training results: {len(results)} iterations completed")
        
        return True
        
    except Exception as e:
        print(f"❌ CFR trainer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing vast.ai JAX fixes (v2)...")
    
    # Test JAX setup
    jax_ok = test_jax_setup()
    
    # Test NLHE engine
    nlhe_ok = test_nlhe_engine()
    
    # Test CFR trainer
    cfr_ok = test_cfr_trainer()
    
    if jax_ok and nlhe_ok and cfr_ok:
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n❌ Some tests failed. Check the setup.")
        if jax_ok:
            print("✅ JAX is working, can proceed with CPU training") 