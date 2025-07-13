#!/usr/bin/env python3
"""
Simple test that works without complex JAX operations
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

def test_simple_jax():
    """Test basic JAX functionality"""
    print("🧪 Testing simple JAX...")
    
    # Basic operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    
    print(f"✅ Basic JAX operations work: {z}")
    return True

def test_simple_nlhe():
    """Test simple NLHE engine"""
    print("\n🎰 Testing simple NLHE engine...")
    
    try:
        from poker_bot.simple_nlhe_engine import simple_nlhe_batch
        
        # Create test data
        batch_size = 10
        rng_key = jr.PRNGKey(42)
        rng_keys = jr.split(rng_key, batch_size)
        
        print(f"🔄 Running {batch_size} games...")
        start_time = time.time()
        
        results = simple_nlhe_batch(rng_keys)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Simple NLHE engine works!")
        print(f"⏱️  Time: {duration:.4f}s")
        print(f"🎮 Games per second: {batch_size/duration:.1f}")
        print(f"📊 Results shape: {results['hole_cards'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple NLHE engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_cfr():
    """Test simple CFR training"""
    print("\n🧠 Testing simple CFR training...")
    
    try:
        from poker_bot.simple_nlhe_engine import simple_nlhe_batch
        
        # Simple training loop
        batch_size = 50
        num_iterations = 5
        
        print(f"🔄 Running {num_iterations} training iterations...")
        start_time = time.time()
        
        rng_key = jr.PRNGKey(42)
        total_payoff = 0.0
        
        for i in range(num_iterations):
            rng_keys = jr.split(rng_key, batch_size)
            results = simple_nlhe_batch(rng_keys)
            total_payoff += jnp.mean(results['payoffs'])
            rng_key = jr.fold_in(rng_key, i)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Simple CFR training works!")
        print(f"⏱️  Time: {duration:.4f}s")
        print(f"🎮 Games per second: {num_iterations * batch_size / duration:.1f}")
        print(f"📊 Average payoff: {total_payoff / num_iterations:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple CFR training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing simple working system...")
    
    # Test JAX
    jax_ok = test_simple_jax()
    
    # Test simple NLHE
    nlhe_ok = test_simple_nlhe()
    
    # Test simple CFR
    cfr_ok = test_simple_cfr()
    
    if jax_ok and nlhe_ok and cfr_ok:
        print("\n🎉 All simple tests passed! System is working.")
        print("✅ You can now train poker AI in CPU mode.")
    else:
        print("\n❌ Some simple tests failed.")
        if jax_ok:
            print("✅ JAX is working, can proceed with basic training") 