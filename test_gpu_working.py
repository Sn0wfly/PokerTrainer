#!/usr/bin/env python3
"""
🧪 SIMPLE GPU TEST
Verifica que la GPU está funcionando correctamente
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

def test_gpu_basic():
    """Test basic GPU operations"""
    print("🧪 Testing basic GPU operations...")
    
    try:
        # Check devices
        devices = jax.devices()
        gpu_devices = jax.devices('gpu')
        
        print(f"📱 All devices: {devices}")
        print(f"🎮 GPU devices: {gpu_devices}")
        
        if len(gpu_devices) == 0:
            print("❌ No GPU devices found")
            return False
        
        # Test GPU operations
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = x + y
            
            print(f"✅ GPU operations work: {z}")
            print(f"✅ Device used: {x.devices()}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_poker_gpu():
    """Test poker bot GPU usage"""
    print("\n🎰 Testing poker bot GPU usage...")
    
    try:
        from poker_bot.simple_nlhe_engine import simple_nlhe_batch
        
        # Create test data
        batch_size = 10
        rng_key = jr.PRNGKey(42)
        rng_keys = jr.split(rng_key, batch_size)
        
        print(f"🔄 Running {batch_size} games on GPU...")
        start_time = time.time()
        
        results = simple_nlhe_batch(rng_keys)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Poker bot GPU operations work!")
        print(f"⏱️  Time: {duration:.4f}s")
        print(f"🎮 Games per second: {batch_size/duration:.1f}")
        print(f"📊 Results shape: {results['hole_cards'].shape}")
        print(f"📊 Results devices: {results['hole_cards'].devices()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Poker bot GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_gpu():
    """Test training GPU usage"""
    print("\n🧠 Testing training GPU usage...")
    
    try:
        from poker_bot.real_cfvfp_trainer import RealCFVFPTrainer, RealCFVFPConfig
        from poker_bot.cli import batch_simulate_real_holdem
        
        # Create trainer
        config = RealCFVFPConfig(batch_size=256)
        trainer = RealCFVFPTrainer(config)
        
        print(f"✅ Trainer created with batch_size={config.batch_size}")
        
        # Generate test data
        rng_key = jr.PRNGKey(42)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        game_config = {
            'players': 6,
            'starting_stack': 100.0,
            'small_blind': 1.0,
            'big_blind': 2.0
        }
        
        # Run training step
        print("🔥 Running training step on GPU...")
        start_time = time.time()
        
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(rng_key, game_results)
        
        end_time = time.time()
        step_time = end_time - start_time
        
        print(f"✅ Training step completed in {step_time:.3f}s")
        print(f"   Games processed: {results['games_processed']:,}")
        print(f"   Total info sets: {results['total_info_sets']:,}")
        print(f"   Q-values count: {results['q_values_count']:,}")
        print(f"   Strategies count: {results['strategies_count']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 GPU WORKING TEST")
    print("=" * 50)
    
    # Test basic GPU
    gpu_ok = test_gpu_basic()
    
    # Test poker bot GPU
    poker_ok = test_poker_gpu()
    
    # Test training GPU
    training_ok = test_training_gpu()
    
    print("\n📊 TEST RESULTS:")
    print("=" * 30)
    print(f"Basic GPU: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"Poker bot GPU: {'✅ PASS' if poker_ok else '❌ FAIL'}")
    print(f"Training GPU: {'✅ PASS' if training_ok else '❌ FAIL'}")
    
    if gpu_ok and poker_ok and training_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ GPU is working correctly")
        print("✅ Poker bot can use GPU")
        print("✅ Training can use GPU")
        print("✅ Your modifications didn't break GPU usage")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Check the specific errors above")
    
    print("\n" + "=" * 50) 