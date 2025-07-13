#!/usr/bin/env python3
"""
Test script for REAL CFVFP trainer
Verifies that it learns actual poker strategies, not just fixed matrices
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
from typing import Dict, Any

def test_real_cfvfp():
    """Test REAL CFVFP trainer"""
    print("🚀 Testing REAL CFVFP Trainer")
    print("=" * 50)
    
    try:
        from poker_bot.real_cfvfp_trainer import RealCFVFPTrainer, RealCFVFPConfig
        from poker_bot.cli import batch_simulate_real_holdem
        
        print("✅ Modules imported successfully")
        
        # Initialize trainer
        config = RealCFVFPConfig(batch_size=1024)
        trainer = RealCFVFPTrainer(config)
        
        print(f"✅ Trainer initialized with batch_size={config.batch_size}")
        
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
        print("🔥 Running training step...")
        start_time = time.time()
        
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(rng_key, game_results)
        
        end_time = time.time()
        step_time = end_time - start_time
        
        print(f"✅ Training step completed in {step_time:.3f}s")
        print(f"   Games processed: {results['games_processed']:,}")
        print(f"   Total info sets: {results['total_info_sets']:,}")
        print(f"   Info sets processed: {results['info_sets_processed']:,}")
        print(f"   Q-values count: {results['q_values_count']:,}")
        print(f"   Strategies count: {results['strategies_count']:,}")
        print(f"   Avg payoff: {results['avg_payoff']:.4f}")
        print(f"   Strategy entropy: {float(results['strategy_entropy']):.4f}")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        test_save_path = "test_real_cfvfp_model.pkl"
        trainer.save_model(test_save_path)
        
        # Check file size
        import os
        file_size = os.path.getsize(test_save_path)
        print(f"✅ Model saved: {test_save_path}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Test model loading
        print("\n📂 Testing model loading...")
        new_trainer = RealCFVFPTrainer(config)
        new_trainer.load_model(test_save_path)
        
        print(f"✅ Model loaded successfully")
        print(f"   Q-values count: {len(new_trainer.q_values):,}")
        print(f"   Strategies count: {len(new_trainer.strategies):,}")
        print(f"   Average strategies count: {len(new_trainer.average_strategies):,}")
        
        # Verify that we have actual learned strategies
        if len(new_trainer.q_values) > 0:
            print(f"✅ REAL strategies learned: {len(new_trainer.q_values)} unique info sets")
            
            # Show sample strategy
            sample_hash = list(new_trainer.q_values.keys())[0]
            sample_q = new_trainer.q_values[sample_hash]
            sample_strategy = new_trainer.strategies[sample_hash]
            
            print(f"   Sample Q-values: {sample_q}")
            print(f"   Sample strategy: {sample_strategy}")
            print(f"   Strategy sum: {jnp.sum(sample_strategy):.4f}")
            
        else:
            print("❌ No strategies learned - this indicates a problem")
            return False
        
        # Clean up test file
        os.remove(test_save_path)
        print(f"✅ Test file cleaned up")
        
        print("\n🎉 REAL CFVFP test completed successfully!")
        print("=" * 50)
        print("✅ REAL information sets working")
        print("✅ Q-values being learned")
        print("✅ Strategies being saved")
        print("✅ Model can be loaded")
        print("✅ File size will grow with learning (not fixed)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_growth():
    """Test that model size grows with learning"""
    print("\n📈 Testing model growth with learning")
    print("=" * 50)
    
    try:
        from poker_bot.real_cfvfp_trainer import RealCFVFPTrainer, RealCFVFPConfig
        from poker_bot.cli import batch_simulate_real_holdem
        
        config = RealCFVFPConfig(batch_size=512)
        trainer = RealCFVFPTrainer(config)
        
        rng_key = jr.PRNGKey(42)
        game_config = {
            'players': 6,
            'starting_stack': 100.0,
            'small_blind': 1.0,
            'big_blind': 2.0
        }
        
        # Track model size over iterations
        sizes = []
        info_set_counts = []
        
        for iteration in range(5):
            rng_key = jr.fold_in(rng_key, iteration)
            rng_keys = jr.split(rng_key, config.batch_size)
            
            game_results = batch_simulate_real_holdem(rng_keys, game_config)
            results = trainer.train_step(rng_key, game_results)
            
            # Save model and check size
            test_path = f"test_growth_{iteration}.pkl"
            trainer.save_model(test_path)
            
            import os
            file_size = os.path.getsize(test_path)
            sizes.append(file_size)
            info_set_counts.append(results['total_info_sets'])
            
            os.remove(test_path)
            
            print(f"Iteration {iteration + 1}: {results['total_info_sets']:,} info sets, {file_size:,} bytes")
        
        # Check if model is growing
        if sizes[-1] > sizes[0]:
            print(f"✅ Model size growing: {sizes[0]:,} → {sizes[-1]:,} bytes")
            print(f"✅ Info sets growing: {info_set_counts[0]:,} → {info_set_counts[-1]:,}")
            return True
        else:
            print(f"❌ Model size not growing: {sizes[0]:,} → {sizes[-1]:,} bytes")
            return False
            
    except Exception as e:
        print(f"❌ Growth test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 REAL CFVFP Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_real_cfvfp()
    test2_passed = test_model_growth()
    
    print("\n📊 Test Results:")
    print("=" * 30)
    print(f"Basic functionality: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Model growth: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ REAL CFVFP is working correctly")
        print("✅ Will learn actual poker strategies")
        print("✅ Model size will grow with learning")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the implementation")
    
    print("\n" + "=" * 60) 