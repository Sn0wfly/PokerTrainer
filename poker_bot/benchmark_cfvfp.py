"""
🚀 CFVFP vs Traditional CFR Benchmark
Compare performance and measure speedup
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import logging
from typing import Dict, Any

from .cfvfp_optimized import OptimizedCFVFPTrainer, CFVFPConfig, benchmark_cfvfp_performance
from .nlhe_real_engine import nlhe_6player_batch

logger = logging.getLogger(__name__)

def benchmark_traditional_cfr():
    """Benchmark traditional CFR performance"""
    logger.info("🔄 Traditional CFR Benchmark")
    logger.info("=" * 50)
    
    # Test configuration
    batch_size = 8192
    num_actions = 4
    num_iterations = 100
    
    # Traditional CFR simulation (regret-based)
    def traditional_cfr_step(regrets: jnp.ndarray, payoffs: jnp.ndarray) -> jnp.ndarray:
        """Traditional CFR with regret matching"""
        # Update regrets
        new_regrets = regrets + payoffs
        
        # Regret matching
        positive_regrets = jnp.maximum(new_regrets, 0)
        regret_sums = jnp.sum(positive_regrets, axis=-1, keepdims=True)
        
        # Avoid division by zero
        regret_sums = jnp.where(regret_sums > 0, regret_sums, 1.0)
        
        # Strategy computation
        strategies = positive_regrets / regret_sums
        
        return strategies
    
    # Compile function
    traditional_cfr_step = jax.jit(traditional_cfr_step)
    
    # Generate test data
    rng_key = jax.random.PRNGKey(42)
    test_regrets = jnp.zeros((batch_size, num_actions))
    test_payoffs = jax.random.normal(rng_key, (batch_size, num_actions))
    
    # Warm-up compilation
    logger.info("🔥 Warming up traditional CFR compilation...")
    _ = traditional_cfr_step(test_regrets, test_payoffs)
    
    # Benchmark
    logger.info("🔄 Benchmarking traditional CFR performance...")
    start_time = time.time()
    
    for i in range(num_iterations):
        _ = traditional_cfr_step(test_regrets, test_payoffs)
    
    end_time = time.time()
    total_time = end_time - start_time
    iterations_per_sec = num_iterations / total_time
    games_per_sec = iterations_per_sec * batch_size
    
    logger.info(f"✅ Traditional CFR Performance Results:")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Total time: {total_time:.3f}s")
    logger.info(f"   Iterations/sec: {iterations_per_sec:.1f}")
    logger.info(f"   Games/sec: {games_per_sec:,.0f}")
    
    return games_per_sec

def benchmark_full_pipeline():
    """Benchmark full CFVFP pipeline with real poker simulation"""
    logger.info("🚀 Full CFVFP Pipeline Benchmark")
    logger.info("=" * 50)
    
    # Test configuration
    batch_size = 8192
    num_iterations = 50
    
    # Initialize CFVFP trainer
    config = CFVFPConfig(batch_size=batch_size)
    trainer = OptimizedCFVFPTrainer(config)
    
    # Generate RNG keys
    rng_key = jr.PRNGKey(42)
    
    # Game configuration
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Warm-up compilation
    logger.info("🔥 Warming up full pipeline compilation...")
    test_rng_keys = jr.split(rng_key, batch_size)
    test_results = nlhe_6player_batch(test_rng_keys)
    _ = trainer.train_step(rng_key, test_results)
    
    # Benchmark
    logger.info("🚀 Benchmarking full CFVFP pipeline...")
    start_time = time.time()
    
    for i in range(num_iterations):
        rng_key = jr.fold_in(rng_key, i)
        rng_keys = jr.split(rng_key, batch_size)
        
        # Real poker simulation + CFVFP update
        game_results = nlhe_6player_batch(rng_keys)
        _ = trainer.train_step(rng_key, game_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    iterations_per_sec = num_iterations / total_time
    games_per_sec = iterations_per_sec * batch_size
    
    logger.info(f"✅ Full CFVFP Pipeline Results:")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Total time: {total_time:.3f}s")
    logger.info(f"   Iterations/sec: {iterations_per_sec:.1f}")
    logger.info(f"   Games/sec: {games_per_sec:,.0f}")
    logger.info(f"   Target achieved: {'✅' if games_per_sec > 1000 else '❌'}")
    
    return games_per_sec

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing all approaches"""
    logger.info("🎯 Comprehensive CFVFP vs CFR Benchmark")
    logger.info("=" * 60)
    
    # Benchmark CFVFP core algorithm
    cfvfp_games_per_sec = benchmark_cfvfp_performance()
    
    # Benchmark traditional CFR
    traditional_games_per_sec = benchmark_traditional_cfr()
    
    # Benchmark full pipeline
    pipeline_games_per_sec = benchmark_full_pipeline()
    
    # Calculate speedups
    cfvfp_speedup = cfvfp_games_per_sec / traditional_games_per_sec
    pipeline_speedup = pipeline_games_per_sec / traditional_games_per_sec
    
    # Results summary
    logger.info("\n🎯 COMPREHENSIVE BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"📊 Performance Comparison:")
    logger.info(f"   Traditional CFR: {traditional_games_per_sec:,.0f} games/sec")
    logger.info(f"   CFVFP Core: {cfvfp_games_per_sec:,.0f} games/sec")
    logger.info(f"   CFVFP Full Pipeline: {pipeline_games_per_sec:,.0f} games/sec")
    logger.info("")
    logger.info(f"🚀 Speedup Analysis:")
    logger.info(f"   CFVFP Core vs Traditional: {cfvfp_speedup:.1f}x")
    logger.info(f"   Full Pipeline vs Traditional: {pipeline_speedup:.1f}x")
    logger.info("")
    logger.info(f"🎯 Target Achievement:")
    logger.info(f"   CFVFP Core: {'✅' if cfvfp_games_per_sec > 1000 else '❌'} (Target: 1000+ games/sec)")
    logger.info(f"   Full Pipeline: {'✅' if pipeline_games_per_sec > 1000 else '❌'} (Target: 1000+ games/sec)")
    
    return {
        'traditional_cfr': traditional_games_per_sec,
        'cfvfp_core': cfvfp_games_per_sec,
        'cfvfp_pipeline': pipeline_games_per_sec,
        'cfvfp_speedup': cfvfp_speedup,
        'pipeline_speedup': pipeline_speedup
    }

if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    print(f"\n🎯 Final Results:")
    print(f"   CFVFP Core: {results['cfvfp_core']:,.0f} games/sec")
    print(f"   Full Pipeline: {results['cfvfp_pipeline']:,.0f} games/sec")
    print(f"   Speedup: {results['cfvfp_speedup']:.1f}x") 