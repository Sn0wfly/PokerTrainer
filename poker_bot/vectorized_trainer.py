"""
Vectorized CFR Trainer - Uses the ultra-fast vectorized poker engine
Target: 10,000+ games/second with real CFR training
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Tuple
import time
from tqdm import tqdm
import pickle

from .engine_vectorized import vectorized_poker_batch, vectorized_cfr_step

class VectorizedCFRTrainer:
    """
    Ultra-fast CFR trainer using vectorized operations
    """
    
    def __init__(self, batch_size: int = 10000, learning_rate: float = 0.1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize strategy and regret matrices
        self.strategies = jnp.ones((batch_size, 3)) * 0.33  # [fold, call, raise]
        self.regrets = jnp.zeros((batch_size, 3))
        
        # Training stats
        self.iterations = 0
        self.total_games = 0
        
    @jax.jit
    def train_step(self, rng_key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Single training step: Vectorized poker + CFR
        """
        # Generate RNG keys for batch
        rng_keys = jr.split(rng_key, self.batch_size)
        
        # VECTORIZED POKER SIMULATION
        game_results = vectorized_poker_batch(rng_keys)
        
        # VECTORIZED CFR UPDATE
        new_strategies, new_regrets = vectorized_cfr_step(
            self.strategies, self.regrets, game_results
        )
        
        # Update state
        self.strategies = new_strategies
        self.regrets = new_regrets
        
        # Calculate metrics
        avg_payoff = jnp.mean(game_results['payoffs'])
        strategy_entropy = -jnp.sum(self.strategies * jnp.log(self.strategies + 1e-8), axis=1).mean()
        
        return {
            'avg_payoff': avg_payoff,
            'strategy_entropy': strategy_entropy,
            'games_processed': self.batch_size,
            'winners': game_results['winners'],
            'payoffs': game_results['payoffs']
        }
    
    def train(self, num_iterations: int, save_interval: int = 100):
        """
        Train for specified number of iterations
        """
        print(f"Starting vectorized CFR training...")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Expected games: {num_iterations * self.batch_size:,}")
        
        rng_key = jr.PRNGKey(42)
        
        # Warm-up compilation
        print("Compiling JAX functions...")
        _ = self.train_step(rng_key)
        
        # Training loop
        print("Starting training...")
        start_time = time.time()
        
        for iteration in tqdm(range(num_iterations), desc="Training"):
            # Generate new RNG key for this iteration
            rng_key = jr.fold_in(rng_key, iteration)
            
            # Training step
            results = self.train_step(rng_key)
            
            # Log progress
            if (iteration + 1) % save_interval == 0:
                elapsed = time.time() - start_time
                games_per_second = (iteration + 1) * self.batch_size / elapsed
                
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  Games/second: {games_per_second:,.1f}")
                print(f"  Avg payoff: {results['avg_payoff']:.4f}")
                print(f"  Strategy entropy: {results['strategy_entropy']:.4f}")
                print(f"  Elapsed time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        total_games = num_iterations * self.batch_size
        final_games_per_second = total_games / total_time
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Total games: {total_games:,}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average games/second: {final_games_per_second:,.1f}")
        print(f"Games per iteration: {self.batch_size:,}")
        
        return {
            'total_games': total_games,
            'total_time': total_time,
            'games_per_second': final_games_per_second,
            'final_strategies': self.strategies,
            'final_regrets': self.regrets
        }
    
    def save_model(self, path: str):
        """Save trained model"""
        model_data = {
            'strategies': self.strategies,
            'regrets': self.regrets,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

def test_vectorized_trainer():
    """Test the vectorized trainer performance"""
    print("=== VECTORIZED CFR TRAINER TEST ===")
    
    # Test with different batch sizes
    batch_sizes = [1000, 5000, 10000]
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch_size={batch_size} ---")
        
        trainer = VectorizedCFRTrainer(batch_size=batch_size)
        
        # Test 10 iterations
        start_time = time.time()
        results = trainer.train(num_iterations=10, save_interval=5)
        end_time = time.time()
        
        print(f"Results for batch_size={batch_size}:")
        print(f"  Games/second: {results['games_per_second']:,.1f}")
        print(f"  Total time: {end_time - start_time:.2f}s")
        print(f"  Total games: {results['total_games']:,}")

if __name__ == "__main__":
    test_vectorized_trainer() 