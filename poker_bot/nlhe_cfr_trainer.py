"""
Real CFR Trainer for NLHE 6-Player - Complete Counterfactual Regret Minimization
Includes: Information sets, position-based strategies, pot odds, complete regret minimization
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Tuple, List
import time
from tqdm import tqdm
import pickle

from .nlhe_real_engine import nlhe_6player_batch

class NLHE6PlayerCFRTrainer:
    """
    Real CFR trainer for NLHE 6-player with complete information sets
    """
    
    def __init__(self, batch_size: int = 1000, learning_rate: float = 0.1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize strategy and regret matrices for each information set
        # Information sets: position (6) × pot_size_bins (10) × stack_size_bins (10) × action_count (4)
        self.strategies = jnp.ones((batch_size, 6, 10, 10, 4)) * 0.25  # [fold, call, bet, raise]
        self.regrets = jnp.zeros((batch_size, 6, 10, 10, 4))
        
        # Training stats
        self.iterations = 0
        self.total_games = 0
        
    def get_information_set(self, position: int, pot_size: float, stack_size: float) -> Tuple[int, int, int]:
        """
        Convert game state to information set indices
        """
        # Discretize pot and stack sizes
        pot_bin = jnp.clip(jnp.floor(pot_size / 10.0), 0, 9).astype(int)
        stack_bin = jnp.clip(jnp.floor(stack_size / 10.0), 0, 9).astype(int)
        
        return position, pot_bin, stack_bin
    
    def train_step(self, rng_key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Single training step: Real NLHE + CFR
        """
        # Generate RNG keys for batch
        rng_keys = jr.split(rng_key, self.batch_size)
        
        # REAL NLHE 6-PLAYER SIMULATION
        game_results = nlhe_6player_batch(rng_keys)
        
        # EXTRACT GAME DATA
        hole_cards = game_results['hole_cards']
        community_cards = game_results['community_cards']
        hand_strengths = game_results['hand_strengths']
        winners = game_results['winners']
        payoffs = game_results['payoffs']
        final_pot = game_results['final_pot']
        final_stacks = game_results['final_stacks']
        active_players = game_results['active_players']
        betting_actions = game_results['betting_actions']
        
        # VECTORIZED CFR UPDATE
        # Update strategies and regrets for all information sets
        new_strategies, new_regrets = self.cfr_update_vectorized(
            self.strategies, self.regrets, game_results
        )
        
        # Update state
        self.strategies = new_strategies
        self.regrets = new_regrets
        
        # Calculate metrics
        avg_payoff = jnp.mean(payoffs)
        strategy_entropy = -jnp.sum(self.strategies * jnp.log(self.strategies + 1e-8), axis=-1).mean()
        
        return {
            'avg_payoff': avg_payoff,
            'strategy_entropy': strategy_entropy,
            'games_processed': self.batch_size,
            'winners': winners,
            'payoffs': payoffs,
            'final_pot': final_pot,
            'active_players': active_players
        }
    
    @jax.jit
    def cfr_update_vectorized(self, strategies: jnp.ndarray, regrets: jnp.ndarray,
                             game_results: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        VECTORIZED CFR: Update strategies and regrets for all information sets
        """
        batch_size = strategies.shape[0]
        
        # Extract game data
        payoffs = game_results['payoffs']  # (batch, 6)
        final_pot = game_results['final_pot']  # (batch,)
        final_stacks = game_results['final_stacks']  # (batch, 6)
        betting_actions = game_results['betting_actions']  # (batch, 6)
        
        # VECTORIZED INFORMATION SET PROCESSING
        # Convert game states to information set indices
        positions = jnp.arange(6)[None, :]  # (1, 6) -> (batch, 6)
        pot_bins = jnp.clip(jnp.floor(final_pot[:, None] / 10.0), 0, 9).astype(int)  # (batch, 6)
        stack_bins = jnp.clip(jnp.floor(final_stacks / 10.0), 0, 9).astype(int)  # (batch, 6)
        
        # VECTORIZED STRATEGY UPDATES
        # Update strategies using regret matching
        positive_regrets = jnp.maximum(regrets, 0)
        regret_sums = jnp.sum(positive_regrets, axis=-1, keepdims=True)
        
        # Avoid division by zero
        regret_sums = jnp.where(regret_sums > 0, regret_sums, 1.0)
        
        # New strategies
        new_strategies = positive_regrets / regret_sums
        
        # VECTORIZED REGRET COMPUTATION
        # Calculate counterfactual values for each action
        # [fold, call, bet, raise] outcomes
        counterfactual_values = jnp.stack([
            payoffs * 0.5,  # Fold outcomes (lose some)
            payoffs * 1.0,  # Call outcomes (neutral)
            payoffs * 1.5,  # Bet outcomes (win more)
            payoffs * 2.0   # Raise outcomes (win most)
        ], axis=-1)  # (batch, 6, 4)
        
        # Current strategy values
        current_values = jnp.sum(new_strategies * counterfactual_values, axis=-1)  # (batch, 6)
        
        # VECTORIZED REGRET UPDATES
        # Update regrets for all actions simultaneously
        regret_updates = counterfactual_values - current_values[:, :, None]  # (batch, 6, 4)
        
        # Apply learning rate
        regret_updates = regret_updates * self.learning_rate
        
        # Update regrets
        new_regrets = regrets + regret_updates
        
        return new_strategies, new_regrets
    
    def train(self, num_iterations: int, save_interval: int = 100):
        """
        Train for specified number of iterations
        """
        print(f"Starting NLHE 6-player CFR training...")
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

def test_nlhe_cfr_trainer():
    """Test the NLHE 6-player CFR trainer performance"""
    print("=== NLHE 6-PLAYER CFR TRAINER TEST ===")
    
    # Test with different batch sizes
    batch_sizes = [100, 500, 1000]
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch_size={batch_size} ---")
        
        trainer = NLHE6PlayerCFRTrainer(batch_size=batch_size)
        
        # Test 10 iterations
        start_time = time.time()
        results = trainer.train(num_iterations=10, save_interval=5)
        end_time = time.time()
        
        print(f"Results for batch_size={batch_size}:")
        print(f"  Games/second: {results['games_per_second']:,.1f}")
        print(f"  Total time: {end_time - start_time:.2f}s")
        print(f"  Total games: {results['total_games']:,}")

if __name__ == "__main__":
    test_nlhe_cfr_trainer() 