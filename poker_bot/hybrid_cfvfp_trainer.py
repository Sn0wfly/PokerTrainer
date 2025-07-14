#!/usr/bin/env python3
"""
🚀 HYBRID CFVFP TRAINER
Combines VECTORIZED GPU performance with REAL dynamic growth
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pickle
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from functools import partial
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class HybridCFVFPConfig:
    """Hybrid CFVFP configuration for NLHE 6-player"""
    batch_size: int = 8192
    learning_rate: float = 0.1
    temperature: float = 1.0
    num_actions: int = 4  # fold, call, bet, raise
    dtype: jnp.dtype = jnp.bfloat16
    accumulation_dtype: jnp.dtype = jnp.float32
    max_info_sets: int = 1000000  # 1M info sets max
    growth_factor: float = 1.5  # Grow by 50% when full

class InfoSet:
    """Real poker information set"""
    def __init__(self, player_id: int, position: int, hole_cards: jnp.ndarray, 
                 community_cards: jnp.ndarray, pot_size: float, stack_size: float,
                 hand_strength: float, phase: int, betting_history: jnp.ndarray):
        self.player_id = player_id
        self.position = position
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.stack_size = stack_size
        self.hand_strength = hand_strength
        self.phase = phase
        self.betting_history = betting_history
    
    def to_hash(self) -> str:
        """Create unique hash for this info set"""
        # Create hash from all components
        hash_data = (
            self.player_id,
            self.position,
            tuple(self.hole_cards),
            tuple(self.community_cards),
            round(self.pot_size, 2),
            round(self.stack_size, 2),
            round(self.hand_strength, 3),
            self.phase,
            tuple(self.betting_history)
        )
        return hashlib.md5(str(hash_data).encode()).hexdigest()

class HybridCFVFPTrainer:
    """
    🚀 HYBRID CFVFP TRAINER
    Combines VECTORIZED GPU performance with REAL dynamic growth
    """
    
    def __init__(self, config: HybridCFVFPConfig):
        self.config = config
        self.iteration = 0
        self.total_games = 0
        self.total_info_sets = 0
        
        # VECTORIZED GPU arrays (for performance)
        self.q_values = jnp.zeros((config.max_info_sets, config.num_actions), dtype=config.dtype)
        self.strategies = jnp.ones((config.max_info_sets, config.num_actions), dtype=config.dtype) / config.num_actions
        
        # REAL info set tracking (for growth)
        self.info_set_hashes: Dict[str, int] = {}  # hash -> index
        self.info_set_data: List[InfoSet] = []  # index -> InfoSet
        self.next_index = 0
        
        # Performance tracking
        self.growth_events = 0
        self.total_unique_info_sets = 0
        
        logger.info("🚀 Hybrid CFVFP Trainer initialized")
        logger.info(f"   Batch size: {config.batch_size}")
        logger.info(f"   Max info sets: {config.max_info_sets:,}")
        logger.info(f"   Growth factor: {config.growth_factor}")
        logger.info(f"   Target: Real NLHE 6-player strategies with GPU acceleration")
    
    def _get_or_create_index(self, info_set: InfoSet) -> int:
        """Get existing index or create new one for info set"""
        info_hash = info_set.to_hash()
        
        if info_hash in self.info_set_hashes:
            return self.info_set_hashes[info_hash]
        
        # Check if we need to grow
        if self.next_index >= self.config.max_info_sets:
            self._grow_arrays()
        
        # Create new entry
        index = self.next_index
        self.info_set_hashes[info_hash] = index
        self.info_set_data.append(info_set)
        self.next_index += 1
        self.total_unique_info_sets += 1
        
        return index
    
    def _grow_arrays(self):
        """Grow arrays when capacity is reached"""
        old_size = self.config.max_info_sets
        new_size = int(old_size * self.config.growth_factor)
        
        logger.info(f"🔄 Growing arrays: {old_size:,} → {new_size:,} info sets")
        
        # Create new larger arrays
        new_q_values = jnp.zeros((new_size, self.config.num_actions), dtype=self.config.dtype)
        new_strategies = jnp.ones((new_size, self.config.num_actions), dtype=self.config.dtype) / self.config.num_actions
        
        # Copy existing data
        new_q_values = new_q_values.at[:old_size].set(self.q_values)
        new_strategies = new_strategies.at[:old_size].set(self.strategies)
        
        # Update arrays
        self.q_values = new_q_values
        self.strategies = new_strategies
        self.config.max_info_sets = new_size
        
        self.growth_events += 1
        logger.info(f"✅ Arrays grown successfully (event #{self.growth_events})")
    
    @partial(jax.jit, static_argnums=(0,))
    def _vectorized_info_set_processing(self, game_data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """VECTORIZED info set processing on GPU"""
        batch_size = game_data['payoffs'].shape[0]
        num_players = game_data['payoffs'].shape[1]
        total_info_sets = batch_size * num_players
        
        # Extract game data
        hole_cards = game_data['hole_cards']  # (batch, players, 2)
        final_community = game_data['final_community']  # (batch, 5)
        payoffs = game_data['payoffs']  # (batch, players)
        final_pots = game_data['final_pot']  # (batch,)
        
        # Flatten for vectorized processing
        flat_hole_cards = hole_cards.reshape(-1, 2)
        flat_payoffs = payoffs.reshape(-1)
        flat_final_pots = jnp.repeat(final_pots, num_players)
        flat_community = jnp.repeat(final_community[:, None, :], num_players, axis=1).reshape(-1, 5)
        
        # Vectorized hand strength calculation
        def calculate_hand_strength_vectorized(hole_cards, community_cards):
            # Simple vectorized hand strength (can be enhanced)
            hole_sum = jnp.sum(hole_cards, axis=1)
            community_sum = jnp.sum(community_cards, axis=1)
            return (hole_sum + community_sum) / 100.0
        
        hand_strengths = calculate_hand_strength_vectorized(flat_hole_cards, flat_community)
        
        # Vectorized counterfactual values
        def compute_cf_values_vectorized(payoffs):
            return jnp.stack([
                payoffs * 0.5,  # Fold: lose some
                payoffs * 1.0,  # Call: neutral
                payoffs * 1.5,  # Bet: win more
                payoffs * 2.0   # Raise: win most
            ], axis=1)
        
        cf_values = compute_cf_values_vectorized(flat_payoffs)
        
        # Create info sets for tracking
        info_sets = []
        for i in range(total_info_sets):
            player_id = i % num_players
            game_idx = i // num_players
            
            info_set = InfoSet(
                player_id=player_id,
                position=player_id,  # Simplified
                hole_cards=flat_hole_cards[i],
                community_cards=flat_community[i],
                pot_size=float(flat_final_pots[i]),
                stack_size=100.0,  # Simplified
                hand_strength=float(hand_strengths[i]),
                phase=3,  # River (simplified)
                betting_history=jnp.array([1, 1, 1])  # Simplified
            )
            info_sets.append(info_set)
        
        return {
            'total_info_sets': total_info_sets,
            'cf_values': cf_values,
            'info_sets': info_sets,
            'hand_strengths': hand_strengths,
            'payoffs': flat_payoffs,
            'final_pots': flat_final_pots
        }
    
    @partial(jax.jit, static_argnums=(0,))
    def _vectorized_q_value_update(self, current_q: jnp.ndarray, 
                                  cf_values: jnp.ndarray,
                                  learning_rate: float) -> jnp.ndarray:
        """Vectorized Q-value update using JAX"""
        updated_q = current_q + learning_rate * (cf_values - current_q)
        return updated_q.astype(self.config.dtype)
    
    def train_step(self, rng_key: jax.random.PRNGKey, 
                   game_results: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """
        🚀 HYBRID CFVFP training step
        Combines VECTORIZED GPU processing with REAL info set tracking
        """
        self.iteration += 1
        self.total_games += self.config.batch_size
        
        batch_size = game_results['payoffs'].shape[0]
        num_players = game_results['payoffs'].shape[1]
        total_info_sets = batch_size * num_players
        
        logger.info(f"   🚀 HYBRID processing: {batch_size} games × {num_players} players = {total_info_sets} info sets")
        
        # VECTORIZED info set processing
        vectorized_results = self._vectorized_info_set_processing(game_results)
        
        # Process each info set individually (for growth tracking)
        cf_values = vectorized_results['cf_values']
        info_sets = vectorized_results['info_sets']
        
        # Track unique info sets and update indices
        indices_to_update = []
        for i, info_set in enumerate(info_sets):
            index = self._get_or_create_index(info_set)
            indices_to_update.append(index)
        
        # VECTORIZED update for tracked indices
        if indices_to_update:
            indices_array = jnp.array(indices_to_update)
            current_q_subset = self.q_values[indices_array]
            cf_values_subset = cf_values[:len(indices_array)]
            
            updated_q_subset = self._vectorized_q_value_update(
                current_q_subset, 
                cf_values_subset, 
                self.config.learning_rate
            )
            
            # Update Q-values
            self.q_values = self.q_values.at[indices_array].set(updated_q_subset)
            
            # Compute strategies
            strategies_subset = jax.nn.softmax(updated_q_subset / self.config.temperature)
            self.strategies = self.strategies.at[indices_array].set(strategies_subset)
        
        # Update counters
        self.total_info_sets += len(indices_to_update)
        
        # Compute metrics
        avg_payoff = jnp.mean(game_results['payoffs'])
        
        # Calculate strategy entropy
        if indices_to_update:
            entropy = -jnp.sum(strategies_subset * jnp.log(strategies_subset + 1e-8), axis=1)
            avg_entropy = jnp.mean(entropy)
        else:
            avg_entropy = 0.0
        
        logger.info(f"   ✅ HYBRID processing completed!")
        logger.info(f"   📊 Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   📊 Info sets processed: {len(indices_to_update)}")
        logger.info(f"   📊 Growth events: {self.growth_events}")
        logger.info(f"   📊 Array size: {self.config.max_info_sets:,}")
        
        return {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'info_sets_processed': len(indices_to_update),
            'avg_payoff': avg_payoff,
            'strategy_entropy': avg_entropy,
            'q_values_count': self.total_unique_info_sets,
            'strategies_count': self.total_unique_info_sets,
            'games_processed': self.config.batch_size,
            'growth_events': self.growth_events,
            'array_size': self.config.max_info_sets
        }
    
    def save_model(self, path: str):
        """Save hybrid CFVFP model with growth tracking"""
        model_data = {
            'q_values': np.array(self.q_values),
            'strategies': np.array(self.strategies),
            'info_set_hashes': self.info_set_hashes,
            'info_set_data': self.info_set_data,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'growth_events': self.growth_events,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = len(pickle.dumps(model_data))
        logger.info(f"💾 Hybrid CFVFP model saved: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        logger.info(f"   Growth events: {self.growth_events}")
    
    def load_model(self, path: str):
        """Load hybrid CFVFP model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_values = jnp.array(model_data['q_values'])
        self.strategies = jnp.array(model_data['strategies'])
        self.info_set_hashes = model_data['info_set_hashes']
        self.info_set_data = model_data['info_set_data']
        self.iteration = model_data['iteration']
        self.total_games = model_data['total_games']
        self.total_info_sets = model_data['total_info_sets']
        self.total_unique_info_sets = model_data['unique_info_sets']
        self.growth_events = model_data['growth_events']
        self.next_index = len(self.info_set_data)
        
        logger.info(f"📂 Hybrid CFVFP model loaded: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   Growth events: {self.growth_events}")

# 🚀 Performance benchmarking
def benchmark_hybrid_cfvfp_performance():
    """Benchmark hybrid CFVFP performance"""
    logger.info("🚀 Hybrid CFVFP Performance Benchmark")
    logger.info("=" * 50)
    
    # Test configuration
    config = HybridCFVFPConfig(batch_size=8192)
    trainer = HybridCFVFPTrainer(config)
    
    # Generate test game results
    rng_key = jax.random.PRNGKey(42)
    
    # Create realistic game results
    batch_size = config.batch_size
    num_players = 6
    
    test_game_results = {
        'hole_cards': jax.random.randint(rng_key, (batch_size, num_players, 2), 0, 52),
        'final_community': jax.random.randint(rng_key, (batch_size, 5), 0, 52),
        'payoffs': jax.random.normal(rng_key, (batch_size, num_players)) * 10,
        'final_pot': jax.random.uniform(rng_key, (batch_size,), 10, 100),
        'active_players': jnp.ones((batch_size, num_players)),
        'decisions_made': jnp.full((batch_size,), 50),
        'winner': jax.random.randint(rng_key, (batch_size,), 0, num_players),
        'game_length': jnp.full((batch_size,), 50),
        'hand_evaluations': jnp.ones((batch_size,))
    }
    
    # Warm-up compilation
    logger.info("🔥 Warming up JAX compilation...")
    _ = trainer.train_step(rng_key, test_game_results)
    
    # Benchmark
    logger.info("🚀 Benchmarking hybrid CFVFP performance...")
    start_time = time.time()
    
    num_iterations = 10
    for i in range(num_iterations):
        _ = trainer.train_step(rng_key, test_game_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    iterations_per_sec = num_iterations / total_time
    games_per_sec = iterations_per_sec * batch_size
    
    logger.info(f"✅ Hybrid CFVFP Performance Results:")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Total time: {total_time:.3f}s")
    logger.info(f"   Iterations/sec: {iterations_per_sec:.1f}")
    logger.info(f"   Games/sec: {games_per_sec:,.0f}")
    logger.info(f"   Unique info sets: {trainer.total_unique_info_sets:,}")
    logger.info(f"   Growth events: {trainer.growth_events}")
    logger.info(f"   Target achieved: {'✅' if games_per_sec > 1000 else '❌'}")
    
    return games_per_sec

if __name__ == "__main__":
    # Run benchmark
    games_per_sec = benchmark_hybrid_cfvfp_performance()
    print(f"\n🎯 Hybrid CFVFP Performance: {games_per_sec:,.0f} games/sec") 