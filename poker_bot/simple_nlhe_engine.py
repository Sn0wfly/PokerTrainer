"""
Simple NLHE 6-Player Engine - Works without complex JAX operations
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Tuple, List
import time

@jax.jit
def simple_nlhe_batch(rng_keys: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    SIMPLE NLHE 6-PLAYER: Basic Texas Hold'em simulation
    """
    batch_size = rng_keys.shape[0]
    
    # Deal cards
    deck = jnp.arange(52)
    shuffled_decks = jax.vmap(lambda key: jr.permutation(key, deck))(rng_keys)
    
    # Hole cards and community cards
    hole_cards = shuffled_decks[:, :12].reshape(batch_size, 6, 2)
    community_cards = shuffled_decks[:, 12:17]
    
    # Simple hand strength calculation
    hand_strengths = simple_hand_evaluation(hole_cards, community_cards)
    
    # Simple winner determination
    winners = jnp.argmax(hand_strengths, axis=1)
    
    # Simple payoffs
    payoffs = jnp.zeros((batch_size, 6))
    winner_indices = jnp.arange(batch_size)
    payoffs = payoffs.at[winner_indices, winners].set(100.0)  # Fixed pot size
    
    return {
        'hole_cards': hole_cards,
        'community_cards': community_cards,
        'hand_strengths': hand_strengths,
        'winners': winners,
        'payoffs': payoffs
    }

@jax.jit
def simple_hand_evaluation(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    SIMPLE: Calculate hand strengths using basic rules
    """
    batch_size = hole_cards.shape[0]
    
    # Combine hole cards with community cards
    all_cards = jnp.concatenate([
        hole_cards,  # (batch, 6, 2)
        community_cards[:, None, :].repeat(6, axis=1)  # (batch, 6, 5)
    ], axis=2)
    
    # Simple strength calculation based on high cards
    ranks = all_cards // 4
    suits = all_cards % 4
    
    # Calculate basic strength (high card + pair detection)
    high_cards = jnp.max(ranks, axis=2)
    second_high = jnp.sort(ranks, axis=2)[:, :, -2]
    
    # Simple pair detection
    rank_counts = jnp.zeros((batch_size, 6, 13))
    for rank in range(13):
        rank_counts = rank_counts.at[:, :, rank].set(jnp.sum(ranks == rank, axis=2))
    
    pairs = jnp.sum(rank_counts >= 2, axis=2)
    
    # Calculate strength
    strength = high_cards * 10.0 + second_high + pairs * 100.0
    
    return strength

def test_simple_nlhe():
    """Test simple NLHE engine"""
    import time
    
    batch_size = 1000
    rng_key = jr.PRNGKey(42)
    rng_keys = jr.split(rng_key, batch_size)
    
    print("Testing Simple NLHE 6-player engine...")
    
    # Warm-up
    _ = simple_nlhe_batch(rng_keys)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        results = simple_nlhe_batch(rng_keys)
        results['payoffs'].block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / 10
    games_per_second = batch_size / avg_time
    
    print(f"Simple NLHE 6-player performance:")
    print(f"  Average time: {avg_time:.3f} seconds")
    print(f"  Games per second: {games_per_second:.1f}")
    print(f"  Batch size: {batch_size}")
    
    return results

if __name__ == "__main__":
    test_simple_nlhe() 