"""
Modern CFR Implementation using CFVFP (NeurIPS 2024)
Counterfactual Value Based Fictitious Play
JAX-optimized for GPU acceleration
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import numpy as np
from dataclasses import dataclass
from functools import partial
import logging

from .gpu_config import setup_mixed_precision

logger = logging.getLogger(__name__)

@dataclass
class CFVFPConfig:
    """Configuration for CFVFP training"""
    iterations: int = 1000000
    batch_size: int = 1024
    learning_rate: float = 0.1
    temperature: float = 1.0
    exploration_rate: float = 0.1
    update_interval: int = 100
    save_interval: int = 10000
    dtype: jnp.dtype = jnp.bfloat16
    accumulation_dtype: jnp.dtype = jnp.float32

class InfoState(NamedTuple):
    """Represents an information state in the game"""
    player_id: int
    cards: jnp.ndarray
    history: jnp.ndarray
    pot: float
    round: int

class ActionValue(NamedTuple):
    """Action-value pair for Q-learning"""
    action: int
    value: float
    probability: float

class CFVFPTrainer:
    """
    CFVFP: Counterfactual Value Based Fictitious Play
    
    Key Innovation: Uses Q-values instead of regret values
    - Direct max Q-value action selection
    - Avoids dominated strategy selection
    - Faster convergence than traditional CFR
    """
    
    def __init__(self, config: CFVFPConfig):
        self.config = config
        self.precision_config = setup_mixed_precision()
        
        # Q-values storage (key: info_state, value: action_values)
        self.q_values: Dict[str, jnp.ndarray] = {}
        
        # Strategy storage
        self.strategies: Dict[str, jnp.ndarray] = {}
        
        # Average strategy (for final policy)
        self.average_strategy: Dict[str, jnp.ndarray] = {}
        
        # Training state
        self.iteration = 0
        self.total_utility = 0.0
        
        logger.info(f"CFVFP Trainer initialized with config: {config}")
    
    def _info_state_to_key(self, info_state: InfoState) -> str:
        """Convert info state to string key for dictionary storage"""
        return f"p{info_state.player_id}_c{hash(info_state.cards.tobytes())}_h{hash(info_state.history.tobytes())}_r{info_state.round}"
    
    @partial(jit, static_argnums=(0,))
    def _update_q_values(self, 
                        current_q: jnp.ndarray, 
                        action_values: jnp.ndarray,
                        learning_rate: float) -> jnp.ndarray:
        """Update Q-values with new observations"""
        # Q-learning update rule
        updated_q = current_q + learning_rate * (action_values - current_q)
        return updated_q.astype(self.config.dtype)
    
    @partial(jit, static_argnums=(0,))
    def _compute_strategy(self, q_values: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """
        Compute strategy from Q-values
        CFVFP Innovation: Direct max Q-value action selection
        """
        # Softmax with temperature for exploration
        logits = q_values / temperature
        
        # Numerical stability
        logits = logits - jnp.max(logits)
        
        # Softmax in float32 for numerical stability
        probs = jax.nn.softmax(logits.astype(jnp.float32))
        
        return probs.astype(self.config.dtype)
    
    @partial(jit, static_argnums=(0,))
    def _select_action(self, strategy: jnp.ndarray, key: jnp.ndarray) -> int:
        """Select action based on strategy"""
        return random.choice(key, len(strategy), p=strategy.astype(jnp.float32))
    
    def get_or_create_q_values(self, info_state: InfoState, num_actions: int) -> jnp.ndarray:
        """Get Q-values for info state, create if doesn't exist"""
        key = self._info_state_to_key(info_state)
        
        if key not in self.q_values:
            # Initialize Q-values uniformly
            self.q_values[key] = jnp.zeros(num_actions, dtype=self.config.dtype)
            
        return self.q_values[key]
    
    def update_info_state(self, 
                         info_state: InfoState, 
                         action_values: jnp.ndarray,
                         num_actions: int) -> jnp.ndarray:
        """Update Q-values and strategy for an information state"""
        key = self._info_state_to_key(info_state)
        
        # Get current Q-values
        current_q = self.get_or_create_q_values(info_state, num_actions)
        
        # Update Q-values
        updated_q = self._update_q_values(
            current_q, 
            action_values, 
            self.config.learning_rate
        )
        
        # Store updated Q-values
        self.q_values[key] = updated_q
        
        # Compute new strategy
        strategy = self._compute_strategy(updated_q, self.config.temperature)
        self.strategies[key] = strategy
        
        # Update average strategy for final policy
        if key not in self.average_strategy:
            self.average_strategy[key] = jnp.zeros_like(strategy)
        
        # Running average
        alpha = 1.0 / (self.iteration + 1)
        self.average_strategy[key] = (
            (1 - alpha) * self.average_strategy[key] + alpha * strategy
        )
        
        return strategy
    
    def get_strategy(self, info_state: InfoState, num_actions: int) -> jnp.ndarray:
        """Get current strategy for an information state"""
        key = self._info_state_to_key(info_state)
        
        if key not in self.strategies:
            # Initialize with uniform strategy
            q_values = self.get_or_create_q_values(info_state, num_actions)
            strategy = self._compute_strategy(q_values, self.config.temperature)
            self.strategies[key] = strategy
            
        return self.strategies[key]
    
    def get_average_strategy(self, info_state: InfoState) -> Optional[jnp.ndarray]:
        """Get average strategy for final policy"""
        key = self._info_state_to_key(info_state)
        return self.average_strategy.get(key)
    
    @partial(jit, static_argnums=(0,))
    def _compute_counterfactual_value(self, 
                                     utilities: jnp.ndarray,
                                     strategy: jnp.ndarray,
                                     action_taken: int) -> float:
        """
        Compute counterfactual value for CFVFP
        Key difference: Uses Q-values instead of regret values
        """
        # Expected utility under current strategy
        expected_utility = jnp.sum(utilities * strategy)
        
        # Counterfactual value = difference from expected
        counterfactual_value = utilities[action_taken] - expected_utility
        
        return counterfactual_value.astype(self.config.accumulation_dtype)
    
    def train_step(self, 
                   game_state: Any,
                   player_id: int,
                   key: jnp.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Single training step of CFVFP
        """
        self.iteration += 1
        
        # This would be implemented with actual game logic
        # For now, returning placeholder values
        
        utility = 0.0
        metrics = {
            'iteration': self.iteration,
            'q_values_count': len(self.q_values),
            'strategies_count': len(self.strategies),
            'average_strategies_count': len(self.average_strategy),
        }
        
        return utility, metrics
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'q_values': self.q_values,
            'strategies': self.strategies,
            'average_strategy': self.average_strategy,
            'iteration': self.iteration,
            'total_utility': self.total_utility,
            'config': self.config,
        }
        
        # Convert JAX arrays to numpy for saving
        checkpoint_np = jax.tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            checkpoint
        )
        
        np.savez_compressed(filepath, **checkpoint_np)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = np.load(filepath, allow_pickle=True)
        
        # Convert numpy arrays back to JAX
        self.q_values = {
            k: jnp.array(v) for k, v in checkpoint['q_values'].item().items()
        }
        self.strategies = {
            k: jnp.array(v) for k, v in checkpoint['strategies'].item().items()
        }
        self.average_strategy = {
            k: jnp.array(v) for k, v in checkpoint['average_strategy'].item().items()
        }
        
        self.iteration = int(checkpoint['iteration'])
        self.total_utility = float(checkpoint['total_utility'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_policy(self) -> Dict[str, jnp.ndarray]:
        """Get final policy (average strategy)"""
        return self.average_strategy.copy()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'iteration': self.iteration,
            'total_utility': self.total_utility,
            'q_values_count': len(self.q_values),
            'strategies_count': len(self.strategies),
            'average_strategies_count': len(self.average_strategy),
            'config': self.config,
        }


# Vectorized training functions for batch processing
@jit
def batch_update_q_values(q_values: jnp.ndarray, 
                         action_values: jnp.ndarray,
                         learning_rate: float) -> jnp.ndarray:
    """Vectorized Q-value updates for batch processing"""
    return q_values + learning_rate * (action_values - q_values)

@jit  
def batch_compute_strategies(q_values: jnp.ndarray, 
                           temperature: float) -> jnp.ndarray:
    """Vectorized strategy computation for batch processing"""
    logits = q_values / temperature
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jax.nn.softmax(logits, axis=-1)

# Utility functions for integration with existing code
def create_cfvfp_trainer(config: Optional[CFVFPConfig] = None) -> CFVFPTrainer:
    """Create CFVFP trainer with default or custom config"""
    if config is None:
        config = CFVFPConfig()
    
    return CFVFPTrainer(config)

def info_state_from_game(game_state: Any, player_id: int) -> InfoState:
    """Convert game state to InfoState (placeholder)"""
    # This would be implemented based on actual game state structure
    return InfoState(
        player_id=player_id,
        cards=jnp.array([0, 1]),  # placeholder
        history=jnp.array([0]),   # placeholder
        pot=0.0,
        round=0
    ) 