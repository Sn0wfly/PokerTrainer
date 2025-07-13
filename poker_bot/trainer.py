"""
Simplified MCCFR Trainer for Texas Hold'em
Using JAX directly without CFRX dependency
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import logging

from .engine import PokerEngine, GameState, Action
from .evaluator import HandEvaluator

logger = logging.getLogger(__name__)


@dataclass
class MCCFRConfig:
    """Configuration for MCCFR training"""
    iterations: int = 100000
    batch_size: int = 1024
    players: int = 2
    learning_rate: float = 0.1
    exploration: float = 0.1
    save_interval: int = 1000
    log_interval: int = 100


class SimpleMCCFRTrainer:
    """
    Simplified Monte Carlo Counterfactual Regret Minimization trainer
    Uses JAX for acceleration but avoids CFRX dependency
    """
    
    def __init__(self, config: MCCFRConfig):
        self.config = config
        self.engine = PokerEngine()
        self.evaluator = HandEvaluator()
        
        # Initialize strategy tables
        self.strategy_sum: Dict[str, jnp.ndarray] = {}
        self.regret_sum: Dict[str, jnp.ndarray] = {}
        
        # Action space
        self.actions = ['fold', 'check_call', 'bet_raise']
        self.n_actions = len(self.actions)
        
        # JAX random key
        self.key = jr.PRNGKey(42)
        
        # Simple game counter for terminal condition
        self.max_actions_per_game = 10
        
        logger.info(f"Initialized SimpleMCCFRTrainer with config: {config}")
    
    def get_information_set(self, game_state: GameState, player_id: int) -> str:
        """
        Create information set string for current game state
        """
        # Get player's hole cards
        hole_cards = self.engine.get_hole_cards(game_state, player_id)
        
        # Get community cards
        community = game_state.community_cards
        
        # Get betting history
        betting_history = game_state.betting_history[-10:]  # Last 10 actions
        
        # Create info set string
        info_set = f"hole:{hole_cards}_community:{community}_betting:{betting_history}"
        
        return info_set
    
    def get_strategy(self, info_set: str) -> jnp.ndarray:
        """
        Get strategy for information set using regret matching
        """
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = jnp.zeros(self.n_actions)
        
        regrets = self.regret_sum[info_set]
        
        # Regret matching: set negative regrets to zero
        positive_regrets = jnp.maximum(regrets, 0)
        
        # Normalize to create strategy
        regret_sum = jnp.sum(positive_regrets)
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Default uniform strategy
            strategy = jnp.ones(self.n_actions) / self.n_actions
        
        return strategy
    
    def update_strategy(self, info_set: str, strategy: jnp.ndarray):
        """
        Update strategy sum for average strategy calculation
        """
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = jnp.zeros(self.n_actions)
        
        self.strategy_sum[info_set] += strategy
    
    def sample_action(self, strategy: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[int, float]:
        """
        Sample action according to strategy
        """
        action_idx = jr.choice(key, self.n_actions, p=strategy)
        return action_idx, strategy[action_idx]
    
    def simulate_simple_game(self, key: jax.random.PRNGKey) -> float:
        """
        Simplified game simulation that avoids infinite loops
        """
        # Start new game
        game_state = self.engine.new_game()
        
        # Track actions to prevent infinite loops
        action_count = 0
        
        # Simple simulation loop
        while action_count < self.max_actions_per_game:
            # Check if game should end
            if game_state.is_terminal():
                break
            
            # Get current player
            current_player = game_state.current_player
            
            # Get strategy for current player
            info_set = self.get_information_set(game_state, current_player)
            strategy = self.get_strategy(info_set)
            
            # Sample action
            key, subkey = jr.split(key)
            action_idx, prob = self.sample_action(strategy, subkey)
            
            # Apply action
            try:
                game_state = self.apply_action(game_state, action_idx)
            except Exception as e:
                logger.warning(f"Action failed: {e}, ending game")
                break
            
            action_count += 1
        
        # Return simple utility (player 0 perspective)
        if game_state.winner == 0:
            return 1.0
        elif game_state.winner == 1:
            return -1.0
        else:
            # Draw or timeout
            return 0.0
    
    def apply_action(self, game_state: GameState, action_idx: int) -> GameState:
        """
        Apply action to game state
        """
        action_name = self.actions[action_idx]
        
        if action_name == 'fold':
            action = Action.FOLD
        elif action_name == 'check_call':
            action = Action.CHECK if game_state.current_bet == 0 else Action.CALL
        else:  # bet_raise
            action = Action.BET if game_state.current_bet == 0 else Action.RAISE
        
        return self.engine.apply_action(game_state, action)
    
    def train_iteration(self, iteration: int) -> float:
        """
        Single training iteration using simplified simulation
        """
        total_utility = 0.0
        
        for _ in range(self.config.batch_size):
            # Simple game simulation
            self.key, subkey = jr.split(self.key)
            utility = self.simulate_simple_game(subkey)
            total_utility += utility
        
        return total_utility / self.config.batch_size
    
    def train(self, save_path: str = "models/mccfr_model.pkl"):
        """
        Main training loop
        """
        logger.info("Starting MCCFR training...")
        
        for iteration in tqdm(range(self.config.iterations), desc="Training"):
            # Train iteration
            avg_utility = self.train_iteration(iteration)
            
            # Logging
            if iteration % self.config.log_interval == 0:
                logger.info(f"Iteration {iteration}: Average utility = {avg_utility:.4f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_model(save_path)
        
        # Final save
        self.save_model(save_path)
        logger.info(f"Training completed. Model saved to {save_path}")
    
    def save_model(self, path: str):
        """
        Save trained model
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'strategy_sum': self.strategy_sum,
            'regret_sum': self.regret_sum,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """
        Load trained model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.strategy_sum = model_data['strategy_sum']
        self.regret_sum = model_data['regret_sum']
        self.config = model_data['config']
    
    def get_action_probabilities(self, game_state: GameState, player_id: int) -> Dict[str, float]:
        """
        Get action probabilities for a given game state
        """
        info_set = self.get_information_set(game_state, player_id)
        strategy = self.get_strategy(info_set)
        
        return {
            self.actions[i]: float(strategy[i]) for i in range(self.n_actions)
        }


def create_trainer(iterations: int = 100000, batch_size: int = 1024, 
                  players: int = 2, **kwargs) -> SimpleMCCFRTrainer:
    """Create a MCCFR trainer with given parameters"""
    config = MCCFRConfig(
        iterations=iterations,
        batch_size=batch_size,
        players=players,
        **kwargs
    )
    return SimpleMCCFRTrainer(config) 