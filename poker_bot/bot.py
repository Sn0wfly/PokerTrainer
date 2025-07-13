"""
🤖 PokerBot - AI Poker Player

Uses trained MCCFR strategy to play poker in real-time.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pickle
import time
from dataclasses import dataclass
import logging

from .engine import PokerEngine, GameState, Action, ActionType
from .evaluator import HandEvaluator
from .trainer import SimpleMCCFRTrainer, MCCFRConfig

logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Configuration for the poker bot."""
    model_path: str
    thinking_time: float = 0.5  # Seconds to "think"
    aggression_factor: float = 1.0  # Multiplier for aggressive plays
    bluff_frequency: float = 0.1  # Frequency of bluffing
    randomization: float = 0.05  # Small randomization to avoid predictability
    
    # Real-time performance settings
    max_decision_time: float = 30.0  # Max time per decision (tournament rules)
    enable_logging: bool = True

class PokerBot:
    """
    AI Poker Bot using trained MCCFR strategy
    """
    
    def __init__(self, model_path: str, config: Any = None):
        """
        Initialize poker bot with trained model
        
        Args:
            model_path: Path to trained model file
            config: Bot configuration (optional)
        """
        self.model_path = model_path
        self.config = config or BotConfig(model_path=model_path)
        
        # Initialize components
        self.engine = PokerEngine()
        self.evaluator = HandEvaluator()
        self.trainer = None
        
        # Performance tracking
        self.decisions_made = 0
        self.total_decision_time = 0.0
        self.game_results = []
        
        # Load trained model
        self.load_model()
        
        logger.info(f"PokerBot initialized with model: {model_path}")
    
    def load_model(self):
        """Load the trained MCCFR model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create trainer instance and load model
            self.trainer = SimpleMCCFRTrainer(model_data['config'])
            self.trainer.strategy_sum = model_data['strategy_sum']
            self.trainer.regret_sum = model_data['regret_sum']
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            logger.info("Using fallback strategy")
            self._init_fallback_strategy()
    
    def _init_fallback_strategy(self):
        """Initialize fallback strategy if model loading fails"""
        config = MCCFRConfig(iterations=100, batch_size=32, players=2)
        self.trainer = SimpleMCCFRTrainer(config)
        logger.info("Fallback strategy initialized")
    
    def make_decision(self, 
                     game_state: GameState, 
                     hole_cards: List[int], 
                     player_id: int,
                     valid_actions: List[Action]) -> Action:
        """
        Make a poker decision based on current game state
        
        Args:
            game_state: Current game state
            hole_cards: Player's hole cards
            player_id: Player ID
            valid_actions: List of valid actions
            
        Returns:
            Chosen action
        """
        start_time = time.time()
        
        try:
            # Get action from trained strategy
            action = self._strategic_decision(game_state, hole_cards, player_id, valid_actions)
            
            # Add thinking time for realism
            time.sleep(min(self.config.thinking_time, 0.1))
            
        except Exception as e:
            logger.warning(f"Strategic decision failed: {e}")
            action = self._fallback_decision(valid_actions)
        
        decision_time = time.time() - start_time
        self._log_decision(action, decision_time)
        
        return action
    
    def _strategic_decision(self, 
                          game_state: GameState, 
                          hole_cards: List[int], 
                          player_id: int,
                          valid_actions: List[Action]) -> Action:
        """
        Make strategic decision using trained model
        """
        # Get action probabilities from trainer
        action_probs = self.trainer.get_action_probabilities(game_state, player_id)
        
        # Convert to valid actions
        valid_action_names = [self._action_to_name(action) for action in valid_actions]
        
        # Create probability distribution for valid actions
        probs = []
        for action_name in valid_action_names:
            if action_name in action_probs:
                probs.append(action_probs[action_name])
            else:
                probs.append(0.1)  # Small default probability
        
        # Normalize probabilities
        probs = np.array(probs)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)
        
        # Add randomization
        if self.config.randomization > 0:
            noise = np.random.uniform(0, self.config.randomization, len(probs))
            probs = probs + noise
            probs = probs / probs.sum()
        
        # Sample action
        action_idx = np.random.choice(len(valid_actions), p=probs)
        chosen_action = valid_actions[action_idx]
        
        # Apply aggression factor
        chosen_action = self._apply_aggression(chosen_action, game_state)
        
        return chosen_action
    
    def _action_to_name(self, action: Action) -> str:
        """Convert Action enum to string name"""
        if action == Action.FOLD:
            return 'fold'
        elif action in [Action.CHECK, Action.CALL]:
            return 'check_call'
        elif action in [Action.BET, Action.RAISE]:
            return 'bet_raise'
        else:
            return 'fold'
    
    def _apply_aggression(self, action: Action, game_state: GameState) -> Action:
        """Apply aggression factor to action"""
        if self.config.aggression_factor > 1.0:
            # More aggressive: convert calls to raises, checks to bets
            if action == Action.CALL and np.random.random() < (self.config.aggression_factor - 1.0):
                return Action.RAISE
            elif action == Action.CHECK and np.random.random() < (self.config.aggression_factor - 1.0):
                return Action.BET
        
        return action
    
    def _fallback_decision(self, valid_actions: List[Action]) -> Action:
        """
        Simple fallback decision strategy
        """
        # Simple heuristic: prefer check/call over fold, bet/raise occasionally
        if Action.CHECK in valid_actions:
            return Action.CHECK
        elif Action.CALL in valid_actions:
            if np.random.random() < 0.7:  # 70% call, 30% fold
                return Action.CALL
            else:
                return Action.FOLD
        elif Action.BET in valid_actions:
            if np.random.random() < 0.3:  # 30% bet
                return Action.BET
            else:
                return Action.CHECK if Action.CHECK in valid_actions else Action.FOLD
        else:
            return valid_actions[0]  # Default to first valid action
    
    def _log_decision(self, action: Action, decision_time: float):
        """Log decision for performance tracking"""
        self.decisions_made += 1
        self.total_decision_time += decision_time
        
        if self.config.enable_logging:
            logger.debug(f"Decision {self.decisions_made}: {action} in {decision_time:.3f}s")
    
    def update_game_result(self, payoff: float):
        """Update with game result"""
        self.game_results.append(payoff)
        
        if self.config.enable_logging:
            logger.info(f"Game result: {payoff:.2f}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_decision_time = self.total_decision_time / max(self.decisions_made, 1)
        
        stats = {
            'decisions_made': self.decisions_made,
            'avg_decision_time': avg_decision_time,
            'max_decision_time': self.config.max_decision_time,
            'total_games': len(self.game_results),
            'avg_payoff': np.mean(self.game_results) if self.game_results else 0.0
        }
        
        return stats
    
    def reset_session(self):
        """Reset session statistics"""
        self.decisions_made = 0
        self.total_decision_time = 0.0
        self.game_results = []
        
        logger.info("Session reset")
    
    def play_session(self, hands: int = 100, thinking_time: float = 1.0, 
                    aggressive: bool = False) -> Dict:
        """
        Play a session of poker hands
        
        Args:
            hands: Number of hands to play
            thinking_time: Time to think per decision
            aggressive: Whether to play aggressively
            
        Returns:
            Session results
        """
        logger.info(f"Starting session: {hands} hands")
        
        # Adjust configuration
        original_thinking_time = self.config.thinking_time
        original_aggression = self.config.aggression_factor
        
        self.config.thinking_time = thinking_time
        if aggressive:
            self.config.aggression_factor = 1.5
        
        # Reset session
        self.reset_session()
        
        # Play hands (simplified simulation)
        hands_won = 0
        starting_stack = 100.0
        current_stack = starting_stack
        
        for hand_num in range(hands):
            # Simulate hand outcome (in real implementation, would play actual hands)
            game_state = self.engine.new_game()
            
            # Make some decisions during the hand
            for _ in range(np.random.randint(1, 4)):  # 1-3 decisions per hand
                valid_actions = [Action.FOLD, Action.CHECK, Action.BET]
                action = self.make_decision(game_state, [0, 1], 0, valid_actions)
            
            # Simulate hand result
            hand_result = np.random.uniform(-5.0, 5.0)  # Random win/loss
            current_stack += hand_result
            
            if hand_result > 0:
                hands_won += 1
            
            self.update_game_result(hand_result)
        
        # Restore original configuration
        self.config.thinking_time = original_thinking_time
        self.config.aggression_factor = original_aggression
        
        # Return results
        results = {
            'hands_played': hands,
            'hands_won': hands_won,
            'win_rate': hands_won / hands,
            'starting_stack': starting_stack,
            'final_stack': current_stack,
            'profit_loss': current_stack - starting_stack,
            'avg_decision_time': self.total_decision_time / max(self.decisions_made, 1)
        }
        
        logger.info(f"Session completed: {results}")
        return results

# Factory function for easy bot creation
def create_bot(model_path: str, **kwargs) -> PokerBot:
    """
    Create a poker bot with specified configuration
    
    Args:
        model_path: Path to trained model
        **kwargs: Additional configuration options
        
    Returns:
        Configured PokerBot instance
    """
    return PokerBot(model_path=model_path, **kwargs) 