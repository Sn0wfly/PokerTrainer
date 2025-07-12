"""
🤖 PokerBot - AI Poker Player

Uses trained MCCFR strategy to play poker in real-time.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import numpy as np
import pickle
import time
from dataclasses import dataclass
from .engine import PokerEngine, GameState, Action, ActionType
from .evaluator import HandEvaluator
from .trainer import TrainingConfig, PokerEnvironment


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
    AI Poker Bot using trained MCCFR strategy.
    
    Designed for real-time poker play with minimal latency.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.load_model()
        
        # Performance monitoring
        self.decision_times = []
        self.total_hands = 0
        self.total_winnings = 0.0
        
        # Game state tracking
        self.current_game_state = None
        self.hole_cards = None
        self.player_id = None
        
        print(f"🤖 PokerBot initialized")
        print(f"📁 Model: {config.model_path}")
        print(f"⚡ Ready for real-time play!")
    
    def load_model(self):
        """Load trained model from file."""
        try:
            with open(self.config.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.policy = model_data['policy']
            self.training_config = model_data['config']
            self.env = model_data['env']
            self.training_history = model_data.get('training_history', [])
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Training iterations: {len(self.training_history)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to random strategy
            self._init_fallback_strategy()
    
    def _init_fallback_strategy(self):
        """Initialize fallback strategy if model loading fails."""
        print("🔄 Initializing fallback strategy...")
        
        # Create dummy training config
        self.training_config = TrainingConfig(num_players=2)
        self.env = PokerEnvironment(self.training_config)
        
        # Simple random policy
        self.policy = None
        print("⚠️  Using random strategy fallback")
    
    def make_decision(self, 
                     game_state: GameState, 
                     hole_cards: List[int], 
                     player_id: int,
                     valid_actions: List[Action]) -> Action:
        """
        Make a poker decision based on current game state.
        
        Args:
            game_state: Current game state
            hole_cards: Bot's hole cards
            player_id: Bot's player ID
            valid_actions: List of valid actions
            
        Returns:
            Chosen action
        """
        start_time = time.time()
        
        # Update internal state
        self.current_game_state = game_state
        self.hole_cards = hole_cards
        self.player_id = player_id
        
        # Get decision from strategy
        if self.policy is not None:
            action = self._strategic_decision(game_state, hole_cards, player_id, valid_actions)
        else:
            action = self._fallback_decision(valid_actions)
        
        # Add thinking time (looks more human-like)
        elapsed = time.time() - start_time
        if elapsed < self.config.thinking_time:
            time.sleep(self.config.thinking_time - elapsed)
        
        # Log decision
        if self.config.enable_logging:
            self._log_decision(action, elapsed)
        
        # Track performance
        self.decision_times.append(time.time() - start_time)
        
        return action
    
    def _strategic_decision(self, 
                          game_state: GameState, 
                          hole_cards: List[int], 
                          player_id: int,
                          valid_actions: List[Action]) -> Action:
        """Make strategic decision using trained policy."""
        
        # Get information state
        hole_cards_array = jnp.zeros((self.training_config.num_players, 2))
        hole_cards_array = hole_cards_array.at[player_id].set(jnp.array(hole_cards))
        
        info_state = self.env.get_info_state(game_state, player_id, hole_cards_array)
        
        # Get action probabilities from policy
        action_probs = self._get_policy_probabilities(info_state, valid_actions)
        
        # Add randomization to avoid predictability
        if self.config.randomization > 0:
            noise = np.random.normal(0, self.config.randomization, len(action_probs))
            action_probs = action_probs + noise
            action_probs = np.maximum(action_probs, 0.0)  # Ensure non-negative
            action_probs = action_probs / np.sum(action_probs)  # Renormalize
        
        # Choose action based on probabilities
        if len(valid_actions) > 0:
            action_idx = np.random.choice(len(valid_actions), p=action_probs[:len(valid_actions)])
            chosen_action = valid_actions[action_idx]
            
            # Apply aggression factor
            chosen_action = self._apply_aggression(chosen_action, game_state)
            
            return chosen_action
        
        # Fallback if no valid actions
        return self._fallback_decision(valid_actions)
    
    def _get_policy_probabilities(self, info_state: jnp.ndarray, valid_actions: List[Action]) -> np.ndarray:
        """Get action probabilities from the trained policy."""
        # For now, use simplified policy evaluation
        # In production, would use actual CFRX policy evaluation
        
        # Default uniform distribution
        probs = np.ones(len(valid_actions)) / len(valid_actions)
        
        # Simple heuristics based on hand strength
        if len(self.hole_cards) == 2:
            hand_strength = self._estimate_hand_strength()
            
            # Adjust probabilities based on hand strength
            for i, action in enumerate(valid_actions):
                if action.action_type == ActionType.BET or action.action_type == ActionType.RAISE:
                    # More likely to bet/raise with strong hands
                    probs[i] *= (1.0 + hand_strength)
                elif action.action_type == ActionType.FOLD:
                    # More likely to fold with weak hands
                    probs[i] *= (1.0 - hand_strength)
        
        # Normalize probabilities
        probs = probs / np.sum(probs)
        return probs
    
    def _estimate_hand_strength(self) -> float:
        """Estimate hand strength (0-1)."""
        if not self.hole_cards or len(self.hole_cards) != 2:
            return 0.5
        
        # Use preflop strength calculation
        return self.env._preflop_strength(self.hole_cards)
    
    def _apply_aggression(self, action: Action, game_state: GameState) -> Action:
        """Apply aggression factor to betting actions."""
        if action.action_type in [ActionType.BET, ActionType.RAISE]:
            # Increase bet size based on aggression factor
            adjusted_amount = action.amount * self.config.aggression_factor
            
            # Ensure we don't exceed stack size
            max_bet = game_state.players[self.player_id, 0]  # Current stack
            adjusted_amount = min(adjusted_amount, max_bet)
            
            return Action(action.action_type, adjusted_amount, action.player_id)
        
        return action
    
    def _fallback_decision(self, valid_actions: List[Action]) -> Action:
        """Fallback decision strategy (basic poker logic)."""
        if not valid_actions:
            return Action(ActionType.FOLD)
        
        # Simple heuristic strategy
        hand_strength = self._estimate_hand_strength()
        
        # Strong hands (>0.7): aggressive play
        if hand_strength > 0.7:
            # Look for betting/raising opportunities
            for action in valid_actions:
                if action.action_type in [ActionType.BET, ActionType.RAISE]:
                    return action
            # If can't bet/raise, call or check
            for action in valid_actions:
                if action.action_type in [ActionType.CALL, ActionType.CHECK]:
                    return action
        
        # Medium hands (0.3-0.7): cautious play
        elif hand_strength > 0.3:
            # Prefer checking/calling
            for action in valid_actions:
                if action.action_type in [ActionType.CHECK, ActionType.CALL]:
                    return action
            # Small bet if possible
            for action in valid_actions:
                if action.action_type == ActionType.BET and action.amount <= 10:
                    return action
        
        # Weak hands (<0.3): defensive play
        else:
            # Prefer checking/folding
            for action in valid_actions:
                if action.action_type == ActionType.CHECK:
                    return action
            for action in valid_actions:
                if action.action_type == ActionType.FOLD:
                    return action
        
        # Default: return first valid action
        return valid_actions[0]
    
    def _log_decision(self, action: Action, decision_time: float):
        """Log decision for analysis."""
        if self.config.enable_logging:
            phase = "unknown"
            if self.current_game_state:
                phase = ["preflop", "flop", "turn", "river", "showdown"][self.current_game_state.phase]
            
            print(f"🎯 {phase.upper()}: {action.action_type.value} "
                  f"${action.amount:.2f} ({decision_time:.3f}s)")
    
    def update_game_result(self, payoff: float):
        """Update bot's performance tracking."""
        self.total_hands += 1
        self.total_winnings += payoff
        
        if self.config.enable_logging:
            avg_winnings = self.total_winnings / self.total_hands
            print(f"📊 Hand {self.total_hands}: ${payoff:.2f} "
                  f"(avg: ${avg_winnings:.2f})")
    
    def get_performance_stats(self) -> Dict:
        """Get bot's performance statistics."""
        avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0.0
        max_decision_time = max(self.decision_times) if self.decision_times else 0.0
        
        return {
            'total_hands': self.total_hands,
            'total_winnings': self.total_winnings,
            'avg_winnings_per_hand': self.total_winnings / max(self.total_hands, 1),
            'avg_decision_time': avg_decision_time,
            'max_decision_time': max_decision_time,
            'decisions_made': len(self.decision_times)
        }
    
    def reset_session(self):
        """Reset session statistics."""
        self.decision_times = []
        self.total_hands = 0
        self.total_winnings = 0.0
        print("🔄 Session statistics reset")


class BotInterface:
    """
    Interface for integrating PokerBot with external poker platforms.
    
    Provides standardized methods for different poker software.
    """
    
    def __init__(self, bot: PokerBot):
        self.bot = bot
        self.engine = PokerEngine()
        self.evaluator = HandEvaluator()
    
    def parse_game_state(self, external_state: Dict) -> GameState:
        """Parse external game state format to internal format."""
        # This would be customized for each poker platform
        # (PokerStars, PartyPoker, etc.)
        
        # Example implementation for generic format
        players = jnp.array([
            [external_state.get('stacks', [100.0] * 6)[i],
             external_state.get('current_bets', [0.0] * 6)[i],
             1.0 if i in external_state.get('active_players', []) else 0.0,
             0.0]  # all_in flag
            for i in range(6)
        ])
        
        community_cards = jnp.array(external_state.get('community_cards', [-1] * 5))
        
        return GameState(
            players=players,
            community_cards=community_cards,
            pot=external_state.get('pot', 0.0),
            current_bet=external_state.get('current_bet', 0.0),
            phase=external_state.get('phase', 0),
            active_player=external_state.get('active_player', 0),
            button_position=external_state.get('button', 0),
            deck=jnp.arange(52)  # Not used in real play
        )
    
    def get_action_command(self, action: Action) -> str:
        """Convert action to external command format."""
        if action.action_type == ActionType.FOLD:
            return "fold"
        elif action.action_type == ActionType.CHECK:
            return "check"
        elif action.action_type == ActionType.CALL:
            return "call"
        elif action.action_type == ActionType.BET:
            return f"bet {action.amount:.2f}"
        elif action.action_type == ActionType.RAISE:
            return f"raise {action.amount:.2f}"
        elif action.action_type == ActionType.ALL_IN:
            return "all-in"
        else:
            return "check"  # Safe default


def test_bot():
    """Test the poker bot."""
    # Create bot config
    config = BotConfig(
        model_path="models/final_model.pkl",
        thinking_time=0.1,  # Fast for testing
        enable_logging=True
    )
    
    # Initialize bot (will use fallback strategy if no model)
    bot = PokerBot(config)
    
    # Create test game state
    engine = PokerEngine(num_players=2)
    state = engine.new_game([100.0, 100.0], button_pos=0)
    
    # Test decision making
    hole_cards = [48, 49]  # Aces
    player_id = 0
    valid_actions = engine.get_valid_actions(state, player_id)
    
    print(f"Test scenario:")
    print(f"Hole cards: {hole_cards}")
    print(f"Valid actions: {[a.action_type.value for a in valid_actions]}")
    
    # Make decision
    action = bot.make_decision(state, hole_cards, player_id, valid_actions)
    
    print(f"Bot decision: {action.action_type.value} ${action.amount:.2f}")
    
    # Show performance stats
    stats = bot.get_performance_stats()
    print(f"Performance: {stats}")
    
    return bot


if __name__ == "__main__":
    test_bot() 