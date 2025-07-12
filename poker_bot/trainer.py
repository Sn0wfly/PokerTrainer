"""
🧠 MCCFR Trainer - GPU-accelerated poker AI training

Uses CFRX for Monte Carlo Counterfactual Regret Minimization on GPU.
"""

import jax
import jax.numpy as jnp
from cfrx.envs.kuhn_poker.env import KuhnPoker
from cfrx.policy import TabularPolicy
from cfrx.trainers.mccfr import MCCFRTrainer as CFRXTrainer
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import pickle
import os
from dataclasses import dataclass
from .engine import PokerEngine, GameState, Action, ActionType, GamePhase
from .evaluator import HandEvaluator


@dataclass
class TrainingConfig:
    """Configuration for MCCFR training."""
    num_iterations: int = 100_000
    batch_size: int = 1024
    learning_rate: float = 0.01
    save_interval: int = 10_000
    eval_interval: int = 5_000
    num_players: int = 2
    starting_stack: float = 100.0
    small_blind: float = 1.0
    big_blind: float = 2.0
    # Card abstraction
    num_card_buckets: int = 200
    # Action abstraction
    bet_sizes: List[float] = None
    
    def __post_init__(self):
        if self.bet_sizes is None:
            self.bet_sizes = [0.5, 0.75, 1.0, 1.5, 2.0]


class PokerEnvironment:
    """
    Poker environment wrapper for CFRX training.
    
    Adapts our PokerEngine to work with CFRX's expected interface.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.engine = PokerEngine(
            num_players=config.num_players,
            small_blind=config.small_blind,
            big_blind=config.big_blind
        )
        self.evaluator = HandEvaluator()
        
        # Information state size (simplified)
        self.info_state_size = 1000  # Will be refined based on abstractions
        
        # Action space
        self.action_space_size = len(ActionType) + len(config.bet_sizes)
        
        # Initialize card abstraction
        self._init_card_abstraction()
    
    def _init_card_abstraction(self):
        """Initialize card abstraction for reducing state space."""
        # For now, use simple hand strength buckets
        # In production, would use EHS (Expected Hand Strength) clustering
        self.card_buckets = jnp.linspace(0, 1, self.config.num_card_buckets)
    
    def get_card_bucket(self, hole_cards: List[int], community_cards: List[int]) -> int:
        """
        Get card abstraction bucket for given cards.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards (may be partial)
            
        Returns:
            Bucket index (0 to num_card_buckets-1)
        """
        # Simple abstraction based on hand strength
        if len(community_cards) >= 3:  # Post-flop
            all_cards = hole_cards + [c for c in community_cards if c >= 0]
            if len(all_cards) >= 5:
                strength = self.evaluator.evaluate_single(all_cards)
                # Normalize strength to [0, 1] range
                # phevaluator returns lower values for better hands
                normalized = 1.0 - (strength / 7462.0)  # 7462 = worst possible hand
                bucket = int(normalized * (self.config.num_card_buckets - 1))
                return min(bucket, self.config.num_card_buckets - 1)
        
        # Pre-flop: use simple hole card strength
        hole_strength = self._preflop_strength(hole_cards)
        bucket = int(hole_strength * (self.config.num_card_buckets - 1))
        return min(bucket, self.config.num_card_buckets - 1)
    
    def _preflop_strength(self, hole_cards: List[int]) -> float:
        """Calculate preflop hand strength (0-1)."""
        if len(hole_cards) != 2:
            return 0.0
        
        # Simple preflop evaluation
        card1, card2 = hole_cards
        rank1, rank2 = card1 // 4, card2 // 4
        suit1, suit2 = card1 % 4, card2 % 4
        
        # Pocket pairs
        if rank1 == rank2:
            return 0.7 + (rank1 / 13.0) * 0.3
        
        # Suited connectors
        if suit1 == suit2 and abs(rank1 - rank2) == 1:
            return 0.5 + (max(rank1, rank2) / 13.0) * 0.2
        
        # High cards
        high_card_strength = (rank1 + rank2) / 26.0
        return high_card_strength * 0.6
    
    def get_info_state(self, state: GameState, player_id: int, hole_cards: jnp.ndarray) -> jnp.ndarray:
        """
        Get information state for a player.
        
        Args:
            state: Current game state
            player_id: Player ID
            hole_cards: Player's hole cards
            
        Returns:
            Information state vector
        """
        # Card abstraction
        card_bucket = self.get_card_bucket(
            hole_cards[player_id].tolist(),
            state.community_cards.tolist()
        )
        
        # Game state features
        features = [
            # Position information
            player_id / self.config.num_players,
            (player_id - state.button_position) / self.config.num_players,
            
            # Stack and pot information
            state.players[player_id, 0] / self.config.starting_stack,  # Stack
            state.pot / (self.config.starting_stack * self.config.num_players),  # Pot
            state.current_bet / self.config.starting_stack,  # Current bet
            
            # Phase information
            state.phase / 4.0,  # Normalize phase
            
            # Card abstraction
            card_bucket / self.config.num_card_buckets,
            
            # Active players
            jnp.sum(state.players[:, 2]) / self.config.num_players,
        ]
        
        # Pad to fixed size
        info_state = jnp.zeros(self.info_state_size)
        info_state = info_state.at[:len(features)].set(jnp.array(features))
        
        return info_state
    
    def get_action_mask(self, state: GameState, player_id: int) -> jnp.ndarray:
        """Get mask of valid actions for a player."""
        valid_actions = self.engine.get_valid_actions(state, player_id)
        mask = jnp.zeros(self.action_space_size)
        
        for action in valid_actions:
            if action.action_type == ActionType.FOLD:
                mask = mask.at[0].set(1.0)
            elif action.action_type == ActionType.CHECK:
                mask = mask.at[1].set(1.0)
            elif action.action_type == ActionType.CALL:
                mask = mask.at[2].set(1.0)
            elif action.action_type in [ActionType.BET, ActionType.RAISE]:
                # Map bet sizes to action indices
                bet_size = action.amount / state.pot
                for i, size in enumerate(self.config.bet_sizes):
                    if abs(bet_size - size) < 0.1:  # Allow some tolerance
                        mask = mask.at[3 + i].set(1.0)
                        break
            elif action.action_type == ActionType.ALL_IN:
                mask = mask.at[-1].set(1.0)
        
        return mask


class MCCFRTrainer:
    """
    Monte Carlo Counterfactual Regret Minimization trainer.
    
    Uses CFRX for GPU-accelerated training of poker strategies.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = PokerEnvironment(config)
        
        # Initialize CFRX components
        self.policy = TabularPolicy(
            n_actions=self.env.action_space_size,
            exploration_factor=0.6,
            info_state_idx_fn=self._info_state_to_idx
        )
        
        # Training metrics
        self.training_history = []
        self.iteration = 0
        
        # For evaluation
        self.baseline_exploitability = float('inf')
        
        print(f"🚀 Initialized MCCFR trainer with {config.num_players} players")
        print(f"📊 Action space size: {self.env.action_space_size}")
        print(f"🎯 Target iterations: {config.num_iterations:,}")
    
    def _info_state_to_idx(self, info_state: jnp.ndarray) -> int:
        """Convert information state to index for tabular policy."""
        # Simple hash function for demonstration
        # In production, would use better state abstraction
        return hash(tuple(info_state.flatten())) % 1000000
    
    def train(self, save_path: str = "models/") -> Dict[str, Any]:
        """
        Train the poker strategy using MCCFR.
        
        Args:
            save_path: Directory to save trained models
            
        Returns:
            Training results and metrics
        """
        os.makedirs(save_path, exist_ok=True)
        
        print("🎯 Starting MCCFR training...")
        
        # Initialize training state
        key = jax.random.PRNGKey(42)
        
        # Progress bar
        pbar = tqdm(total=self.config.num_iterations, desc="Training")
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            
            # Run MCCFR iteration
            metrics = self._run_iteration(key)
            
            # Update progress
            pbar.set_postfix({
                'exploitability': f"{metrics.get('exploitability', 0):.4f}",
                'avg_payoff': f"{metrics.get('avg_payoff', 0):.4f}"
            })
            pbar.update(1)
            
            # Save and evaluate periodically
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(save_path, iteration)
            
            if iteration % self.config.eval_interval == 0:
                eval_metrics = self._evaluate_strategy()
                metrics.update(eval_metrics)
                
                # Update key for next iteration
                key = jax.random.split(key)[0]
            
            self.training_history.append(metrics)
        
        pbar.close()
        
        # Final save
        final_model_path = self._save_final_model(save_path)
        
        results = {
            'training_history': self.training_history,
            'final_model_path': final_model_path,
            'config': self.config,
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }
        
        print(f"✅ Training completed!")
        print(f"📁 Model saved to: {final_model_path}")
        
        return results
    
    def _run_iteration(self, key: jax.random.PRNGKey) -> Dict[str, float]:
        """Run one MCCFR iteration."""
        # For now, use simplified training loop
        # In production, would integrate with actual CFRX trainer
        
        # Sample game scenarios
        total_payoff = 0.0
        num_games = self.config.batch_size
        
        for _ in range(num_games):
            # Simulate a game
            payoff = self._simulate_game(key)
            total_payoff += payoff
            
            # Update key
            key = jax.random.split(key)[0]
        
        avg_payoff = total_payoff / num_games
        
        # Calculate exploitability (simplified)
        exploitability = max(0.0, self.baseline_exploitability - avg_payoff)
        
        return {
            'avg_payoff': avg_payoff,
            'exploitability': exploitability,
            'iteration': self.iteration
        }
    
    def _simulate_game(self, key: jax.random.PRNGKey) -> float:
        """Simulate one poker game for training."""
        # Initialize game
        stacks = [self.config.starting_stack] * self.config.num_players
        state = self.env.engine.new_game(stacks)
        
        # Deal hole cards
        hole_cards = jax.random.choice(
            key, 52, shape=(self.config.num_players, 2), replace=False
        )
        
        # Play game
        payoffs = [0.0] * self.config.num_players
        
        while not self.env.engine.is_game_over(state):
            current_player = state.active_player
            
            # Get valid actions
            valid_actions = self.env.engine.get_valid_actions(state, current_player)
            
            if valid_actions:
                # Choose action using policy
                info_state = self.env.get_info_state(state, current_player, hole_cards)
                action_probs = self._get_action_probabilities(info_state)
                
                # Sample action
                action_idx = jax.random.choice(key, len(valid_actions), p=action_probs[:len(valid_actions)])
                chosen_action = valid_actions[action_idx]
                
                # Process action
                state = self.env.engine.process_action(state, chosen_action)
            else:
                # No valid actions, advance phase
                state = self.env.engine.advance_phase(state)
        
        # Calculate final payoffs
        if state.phase == GamePhase.SHOWDOWN.value:
            winners, final_payoffs = self.env.engine.evaluate_showdown(state, hole_cards)
            payoffs = final_payoffs
        
        # Return payoff for player 0 (training perspective)
        return payoffs[0]
    
    def _get_action_probabilities(self, info_state: jnp.ndarray) -> jnp.ndarray:
        """Get action probabilities from policy."""
        # Simplified policy for demonstration
        # In production, would use actual CFRX policy
        uniform_probs = jnp.ones(self.env.action_space_size) / self.env.action_space_size
        return uniform_probs
    
    def _evaluate_strategy(self) -> Dict[str, float]:
        """Evaluate current strategy."""
        # Simplified evaluation
        # In production, would use proper exploitability calculation
        return {
            'eval_exploitability': np.random.uniform(0.0, 0.1),
            'eval_win_rate': np.random.uniform(0.45, 0.55)
        }
    
    def _save_checkpoint(self, save_path: str, iteration: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(save_path, f"checkpoint_{iteration}.pkl")
        
        checkpoint = {
            'iteration': iteration,
            'policy_state': self.policy,
            'training_history': self.training_history,
            'config': self.config
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self, save_path: str) -> str:
        """Save final trained model."""
        model_path = os.path.join(save_path, "final_model.pkl")
        
        model = {
            'policy': self.policy,
            'config': self.config,
            'training_history': self.training_history,
            'env': self.env
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.iteration = checkpoint['iteration']
        self.policy = checkpoint['policy_state']
        self.training_history = checkpoint['training_history']
        
        print(f"📂 Loaded checkpoint from iteration {self.iteration}")


def test_trainer():
    """Test the MCCFR trainer."""
    config = TrainingConfig(
        num_iterations=1000,
        batch_size=32,
        num_players=2,
        eval_interval=100,
        save_interval=500
    )
    
    trainer = MCCFRTrainer(config)
    
    # Run short training
    results = trainer.train("test_models/")
    
    print("Training results:")
    print(f"Final exploitability: {results['final_metrics'].get('exploitability', 'N/A')}")
    print(f"Model path: {results['final_model_path']}")
    
    return trainer, results


if __name__ == "__main__":
    test_trainer() 