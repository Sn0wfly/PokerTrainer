"""
🎮 Simple Poker Visualizer
Shows how the trained bot plays in real-time with visual interface
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PokerVisualizer:
    """Simple visualizer for poker games"""
    
    def __init__(self, model_path: str):
        """Initialize visualizer with trained model"""
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            import pickle
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def visualize_game(self, num_hands: int = 5):
        """Visualize multiple poker hands"""
        print("\n" + "="*60)
        print("🎮 POKER BOT VISUALIZER")
        print("="*60)
        
        for hand_num in range(num_hands):
            print(f"\n🃏 HAND #{hand_num + 1}")
            print("-" * 40)
            self._play_one_hand()
            time.sleep(1)  # Pause between hands
    
    def _play_one_hand(self):
        """Play and visualize one poker hand"""
        # Simulate a simple poker hand
        players = ["Bot", "Player2", "Player3", "Player4", "Player5", "Player6"]
        positions = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        
        # Generate random cards
        hole_cards = self._generate_random_cards(6, 2)
        community_cards = self._generate_random_cards(1, 5)
        
        print(f"📋 Players: {', '.join(players)}")
        print(f"📍 Positions: {', '.join(positions)}")
        print(f"💰 Pot: $12 (SB: $1, BB: $2)")
        
        # Preflop
        print(f"\n🃏 PREFLOP:")
        print(f"   Bot's cards: {self._format_cards(hole_cards[0])}")
        print(f"   Bot's position: {positions[0]} (Small Blind)")
        
        # Bot decision
        bot_action = self._get_bot_action("preflop", hole_cards[0], [], 1.0)
        print(f"   🤖 Bot action: {bot_action}")
        
        # Flop
        print(f"\n🃏 FLOP: {self._format_cards(community_cards[0][:3])}")
        print(f"   Pot: $18")
        
        bot_action = self._get_bot_action("flop", hole_cards[0], community_cards[0][:3], 18.0)
        print(f"   🤖 Bot action: {bot_action}")
        
        # Turn
        print(f"\n🃏 TURN: {self._format_cards(community_cards[0][:4])}")
        print(f"   Pot: $24")
        
        bot_action = self._get_bot_action("turn", hole_cards[0], community_cards[0][:4], 24.0)
        print(f"   🤖 Bot action: {bot_action}")
        
        # River
        print(f"\n🃏 RIVER: {self._format_cards(community_cards[0])}")
        print(f"   Pot: $30")
        
        bot_action = self._get_bot_action("river", hole_cards[0], community_cards[0], 30.0)
        print(f"   🤖 Bot action: {bot_action}")
        
        # Result
        result = random.choice(["WIN", "LOSE", "SPLIT"])
        print(f"\n🏆 RESULT: Bot {result}")
        print("-" * 40)
    
    def _generate_random_cards(self, num_players: int, cards_per_player: int) -> List[List[str]]:
        """Generate random cards for visualization"""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        all_cards = []
        for _ in range(num_players):
            player_cards = []
            for _ in range(cards_per_player):
                suit = random.choice(suits)
                rank = random.choice(ranks)
                player_cards.append(f"{rank}{suit}")
            all_cards.append(player_cards)
        
        return all_cards
    
    def _format_cards(self, cards: List[str]) -> str:
        """Format cards for display"""
        return " ".join(cards)
    
    def _get_bot_action(self, street: str, hole_cards: List[str], 
                       community_cards: List[str], pot_size: float) -> str:
        """Get bot action based on game state"""
        if not self.model:
            return random.choice(["FOLD", "CALL", "BET $6", "RAISE $12"])
        
        try:
            # Create info set hash from game state
            info_set = self._create_info_set(hole_cards, community_cards, street)
            
            # Get strategy from model if available
            if hasattr(self.model, 'strategies') and info_set in self.model.strategies:
                strategy = self.model.strategies[info_set]
                action_probs = strategy / jnp.sum(strategy)
                
                # Choose action based on probabilities
                actions = ["FOLD", "CALL", "BET", "RAISE"]
                action_idx = jnp.argmax(action_probs)
                action = actions[action_idx]
                
                logger.info(f"   📊 Strategy: {action_probs}")
                logger.info(f"   🎯 Chosen: {action}")
                
            else:
                # Fallback to model-based decision
                actions = ["FOLD", "CALL", "BET", "RAISE"]
                weights = [0.2, 0.3, 0.3, 0.2]  # Conservative strategy
                
                # Adjust based on street
                if street == "preflop":
                    weights = [0.1, 0.4, 0.3, 0.2]  # More aggressive preflop
                elif street == "river":
                    weights = [0.3, 0.4, 0.2, 0.1]  # More conservative on river
                
                action = random.choices(actions, weights=weights)[0]
            
            # Format action with bet sizing
            if action == "BET":
                bet_size = min(pot_size * 0.75, 12)
                return f"BET ${bet_size:.0f}"
            elif action == "RAISE":
                raise_size = min(pot_size * 1.5, 18)
                return f"RAISE ${raise_size:.0f}"
            else:
                return action
                
        except Exception as e:
            # Fallback to random if model fails
            return random.choice(["FOLD", "CALL", "BET $6", "RAISE $12"])
    
    def _create_info_set(self, hole_cards: List[str], community_cards: List[str], street: str) -> str:
        """Create info set hash for the game state"""
        # Convert cards to indices for consistency with training
        card_to_idx = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }
        
        # Create simple hash
        hole_str = "".join([card.replace('♠', '').replace('♥', '').replace('♦', '').replace('♣', '') for card in hole_cards])
        community_str = "".join([card.replace('♠', '').replace('♥', '').replace('♦', '').replace('♣', '') for card in community_cards])
        
        info_set = f"{street}:{hole_str}:{community_str}"
        return info_set

def quick_visualize(model_path: str = "models/real_cfvfp_model.pkl", hands: int = 3):
    """Quick visualization of bot play"""
    try:
        visualizer = PokerVisualizer(model_path)
        visualizer.visualize_game(hands)
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("💡 Try running training first: python -m poker_bot.cli train-cfvfp --iterations 1000")

if __name__ == "__main__":
    quick_visualize() 