"""
🎮 Poker Engine - NLHE Game State and Rules

Handles Texas Hold'em game logic, state management, and action processing.
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import evaluator here to avoid circular import
# from .evaluator import HandEvaluator


class Action(Enum):
    """Possible actions in poker."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class GamePhase(Enum):
    """Phases of poker game."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass
class GameConfig:
    """Configuration for poker game."""
    players: int = 2
    starting_stack: float = 100.0
    small_blind: float = 1.0
    big_blind: float = 2.0
    max_raises: int = 3


@dataclass
class Player:
    """Represents a poker player."""
    player_id: int
    stack: float
    hole_cards: List[int]
    is_active: bool = True
    is_all_in: bool = False
    has_acted: bool = False


class GameState(NamedTuple):
    """Immutable game state for JAX compatibility."""
    players: jnp.ndarray  # Player stacks and bets
    community_cards: jnp.ndarray  # Board cards
    pot_size: float
    current_bet: float
    phase: int  # GamePhase as integer
    current_player: int
    button_position: int
    deck: jnp.ndarray  # Remaining cards in deck
    betting_history: List[str]  # Action history for this round
    winner: int = -1  # -1 = no winner yet
    
    def is_terminal(self) -> bool:
        """Check if game is in terminal state"""
        return self.winner != -1 or self.phase >= 4


class PokerEngine:
    """
    Texas Hold'em poker engine with JAX compatibility.
    
    Handles game state, action processing, and game flow.
    """
    
    def __init__(self, config: GameConfig = None):
        """
        Initialize poker engine.
        
        Args:
            config: Game configuration
        """
        self.config = config or GameConfig()
        # Initialize evaluator lazily to avoid circular import
        self.evaluator = None
        
        # Action mappings
        self.action_space = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE]
        
        # Initialize random key for JAX
        self.key = jax.random.PRNGKey(42)
    
    def _get_evaluator(self):
        """Get evaluator instance (lazy initialization)"""
        if self.evaluator is None:
            from .evaluator import HandEvaluator
            self.evaluator = HandEvaluator()
        return self.evaluator
    
    def new_game(self, stacks: List[float] = None, button_pos: int = 0) -> GameState:
        """
        Create a new game state.
        
        Args:
            stacks: Starting stack sizes for each player
            button_pos: Position of the button
            
        Returns:
            Initial game state
        """
        if stacks is None:
            stacks = [self.config.starting_stack] * self.config.players
        
        # Initialize player array [stack, current_bet, is_active, is_all_in]
        players = jnp.array([
            [stacks[i], 0.0, 1.0, 0.0] for i in range(self.config.players)
        ])
        
        # Post blinds
        small_blind_pos = (button_pos + 1) % self.config.players
        big_blind_pos = (button_pos + 2) % self.config.players
        
        # Deduct blinds from stacks and add to bets
        players = players.at[small_blind_pos, 0].subtract(self.config.small_blind)
        players = players.at[small_blind_pos, 1].set(self.config.small_blind)
        players = players.at[big_blind_pos, 0].subtract(self.config.big_blind)
        players = players.at[big_blind_pos, 1].set(self.config.big_blind)
        
        # Initialize deck and deal hole cards
        deck = jnp.arange(52)
        self.key, subkey = jax.random.split(self.key)
        shuffled_deck = jax.random.permutation(subkey, deck)
        
        # Community cards start empty
        community_cards = jnp.array([-1, -1, -1, -1, -1])
        
        # First player to act (after big blind)
        first_player = (big_blind_pos + 1) % self.config.players
        
        return GameState(
            players=players,
            community_cards=community_cards,
            pot_size=self.config.small_blind + self.config.big_blind,
            current_bet=self.config.big_blind,
            phase=0,  # Preflop
            current_player=first_player,
            button_position=button_pos,
            deck=shuffled_deck,
            betting_history=[],
            winner=-1
        )
    
    def get_valid_actions(self, state: GameState, player_id: int) -> List[Action]:
        """
        Get valid actions for a player in the current state.
        
        Args:
            state: Current game state
            player_id: Player ID
            
        Returns:
            List of valid actions
        """
        if not self.is_player_active(state, player_id):
            return []
        
        player_stack = state.players[player_id, 0]
        player_bet = state.players[player_id, 1]
        
        valid_actions = []
        
        # Can always fold (except when checking is free)
        if state.current_bet > player_bet:
            valid_actions.append(Action.FOLD)
        
        # Check if no bet to call
        if state.current_bet == player_bet:
            valid_actions.append(Action.CHECK)
        else:
            # Call if can afford it
            call_amount = state.current_bet - player_bet
            if player_stack >= call_amount:
                valid_actions.append(Action.CALL)
            else:
                valid_actions.append(Action.FOLD)
        
        # Bet/Raise if have chips
        if player_stack > 0:
            if state.current_bet == 0:
                valid_actions.append(Action.BET)
            else:
                valid_actions.append(Action.RAISE)
        
        return valid_actions
    
    def apply_action(self, state: GameState, action: Action) -> GameState:
        """
        Apply an action to the game state.
        
        Args:
            state: Current game state
            action: Action to apply
            
        Returns:
            New game state after action
        """
        player_id = state.current_player
        
        # Update player state based on action
        new_players = state.players
        new_pot = state.pot_size
        new_current_bet = state.current_bet
        new_betting_history = state.betting_history + [action.value]
        
        if action == Action.FOLD:
            # Player folds
            new_players = new_players.at[player_id, 2].set(0.0)  # Mark inactive
            
        elif action == Action.CHECK:
            # Player checks (no cost)
            pass
            
        elif action == Action.CALL:
            # Player calls
            call_amount = state.current_bet - state.players[player_id, 1]
            new_players = new_players.at[player_id, 0].subtract(call_amount)
            new_players = new_players.at[player_id, 1].set(state.current_bet)
            new_pot += call_amount
            
        elif action == Action.BET:
            # Player bets (assume minimum bet)
            bet_amount = self.config.big_blind
            new_players = new_players.at[player_id, 0].subtract(bet_amount)
            new_players = new_players.at[player_id, 1].set(bet_amount)
            new_current_bet = bet_amount
            new_pot += bet_amount
            
        elif action == Action.RAISE:
            # Player raises
            raise_amount = state.current_bet * 2  # Simple 2x raise
            total_bet = raise_amount
            to_call = total_bet - state.players[player_id, 1]
            new_players = new_players.at[player_id, 0].subtract(to_call)
            new_players = new_players.at[player_id, 1].set(total_bet)
            new_current_bet = total_bet
            new_pot += to_call
        
        # Find next active player
        next_player = self._next_active_player(state, player_id)
        
        # Check if betting round is complete
        if self._is_betting_round_complete(state, next_player):
            # Advance to next phase or showdown
            return self._advance_phase(state._replace(
                players=new_players,
                pot_size=new_pot,
                current_bet=new_current_bet,
                betting_history=new_betting_history
            ))
        
        return state._replace(
            players=new_players,
            pot_size=new_pot,
            current_bet=new_current_bet,
            current_player=next_player,
            betting_history=new_betting_history
        )
    
    def _next_active_player(self, state: GameState, current_player: int) -> int:
        """Find next active player"""
        for i in range(1, self.config.players):
            next_player = (current_player + i) % self.config.players
            if self.is_player_active(state, next_player):
                return next_player
        return current_player
    
    def _is_betting_round_complete(self, state: GameState, next_player: int) -> bool:
        """Check if betting round is complete"""
        # Simple check: if we're back to the first player who bet
        active_players = [i for i in range(self.config.players) if self.is_player_active(state, i)]
        return len(active_players) <= 1
    
    def _advance_phase(self, state: GameState) -> GameState:
        """Advance to next phase of the game"""
        new_phase = state.phase + 1
        
        if new_phase >= 4:  # Showdown
            # Determine winner
            winner = self._determine_winner(state)
            return state._replace(phase=new_phase, winner=winner)
        
        # Deal community cards
        cards_to_deal = [3, 1, 1][new_phase - 1]  # Flop: 3, Turn: 1, River: 1
        new_community_cards = state.community_cards
        
        for i in range(cards_to_deal):
            card_idx = new_phase * 3 + i if new_phase == 1 else 3 + (new_phase - 1) + i
            if card_idx < 5:
                new_community_cards = new_community_cards.at[card_idx].set(state.deck[card_idx])
        
        # Reset current bet and find first active player
        first_player = self._first_active_player_after_button(state)
        
        return state._replace(
            phase=new_phase,
            community_cards=new_community_cards,
            current_bet=0.0,
            current_player=first_player,
            betting_history=[]
        )
    
    def _first_active_player_after_button(self, state: GameState) -> int:
        """Find first active player after button"""
        for i in range(1, self.config.players):
            player = (state.button_position + i) % self.config.players
            if self.is_player_active(state, player):
                return player
        return state.button_position
    
    def _determine_winner(self, state: GameState) -> int:
        """Determine winner at showdown"""
        # Simple winner determination (first active player wins)
        for i in range(self.config.players):
            if self.is_player_active(state, i):
                return i
        return 0
    
    def is_player_active(self, state: GameState, player_id: int) -> bool:
        """Check if player is active"""
        return state.players[player_id, 2] > 0
    
    def get_hole_cards(self, state: GameState, player_id: int) -> List[int]:
        """Get hole cards for a player"""
        # Deal from deck (simplified)
        return [state.deck[player_id * 2].item(), state.deck[player_id * 2 + 1].item()]


# Factory function
def create_engine(config: GameConfig = None) -> PokerEngine:
    """Create a poker engine with given configuration"""
    return PokerEngine(config)


# Compatibility aliases
ActionType = Action  # For backward compatibility 