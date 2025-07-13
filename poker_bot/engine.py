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
    players: int = 6  # Default to 6-max (most popular online format)
    starting_stack: float = 100.0
    small_blind: float = 1.0
    big_blind: float = 2.0
    max_raises: int = -1  # No limit on raises (NLHE)


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
        # Game is terminal if we have a winner
        if self.winner != -1:
            return True
        
        # Game is terminal if we've reached showdown phase (phase 4+)
        if self.phase >= 4:
            return True
            
        # Game is terminal if only one or zero players are active
        active_count = 0
        for i in range(len(self.players)):
            if self.players[i, 2] > 0:  # is_active flag
                active_count += 1
        
        if active_count <= 1:
            return True
            
        return False


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
        players = players.at[small_blind_pos, 0].add(-self.config.small_blind)
        players = players.at[small_blind_pos, 1].set(self.config.small_blind)
        players = players.at[big_blind_pos, 0].add(-self.config.big_blind)
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
        
        # Bet/Raise if have chips (No Limit rules)
        if player_stack > 0:
            if state.current_bet == 0:
                valid_actions.append(Action.BET)
            else:
                # Can raise if have more than call amount
                call_amount = state.current_bet - player_bet
                if player_stack > call_amount:
                    valid_actions.append(Action.RAISE)
            
            # Can always go all-in if have chips (No Limit)
            valid_actions.append(Action.ALL_IN)
        
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
            new_players = new_players.at[player_id, 0].add(-call_amount)
            new_players = new_players.at[player_id, 1].set(state.current_bet)
            new_pot += call_amount
            
        elif action == Action.BET:
            # Player bets (No Limit - can bet any amount up to stack)
            player_stack = state.players[player_id, 0]
            # For simplicity, bet pot-sized (can be optimized later)
            bet_amount = min(state.pot_size, player_stack)
            if bet_amount < self.config.big_blind:
                bet_amount = min(self.config.big_blind, player_stack)
            
            new_players = new_players.at[player_id, 0].add(-bet_amount)
            new_players = new_players.at[player_id, 1].set(bet_amount)
            new_current_bet = bet_amount
            new_pot += bet_amount
            
        elif action == Action.RAISE:
            # Player raises (No Limit - can raise any amount)
            player_stack = state.players[player_id, 0]
            current_player_bet = state.players[player_id, 1]
            
            # Minimum raise = current bet + previous raise amount
            min_raise = state.current_bet * 2
            
            # For AI simplicity, raise pot-sized (can be optimized)
            raise_to = min(min_raise + state.pot_size, state.current_bet + player_stack)
            
            to_call = raise_to - current_player_bet
            new_players = new_players.at[player_id, 0].add(-to_call)
            new_players = new_players.at[player_id, 1].set(raise_to)
            new_current_bet = raise_to
            new_pot += to_call
            
        elif action == Action.ALL_IN:
            # Player goes all-in (No Limit)
            player_stack = state.players[player_id, 0]
            current_player_bet = state.players[player_id, 1]
            
            # Bet entire remaining stack
            all_in_amount = current_player_bet + player_stack
            new_players = new_players.at[player_id, 0].set(0.0)  # Stack = 0
            new_players = new_players.at[player_id, 1].set(all_in_amount)
            new_players = new_players.at[player_id, 3].set(1.0)  # Mark as all-in
            
            if all_in_amount > state.current_bet:
                new_current_bet = all_in_amount
            
            new_pot += player_stack
        
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
        active_players = [i for i in range(self.config.players) if self.is_player_active(state, i)]
        
        # If only one or zero players active, round is complete
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have matched the current bet
        all_bets_equal = True
        max_bet = 0.0
        
        for player_id in active_players:
            player_bet = state.players[player_id, 1]
            max_bet = max(max_bet, player_bet)
        
        for player_id in active_players:
            player_bet = state.players[player_id, 1]
            player_stack = state.players[player_id, 0]
            is_all_in = state.players[player_id, 3] > 0
            
            # Player must either match max bet or be all-in
            if not is_all_in and player_bet < max_bet and player_stack > 0:
                all_bets_equal = False
                break
        
        # If all bets are equal and everyone has had a chance to act, round is complete
        if all_bets_equal:
            # Additional check: if everyone just checked and there are no more actions needed
            if state.current_bet == 0 and len(state.betting_history) >= len(active_players):
                return True
            # Or if everyone has called/matched the current bet
            elif state.current_bet > 0 and all_bets_equal:
                return True
        
        # Special case: if we've had too many consecutive checks, force end round
        recent_actions = state.betting_history[-10:]
        consecutive_checks = 0
        for action in reversed(recent_actions):
            if action == 'check':
                consecutive_checks += 1
            else:
                break
        
        # If we have more checks than active players, something is wrong - end round
        if consecutive_checks >= len(active_players) * 2:
            return True
        
        return False
    
    def _advance_phase(self, state: GameState) -> GameState:
        """Advance to next phase of the game"""
        new_phase = state.phase + 1
        
        # Check for showdown or game end
        active_players = [i for i in range(self.config.players) if self.is_player_active(state, i)]
        
        if len(active_players) <= 1:
            # Only one player left - they win
            winner = active_players[0] if active_players else 0
            return state._replace(phase=4, winner=winner)  # Game over
        
        if new_phase >= 4:  # Showdown phase
            # Determine winner at showdown
            winner = self._determine_winner(state)
            return state._replace(phase=4, winner=winner)
        
        # Deal community cards for new phase
        new_community_cards = state.community_cards
        
        if new_phase == 1:  # Flop - deal 3 cards
            for i in range(3):
                if i < len(state.deck):
                    new_community_cards = new_community_cards.at[i].set(state.deck[i])
        elif new_phase == 2:  # Turn - deal 1 card
            if 3 < len(state.deck):
                new_community_cards = new_community_cards.at[3].set(state.deck[3])
        elif new_phase == 3:  # River - deal 1 card
            if 4 < len(state.deck):
                new_community_cards = new_community_cards.at[4].set(state.deck[4])
        
        # Reset betting for new phase
        # Clear all player bets (keep them in pot)
        new_players = state.players
        for i in range(self.config.players):
            if self.is_player_active(state, i):
                new_players = new_players.at[i, 1].set(0.0)  # Reset current bet to 0
        
        # Find first active player after button for new betting round
        first_player = self._first_active_player_after_button(state)
        
        return state._replace(
            phase=new_phase,
            community_cards=new_community_cards,
            players=new_players,
            current_bet=0.0,  # Reset current bet for new round
            current_player=first_player,
            betting_history=[]  # Clear betting history for new round
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
        active_players = [i for i in range(self.config.players) if self.is_player_active(state, i)]
        
        if not active_players:
            return 0  # Default to player 0 if no active players
        
        if len(active_players) == 1:
            return active_players[0]  # Last player standing wins
        
        # For simplicity in training, use a deterministic winner based on player position
        # In a real implementation, this would evaluate hand strength
        # For now, use the player with the lowest ID among active players
        return min(active_players)
    
    def is_player_active(self, state: GameState, player_id: int) -> bool:
        """Check if player is active"""
        return state.players[player_id, 2] > 0
    
    def get_hole_cards(self, state: GameState, player_id: int) -> List[int]:
        """Get hole cards for a player"""
        # Deal from deck (simplified)
        return [state.deck[player_id * 2].item(), state.deck[player_id * 2 + 1].item()]
    
    def get_information_set(self, state: GameState, player_id: int) -> str:
        """
        Get information set string for CFR training
        
        Args:
            state: Current game state
            player_id: Player ID
            
        Returns:
            Information set string for this player's perspective
        """
        # Get player's hole cards
        hole_cards = self.get_hole_cards(state, player_id)
        
        # Get visible community cards based on phase
        visible_community = []
        if state.phase >= 1:  # Flop
            visible_community = state.community_cards[:3].tolist()
        if state.phase >= 2:  # Turn
            visible_community = state.community_cards[:4].tolist()
        if state.phase >= 3:  # River
            visible_community = state.community_cards[:5].tolist()
        
        # Get betting action history for this round
        recent_history = state.betting_history[-10:]  # Last 10 actions
        
        # Get position info
        position = (player_id - state.button_position) % self.config.players
        
        # Create compact info set string
        info_set = (f"pos{position}_phase{state.phase}_"
                   f"hole{'-'.join(map(str, hole_cards))}_"
                   f"board{'-'.join(map(str, visible_community))}_"
                   f"pot{int(state.pot_size)}_"
                   f"bet{int(state.current_bet)}_"
                   f"hist{''.join(recent_history[:5])}")  # Last 5 actions
        
        return info_set


# Factory function
def create_engine(config: GameConfig = None) -> PokerEngine:
    """Create a poker engine with given configuration"""
    return PokerEngine(config)


# Compatibility aliases
ActionType = Action  # For backward compatibility 