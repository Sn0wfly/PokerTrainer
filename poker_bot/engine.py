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
from .evaluator import HandEvaluator


class ActionType(Enum):
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
class Action:
    """Represents a poker action."""
    action_type: ActionType
    amount: float = 0.0
    player_id: int = 0


@dataclass
class Player:
    """Represents a poker player."""
    player_id: int
    stack: float
    hole_cards: List[int]
    is_active: bool = True
    is_all_in: bool = False
    current_bet: float = 0.0
    has_acted: bool = False


class GameState(NamedTuple):
    """Immutable game state for JAX compatibility."""
    players: jnp.ndarray  # Player stacks and bets
    community_cards: jnp.ndarray  # Board cards
    pot: float
    current_bet: float
    phase: int  # GamePhase as integer
    active_player: int
    button_position: int
    deck: jnp.ndarray  # Remaining cards in deck


class PokerEngine:
    """
    High-performance Texas Hold'em poker engine.
    
    Handles game state, action processing, and rule enforcement.
    """
    
    def __init__(self, num_players: int = 6, small_blind: float = 1.0, big_blind: float = 2.0):
        """
        Initialize the poker engine.
        
        Args:
            num_players: Number of players (2-9)
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = HandEvaluator()
        
        # Initialize deck (52 cards)
        self.full_deck = jnp.arange(52)
        
        # Action abstraction (geometric bet sizes)
        self.bet_sizes = [0.5, 0.75, 1.0, 1.5, 2.0]  # Pot fractions
        
        # JIT compile core functions
        self.process_action_jit = jax.jit(self._process_action_impl)
        self.evaluate_showdown_jit = jax.jit(self._evaluate_showdown_impl)
    
    def new_game(self, stacks: List[float], button_pos: int = 0) -> GameState:
        """
        Start a new poker game.
        
        Args:
            stacks: Initial stack sizes for each player
            button_pos: Position of dealer button
            
        Returns:
            Initial game state
        """
        # Initialize players array: [stack, current_bet, is_active, is_all_in]
        players = jnp.zeros((self.num_players, 4))
        players = players.at[:, 0].set(jnp.array(stacks))  # Set stacks
        players = players.at[:, 2].set(1.0)  # Set all active
        
        # Deal hole cards (2 per player)
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
        deck = jax.random.permutation(key, self.full_deck)
        
        # Community cards (empty initially)
        community_cards = jnp.full(5, -1)
        
        # Post blinds
        sb_pos = (button_pos + 1) % self.num_players
        bb_pos = (button_pos + 2) % self.num_players
        
        players = players.at[sb_pos, 0].add(-self.small_blind)  # Subtract from stack
        players = players.at[sb_pos, 1].set(self.small_blind)   # Set current bet
        players = players.at[bb_pos, 0].add(-self.big_blind)
        players = players.at[bb_pos, 1].set(self.big_blind)
        
        return GameState(
            players=players,
            community_cards=community_cards,
            pot=self.small_blind + self.big_blind,
            current_bet=self.big_blind,
            phase=GamePhase.PREFLOP.value,
            active_player=(bb_pos + 1) % self.num_players,
            button_position=button_pos,
            deck=deck[self.num_players * 2:]  # Remove dealt cards
        )
    
    def get_valid_actions(self, state: GameState, player_id: int) -> List[Action]:
        """
        Get valid actions for a player.
        
        Args:
            state: Current game state
            player_id: Player to get actions for
            
        Returns:
            List of valid actions
        """
        actions = []
        player_stack = state.players[player_id, 0]
        player_bet = state.players[player_id, 1]
        to_call = state.current_bet - player_bet
        
        # Always can fold (except when can check)
        if to_call > 0:
            actions.append(Action(ActionType.FOLD, 0.0, player_id))
        
        # Can check if no bet to call
        if to_call == 0:
            actions.append(Action(ActionType.CHECK, 0.0, player_id))
        
        # Can call if there's a bet and player has chips
        if to_call > 0 and player_stack >= to_call:
            actions.append(Action(ActionType.CALL, to_call, player_id))
        
        # Can bet/raise if player has chips
        if player_stack > to_call:
            remaining_stack = player_stack - to_call
            
            # Bet/raise sizes based on pot
            for size in self.bet_sizes:
                bet_amount = state.pot * size
                if remaining_stack >= bet_amount:
                    action_type = ActionType.BET if to_call == 0 else ActionType.RAISE
                    actions.append(Action(action_type, bet_amount + to_call, player_id))
        
        # Can always go all-in
        if player_stack > 0:
            actions.append(Action(ActionType.ALL_IN, player_stack, player_id))
        
        return actions
    
    def process_action(self, state: GameState, action: Action) -> GameState:
        """
        Process a player action and return new game state.
        
        Args:
            state: Current game state
            action: Action to process
            
        Returns:
            New game state after action
        """
        return self.process_action_jit(state, action)
    
    def _process_action_impl(self, state: GameState, action: Action) -> GameState:
        """JAX-compiled action processing implementation."""
        player_id = action.player_id
        players = state.players
        
        if action.action_type == ActionType.FOLD:
            # Mark player as inactive
            players = players.at[player_id, 2].set(0.0)
        
        elif action.action_type == ActionType.CHECK:
            # No chips change, just mark as acted
            pass
        
        elif action.action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE]:
            # Add chips to pot and current bet
            bet_amount = action.amount
            players = players.at[player_id, 0].add(-bet_amount)  # Subtract from stack
            players = players.at[player_id, 1].add(bet_amount)   # Add to current bet
            
            # Update pot and current bet
            new_pot = state.pot + bet_amount
            new_current_bet = jnp.maximum(state.current_bet, players[player_id, 1])
            
            return state._replace(
                players=players,
                pot=new_pot,
                current_bet=new_current_bet,
                active_player=self._next_active_player(state, player_id)
            )
        
        elif action.action_type == ActionType.ALL_IN:
            # Player goes all-in
            all_in_amount = players[player_id, 0]
            players = players.at[player_id, 0].set(0.0)  # Stack becomes 0
            players = players.at[player_id, 1].add(all_in_amount)  # Add to current bet
            players = players.at[player_id, 3].set(1.0)  # Mark as all-in
            
            new_pot = state.pot + all_in_amount
            new_current_bet = jnp.maximum(state.current_bet, players[player_id, 1])
            
            return state._replace(
                players=players,
                pot=new_pot,
                current_bet=new_current_bet,
                active_player=self._next_active_player(state, player_id)
            )
        
        # Default: just move to next player
        return state._replace(
            players=players,
            active_player=self._next_active_player(state, player_id)
        )
    
    def _next_active_player(self, state: GameState, current_player: int) -> int:
        """Find next active player."""
        for i in range(1, self.num_players):
            next_player = (current_player + i) % self.num_players
            if state.players[next_player, 2] == 1.0:  # is_active
                return next_player
        return current_player  # No active players found
    
    def advance_phase(self, state: GameState) -> GameState:
        """
        Advance to next phase of the game.
        
        Args:
            state: Current game state
            
        Returns:
            Game state in next phase
        """
        current_phase = state.phase
        community_cards = state.community_cards
        
        if current_phase == GamePhase.PREFLOP.value:
            # Deal flop (3 cards)
            new_cards = state.deck[:3]
            community_cards = community_cards.at[:3].set(new_cards)
            new_deck = state.deck[3:]
            new_phase = GamePhase.FLOP.value
        
        elif current_phase == GamePhase.FLOP.value:
            # Deal turn (1 card)
            new_cards = state.deck[:1]
            community_cards = community_cards.at[3].set(new_cards[0])
            new_deck = state.deck[1:]
            new_phase = GamePhase.TURN.value
        
        elif current_phase == GamePhase.TURN.value:
            # Deal river (1 card)
            new_cards = state.deck[:1]
            community_cards = community_cards.at[4].set(new_cards[0])
            new_deck = state.deck[1:]
            new_phase = GamePhase.RIVER.value
        
        else:
            # Already at river or showdown
            new_deck = state.deck
            new_phase = GamePhase.SHOWDOWN.value
        
        # Reset betting round
        players = state.players.at[:, 1].set(0.0)  # Reset current bets
        
        return state._replace(
            community_cards=community_cards,
            deck=new_deck,
            phase=new_phase,
            players=players,
            current_bet=0.0,
            active_player=self._first_active_player_after_button(state)
        )
    
    def _first_active_player_after_button(self, state: GameState) -> int:
        """Find first active player after button for new betting round."""
        for i in range(1, self.num_players):
            player = (state.button_position + i) % self.num_players
            if state.players[player, 2] == 1.0:  # is_active
                return player
        return state.button_position
    
    def evaluate_showdown(self, state: GameState, hole_cards: jnp.ndarray) -> Tuple[List[int], List[float]]:
        """
        Evaluate showdown and determine winners.
        
        Args:
            state: Game state at showdown
            hole_cards: Hole cards for each player (num_players, 2)
            
        Returns:
            (winners, payouts) - List of winning player IDs and their payouts
        """
        active_players = []
        hand_strengths = []
        
        # Evaluate each active player's hand
        for player_id in range(self.num_players):
            if state.players[player_id, 2] == 1.0:  # is_active
                # Combine hole cards with community cards
                player_cards = jnp.concatenate([
                    hole_cards[player_id],
                    state.community_cards
                ])
                
                # Evaluate 7-card hand
                strength = self.evaluator.evaluate_single(player_cards.tolist())
                active_players.append(player_id)
                hand_strengths.append(strength)
        
        # Find winner(s) (lowest strength wins in phevaluator)
        best_strength = min(hand_strengths)
        winners = [active_players[i] for i, strength in enumerate(hand_strengths) 
                  if strength == best_strength]
        
        # Calculate payouts
        payout_per_winner = state.pot / len(winners)
        payouts = [payout_per_winner if i in winners else 0.0 
                  for i in range(self.num_players)]
        
        return winners, payouts
    
    def is_game_over(self, state: GameState) -> bool:
        """Check if game is over."""
        active_players = jnp.sum(state.players[:, 2])
        return active_players <= 1 or state.phase == GamePhase.SHOWDOWN.value
    
    def get_game_info(self, state: GameState) -> Dict:
        """Get human-readable game information."""
        return {
            "pot": float(state.pot),
            "current_bet": float(state.current_bet),
            "phase": GamePhase(state.phase).name,
            "active_player": int(state.active_player),
            "community_cards": state.community_cards.tolist(),
            "num_active_players": int(jnp.sum(state.players[:, 2])),
        }


def test_engine():
    """Test the poker engine."""
    engine = PokerEngine(num_players=3, small_blind=1.0, big_blind=2.0)
    
    # Start new game
    stacks = [100.0, 100.0, 100.0]
    state = engine.new_game(stacks, button_pos=0)
    
    print("Initial game state:")
    print(engine.get_game_info(state))
    
    # Test valid actions
    player_id = state.active_player
    actions = engine.get_valid_actions(state, player_id)
    print(f"Valid actions for player {player_id}: {[a.action_type.value for a in actions]}")
    
    # Process a call action
    call_action = Action(ActionType.CALL, 2.0, player_id)
    new_state = engine.process_action(state, call_action)
    
    print("After call:")
    print(engine.get_game_info(new_state))
    
    return engine, new_state


if __name__ == "__main__":
    test_engine() 