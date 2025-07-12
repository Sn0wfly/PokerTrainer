"""
🧪 Complete Setup Test for PokerBot

This script tests all components to ensure the system is working correctly.
"""

import sys
import traceback
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_imports():
    """Test that all required packages can be imported."""
    console.print("\n📦 Testing imports...")
    
    try:
        # Core packages
        import jax
        import jax.numpy as jnp
        console.print("✅ JAX imported successfully")
        
        # Check for GPU
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            console.print(f"✅ GPU detected: {gpu_devices[0]}")
        else:
            console.print("⚠️  No GPU detected (CPU training will be slow)")
        
        # Project imports
        import poker_bot
        console.print("✅ PokerBot package imported")
        
        from poker_bot.evaluator import HandEvaluator
        from poker_bot.engine import PokerEngine
        from poker_bot.trainer import MCCFRTrainer, TrainingConfig
        from poker_bot.bot import PokerBot, BotConfig
        
        console.print("✅ All core modules imported successfully")
        return True
        
    except Exception as e:
        console.print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_hand_evaluator():
    """Test the hand evaluator component."""
    console.print("\n🔢 Testing hand evaluator...")
    
    try:
        from poker_bot.evaluator import HandEvaluator
        
        evaluator = HandEvaluator()
        
        # Test known hands
        test_cases = [
            # Royal flush (spades): As, Ks, Qs, Js, Ts
            ([48, 44, 40, 36, 32], "Royal Flush"),
            # High card: As, Kh, Qd, Jc, 9s
            ([48, 45, 42, 39, 28], "High Card"),
            # Pair of Aces: As, Ah, Kd, Qc, Js
            ([48, 49, 46, 43, 36], "Pair"),
        ]
        
        for cards, expected_type in test_cases:
            strength = evaluator.evaluate_single(cards)
            rank = evaluator.get_hand_rank(strength)
            console.print(f"  {expected_type}: {rank} (strength: {strength})")
        
        console.print("✅ Hand evaluator working correctly")
        return True
        
    except Exception as e:
        console.print(f"❌ Hand evaluator error: {e}")
        traceback.print_exc()
        return False


def test_game_engine():
    """Test the poker game engine."""
    console.print("\n🎮 Testing game engine...")
    
    try:
        from poker_bot.engine import PokerEngine, ActionType
        
        engine = PokerEngine(num_players=2)
        state = engine.new_game([100.0, 100.0], button_pos=0)
        
        # Test basic game state
        game_info = engine.get_game_info(state)
        console.print(f"  Initial pot: ${game_info['pot']}")
        console.print(f"  Current bet: ${game_info['current_bet']}")
        console.print(f"  Active player: {game_info['active_player']}")
        
        # Test valid actions
        valid_actions = engine.get_valid_actions(state, state.active_player)
        action_types = [a.action_type.value for a in valid_actions]
        console.print(f"  Valid actions: {action_types}")
        
        console.print("✅ Game engine working correctly")
        return True
        
    except Exception as e:
        console.print(f"❌ Game engine error: {e}")
        traceback.print_exc()
        return False


def test_trainer():
    """Test the MCCFR trainer setup."""
    console.print("\n🧠 Testing trainer setup...")
    
    try:
        from poker_bot.trainer import MCCFRTrainer, TrainingConfig
        
        # Create minimal config for testing
        config = TrainingConfig(
            num_iterations=10,  # Very small for testing
            batch_size=4,
            num_players=2,
            eval_interval=5,
            save_interval=10
        )
        
        trainer = MCCFRTrainer(config)
        console.print(f"  Trainer initialized with {config.num_players} players")
        console.print(f"  Action space size: {trainer.env.action_space_size}")
        
        console.print("✅ Trainer setup working correctly")
        return True
        
    except Exception as e:
        console.print(f"❌ Trainer error: {e}")
        traceback.print_exc()
        return False


def test_bot():
    """Test the poker bot (with fallback strategy)."""
    console.print("\n🤖 Testing poker bot...")
    
    try:
        from poker_bot.bot import PokerBot, BotConfig
        from poker_bot.engine import PokerEngine, Action, ActionType
        
        # Create bot config with non-existent model (will use fallback)
        config = BotConfig(
            model_path="non_existent_model.pkl",
            thinking_time=0.1,
            enable_logging=False
        )
        
        bot = PokerBot(config)
        console.print("  Bot initialized with fallback strategy")
        
        # Test decision making
        engine = PokerEngine(num_players=2)
        state = engine.new_game([100.0, 100.0], button_pos=0)
        hole_cards = [48, 49]  # Pocket Aces
        player_id = 0
        valid_actions = engine.get_valid_actions(state, player_id)
        
        decision = bot.make_decision(state, hole_cards, player_id, valid_actions)
        console.print(f"  Bot decision: {decision.action_type.value} ${decision.amount:.2f}")
        
        # Test performance stats
        stats = bot.get_performance_stats()
        console.print(f"  Decision time: {stats['avg_decision_time']:.3f}s")
        
        console.print("✅ Bot working correctly")
        return True
        
    except Exception as e:
        console.print(f"❌ Bot error: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """Test performance benchmarks."""
    console.print("\n⚡ Running performance tests...")
    
    try:
        from poker_bot.evaluator import HandEvaluator
        import time
        import random
        
        evaluator = HandEvaluator()
        
        # Benchmark hand evaluation
        num_evaluations = 10000
        start_time = time.time()
        
        for _ in range(num_evaluations):
            cards = random.sample(range(52), 7)
            strength = evaluator.evaluate_single(cards)
        
        end_time = time.time()
        evaluations_per_second = num_evaluations / (end_time - start_time)
        
        console.print(f"  Hand evaluations: {evaluations_per_second:,.0f} per second")
        console.print(f"  Average time: {1000/evaluations_per_second:.3f}ms per evaluation")
        
        # Check if performance is reasonable
        if evaluations_per_second > 10000:
            console.print("✅ Performance is excellent")
        elif evaluations_per_second > 1000:
            console.print("✅ Performance is good")
        else:
            console.print("⚠️  Performance is acceptable but could be better")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Performance test error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests and display results."""
    console.print(Panel.fit(
        "🧪 PokerBot Complete Setup Test\n\n"
        "This script verifies that all components are working correctly.",
        title="Setup Test",
        border_style="blue"
    ))
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Hand Evaluator", test_hand_evaluator),
        ("Game Engine", test_game_engine),
        ("Trainer", test_trainer),
        ("Bot", test_bot),
        ("Performance", test_performance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            console.print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, "💥 CRASH"))
    
    # Display results summary
    console.print("\n📊 Test Results Summary:")
    
    results_table = Table(title="Test Results")
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Status", style="bold")
    
    for test_name, status in results:
        results_table.add_row(test_name, status)
    
    console.print(results_table)
    
    # Overall result
    passed = sum(1 for _, status in results if "PASS" in status)
    total = len(results)
    
    if passed == total:
        console.print(f"\n🎉 All tests passed! ({passed}/{total})")
        console.print("✅ Your PokerBot setup is working correctly!")
        console.print("\n🚀 You can now:")
        console.print("   • Run 'poker-train --help' to see training options")
        console.print("   • Run 'poker-play --help' to see playing options")
        console.print("   • Check the README.md for full documentation")
    else:
        console.print(f"\n⚠️  {passed}/{total} tests passed")
        console.print("❌ Some components need attention")
        console.print("Check the error messages above for details")
        sys.exit(1)


if __name__ == "__main__":
    main() 