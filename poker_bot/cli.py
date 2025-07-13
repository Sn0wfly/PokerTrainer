"""
Command Line Interface for PokerTrainer
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from .trainer import create_trainer, MCCFRConfig
from .bot import PokerBot
from .engine import PokerEngine, GameConfig
from .evaluator import HandEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option()
def cli():
    """PokerTrainer - GPU-accelerated poker AI training and playing"""
    pass

@cli.command()
@click.option('--iterations', default=100000, help='Number of training iterations')
@click.option('--batch-size', default=1024, help='Batch size for training')
@click.option('--players', default=2, help='Number of players')
@click.option('--learning-rate', default=0.1, help='Learning rate')
@click.option('--exploration', default=0.1, help='Exploration rate')
@click.option('--save-interval', default=1000, help='Save model every N iterations')
@click.option('--log-interval', default=100, help='Log progress every N iterations')
@click.option('--save-path', default='models/mccfr_model.pkl', help='Path to save trained model')
@click.option('--config-file', help='YAML configuration file')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration (if available)')
def train(iterations: int, batch_size: int, players: int, learning_rate: float,
          exploration: float, save_interval: int, log_interval: int,
          save_path: str, config_file: Optional[str], gpu: bool):
    """Train poker AI using MCCFR algorithm"""
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load configuration from file if provided
    if config_file:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Override with command line arguments
        config_data.update({
            'iterations': iterations,
            'batch_size': batch_size,
            'players': players,
            'learning_rate': learning_rate,
            'exploration': exploration,
            'save_interval': save_interval,
            'log_interval': log_interval
        })
    else:
        config_data = {
            'iterations': iterations,
            'batch_size': batch_size,
            'players': players,
            'learning_rate': learning_rate,
            'exploration': exploration,
            'save_interval': save_interval,
            'log_interval': log_interval
        }
    
    # Check GPU availability
    if gpu:
        try:
            import jax
            devices = jax.devices()
            gpu_available = len([d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]) > 0
            
            if gpu_available:
                logger.info(f"GPU acceleration enabled. Devices: {devices}")
            else:
                logger.warning("GPU requested but not available. Using CPU.")
        except Exception as e:
            logger.warning(f"GPU check failed: {e}. Using CPU.")
    else:
        logger.info("Using CPU training (--no-gpu specified)")
    
    # Create trainer
    trainer = create_trainer(**config_data)
    
    # Start training
    logger.info("Starting MCCFR training...")
    logger.info(f"Configuration: {config_data}")
    
    try:
        trainer.train(save_path=save_path)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model', required=True, help='Path to trained model')
@click.option('--hands', default=100, help='Number of hands to play')
@click.option('--opponents', default=1, help='Number of opponents')
@click.option('--stack', default=100.0, help='Starting stack size')
@click.option('--aggressive/--conservative', default=False, help='Play aggressively')
@click.option('--thinking-time', default=1.0, help='Thinking time in seconds')
@click.option('--log-file', help='Log game to file')
def play(model: str, hands: int, opponents: int, stack: float, 
         aggressive: bool, thinking_time: float, log_file: Optional[str]):
    """Play poker using trained AI model"""
    
    if not os.path.exists(model):
        logger.error(f"Model file not found: {model}")
        sys.exit(1)
    
    # Setup logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    # Create bot configuration
    config = GameConfig(
        players=opponents + 1,
        starting_stack=stack,
        big_blind=2.0,
        small_blind=1.0
    )
    
    # Load trained model and create bot
    try:
        bot = PokerBot(model_path=model, config=config)
        logger.info(f"Loaded model from {model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Play games
    logger.info(f"Starting {hands} hands against {opponents} opponents")
    logger.info(f"Stack: ${stack}, Aggressive: {aggressive}")
    
    try:
        results = bot.play_session(
            hands=hands,
            thinking_time=thinking_time,
            aggressive=aggressive
        )
        
        # Display results
        logger.info("Session completed!")
        logger.info(f"Hands played: {results.get('hands_played', 0)}")
        logger.info(f"Hands won: {results.get('hands_won', 0)}")
        logger.info(f"Final stack: ${results.get('final_stack', 0):.2f}")
        logger.info(f"Profit/Loss: ${results.get('profit_loss', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Playing session failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model', help='Path to trained model to evaluate')
def evaluate(model: Optional[str]):
    """Evaluate poker bot components"""
    
    logger.info("Evaluating PokerTrainer components...")
    
    # Test hand evaluator
    try:
        evaluator = HandEvaluator()
        test_cards = [0, 1, 2, 3, 4, 5, 6]  # Card indices instead of strings
        result = evaluator.evaluate_single(test_cards)
        logger.info(f"✅ Hand evaluator working: {result}")
    except Exception as e:
        logger.error(f"❌ Hand evaluator failed: {e}")
        return
    
    # Test poker engine
    try:
        engine = PokerEngine()
        game_state = engine.new_game()
        logger.info("✅ Poker engine working")
    except Exception as e:
        logger.error(f"❌ Poker engine failed: {e}")
        return
    
    # Test JAX
    try:
        import jax
        import jax.numpy as jnp
        
        logger.info(f"✅ JAX version: {jax.__version__}")
        logger.info(f"✅ JAX devices: {jax.devices()}")
        
        # Test computation
        x = jnp.array([1., 2., 3.])
        result = jnp.sum(x)
        logger.info(f"✅ JAX computation working: {result}")
        
    except Exception as e:
        logger.error(f"❌ JAX failed: {e}")
        return
    
    # Test trainer
    try:
        trainer = create_trainer(iterations=10, batch_size=4, players=2)
        logger.info("✅ MCCFR trainer created successfully")
    except Exception as e:
        logger.error(f"❌ MCCFR trainer failed: {e}")
        return
    
    # Test model loading if provided
    if model:
        if os.path.exists(model):
            try:
                config = GameConfig()
                bot = PokerBot(model_path=model, config=config)
                logger.info(f"✅ Model loaded successfully: {model}")
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                return
        else:
            logger.warning(f"Model file not found: {model}")
    
    logger.info("🎉 All components working!")

@cli.command()
@click.option('--iterations', default=10000, help='Number of test iterations')
@click.option('--batch-size', default=512, help='Batch size for testing')
@click.option('--temperature', default=1.0, help='Temperature for strategy computation')
@click.option('--learning-rate', default=0.1, help='Learning rate for Q-value updates')
def test_modern(iterations: int, batch_size: int, temperature: float, learning_rate: float):
    """Test the modern CFVFP architecture"""
    
    try:
        # Import modern components
        from .modern_cfr import create_cfvfp_trainer, CFVFPConfig, InfoState
        from .gpu_config import init_gpu_environment, get_device_info
        from .memory import MemoryMonitor, log_memory_usage
        from .evaluator import HandEvaluator
        import jax.numpy as jnp
        import jax.random as jr
        import time
        
        logger.info("🚀 Testing Modern CFVFP Architecture")
        logger.info("=" * 50)
        
        # Initialize GPU environment
        logger.info("Initializing GPU environment...")
        env_info = init_gpu_environment()
        device_info = get_device_info()
        
        logger.info(f"✅ GPU Environment initialized")
        logger.info(f"   Platform: {device_info['platform']}")
        logger.info(f"   Devices: {device_info['num_devices']}")
        logger.info(f"   Local devices: {device_info['local_devices']}")
        
        # Test hand evaluator
        logger.info("\nTesting hand evaluator...")
        evaluator = HandEvaluator()
        test_hand = [2, 3, 4, 5, 6]  # Straight
        strength = evaluator.evaluate_single(test_hand)
        logger.info(f"✅ Hand evaluator working: {strength}")
        
        # Test memory monitoring
        logger.info("\nTesting memory monitoring...")
        with MemoryMonitor("Modern CFR Test") as monitor:
            log_memory_usage("Initial: ")
            
            # Create CFVFP trainer
            logger.info("Creating CFVFP trainer...")
            config = CFVFPConfig(
                iterations=iterations,
                batch_size=batch_size,
                temperature=temperature,
                learning_rate=learning_rate
            )
            trainer = create_cfvfp_trainer(config)
            logger.info(f"✅ CFVFP trainer created with config: {config}")
            
            # Test JAX operations
            logger.info("\nTesting JAX operations...")
            key = jr.PRNGKey(42)
            
            # Test Q-value updates
            test_q_values = jnp.array([0.1, 0.2, 0.3, 0.4])
            test_action_values = jnp.array([0.15, 0.25, 0.35, 0.45])
            
            start_time = time.time()
            updated_q = trainer._update_q_values(test_q_values, test_action_values, 0.1)
            compile_time = time.time() - start_time
            logger.info(f"✅ Q-value update (first call/compile): {compile_time:.3f}s")
            
            # Test strategy computation
            start_time = time.time()
            strategy = trainer._compute_strategy(updated_q, 1.0)
            compute_time = time.time() - start_time
            logger.info(f"✅ Strategy computation: {compute_time:.6f}s")
            logger.info(f"   Strategy: {strategy}")
            
            # Test action selection
            start_time = time.time()
            action = trainer._select_action(strategy, key)
            select_time = time.time() - start_time
            logger.info(f"✅ Action selection: {select_time:.6f}s")
            logger.info(f"   Selected action: {action}")
            
            # Test batch operations
            logger.info("\nTesting batch operations...")
            from .modern_cfr import batch_update_q_values, batch_compute_strategies
            
            batch_q = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            batch_values = jnp.array([[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]])
            
            start_time = time.time()
            batch_updated = batch_update_q_values(batch_q, batch_values, 0.1)
            batch_time = time.time() - start_time
            logger.info(f"✅ Batch Q-value update: {batch_time:.6f}s")
            
            start_time = time.time()
            batch_strategies = batch_compute_strategies(batch_updated, 1.0)
            batch_strategy_time = time.time() - start_time
            logger.info(f"✅ Batch strategy computation: {batch_strategy_time:.6f}s")
            
            # Test info state handling
            logger.info("\nTesting info state handling...")
            test_info_state = InfoState(
                player_id=0,
                cards=jnp.array([0, 1]),
                history=jnp.array([0, 1, 2]),
                pot=100.0,
                round=1
            )
            
            strategy = trainer.get_strategy(test_info_state, 4)
            logger.info(f"✅ Info state strategy: {strategy}")
            
            # Update info state
            test_action_values = jnp.array([0.1, 0.3, 0.2, 0.4])
            updated_strategy = trainer.update_info_state(test_info_state, test_action_values, 4)
            logger.info(f"✅ Updated strategy: {updated_strategy}")
            
            # Test training stats
            stats = trainer.get_training_stats()
            logger.info(f"✅ Training stats: {stats}")
            
            monitor.step()
        
        logger.info("\n🎉 All Modern CFR tests passed!")
        logger.info("=" * 50)
        logger.info("✅ GPU environment: Working")
        logger.info("✅ Memory management: Working")
        logger.info("✅ CFVFP trainer: Working")
        logger.info("✅ JAX operations: Working")
        logger.info("✅ Batch processing: Working")
        logger.info("✅ Info state handling: Working")
        logger.info("\n🚀 Ready for Phase 2 - Performance Optimization!")
        
    except Exception as e:
        logger.error(f"❌ Modern CFR test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command()
def list_models():
    """List available trained models"""
    
    models_dir = Path("models")
    if not models_dir.exists():
        logger.info("No models directory found")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        logger.info("No trained models found")
        return
    
    logger.info("Available models:")
    for model_file in model_files:
        size = model_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  {model_file.name} ({size:.1f} MB)")

if __name__ == '__main__':
    cli() 