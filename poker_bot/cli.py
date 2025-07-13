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
        test_cards = ['2s', '2h', '2d', '2c', '3s', '3h', '3d']
        result = evaluator.evaluate_hand(test_cards)
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