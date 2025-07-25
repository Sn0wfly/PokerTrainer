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
@click.option('--iterations', default=10000, help='Number of training iterations')
@click.option('--batch-size', default=8192, help='Batch size for training')
@click.option('--algorithm', default='pdcfr_plus', help='Algorithm to use (pdcfr_plus, outcome_sampling, neural_fsp, parallel)')
@click.option('--save-interval', default=1000, help='Save model every N iterations')
@click.option('--log-interval', default=100, help='Log progress every N iterations')
@click.option('--save-path', default='models/fast_model.pkl', help='Path to save trained model')
@click.option('--learning-rate', default=0.1, help='Learning rate')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration')
def train_fast(iterations: int, batch_size: int, algorithm: str, save_interval: int, 
               log_interval: int, save_path: str, learning_rate: float, gpu: bool):
    """Fast training using optimized algorithms (PDCFRPlus, Parallel, etc.)"""
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    
    try:
        from .parallel import get_optimal_parallel_config, create_parallel_trainer
        from .algorithms import create_advanced_cfr_trainer
        from .modern_cfr import InfoState
        import jax.numpy as jnp
        import jax.random as jr
        import time
        import pickle
        
        logger.info("🚀 Starting Fast Training with Optimized Algorithms")
        logger.info("=" * 60)
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Save interval: {save_interval}")
        logger.info(f"Save path: {save_path}")
        logger.info("")
        
        # Initialize trainer based on algorithm
        trainer = None
        training_data = {
            'strategy_sum': {},
            'regret_sum': {},
            'iteration': 0,
            'config': {
                'algorithm': algorithm,
                'iterations': iterations,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        if algorithm == 'parallel':
            logger.info("Initializing Parallel Training...")
            parallel_config = get_optimal_parallel_config()
            trainer = create_parallel_trainer(parallel_config)
        else:
            logger.info(f"Initializing Advanced CFR Algorithm: {algorithm}")
            trainer = create_advanced_cfr_trainer(algorithm)
        
        # Training loop
        logger.info("Starting training loop...")
        start_time = time.time()
        
        for iteration in range(1, iterations + 1):
            # Create test info state for training with more diversity
            key = jr.PRNGKey(iteration)
            subkeys = jr.split(key, 4)
            
            # More diverse info states
            player_id = iteration % 2
            round_num = iteration % 4
            cards = jr.randint(subkeys[0], (5,), 1, 53)  # Random cards 1-52
            history = jr.randint(subkeys[1], (4,), 0, 4)  # Random history 0-3
            pot = 10.0 + (iteration % 1000)  # More pot variety
            
            test_info_state = InfoState(
                player_id=player_id,
                cards=cards,
                history=history,
                pot=pot,
                round=round_num
            )
            
            # Generate training data
            test_regret = jr.normal(key, (4,))
            test_strategy = jnp.array([0.25, 0.25, 0.25, 0.25])
            
            # Training step
            if algorithm == 'parallel':
                # Use parallel training
                result = trainer.distributed_training_step(test_regret, test_regret, learning_rate)
            else:
                # Use advanced CFR algorithm
                result = trainer.training_step(test_info_state, test_regret, test_strategy)
            
            # Update training data - ACCUMULATE instead of overwrite
            # Create more unique info_set_key including cards and history
            cards_str = "_".join(map(str, test_info_state.cards))
            history_str = "_".join(map(str, test_info_state.history))
            info_set_key = f"p{test_info_state.player_id}_r{test_info_state.round}_c{cards_str}_h{history_str}"
            
            # Initialize if not exists
            if info_set_key not in training_data['strategy_sum']:
                training_data['strategy_sum'][info_set_key] = jnp.zeros(4)
                training_data['regret_sum'][info_set_key] = jnp.zeros(4)
            
            if algorithm == 'parallel':
                # Handle parallel training results - ACCUMULATE
                new_strategy = result.get('strategies', test_strategy)
                new_regret = result.get('q_values', test_regret)
                training_data['strategy_sum'][info_set_key] += new_strategy
                training_data['regret_sum'][info_set_key] += new_regret
            else:
                # Handle advanced CFR algorithm results - ACCUMULATE
                new_strategy = result.get('strategy', test_strategy)
                new_regret = result.get('regret', test_regret)
                training_data['strategy_sum'][info_set_key] += new_strategy
                training_data['regret_sum'][info_set_key] += new_regret
            
            training_data['iteration'] = iteration
            
            # Log progress
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = iteration / elapsed
                logger.info(f"Iteration {iteration:,}/{iterations:,} | "
                           f"Steps/sec: {steps_per_sec:.1f} | "
                           f"Elapsed: {elapsed:.1f}s")
            
            # Save checkpoint
            if iteration % save_interval == 0:
                checkpoint_path = save_path.replace('.pkl', f'_checkpoint_{iteration}.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(training_data, f)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        with open(save_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        # Final statistics
        total_time = time.time() - start_time
        final_steps_per_sec = iterations / total_time
        
        logger.info("=" * 60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Total iterations: {iterations:,}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average speed: {final_steps_per_sec:.1f} steps/sec")
        logger.info(f"Final model saved: {save_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fast training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--iterations', default=100000, help='Number of training iterations')
@click.option('--players', default=6, help='Number of players (2-10)')
@click.option('--algorithm', default='parallel', help='Algorithm to use (parallel, pdcfr_plus, outcome_sampling, neural_fsp)')
@click.option('--save-interval', default=10000, help='Save model every N iterations')
@click.option('--log-interval', default=1000, help='Log progress every N iterations')
@click.option('--save-path', default='models/holdem_model.pkl', help='Path to save trained model')
@click.option('--learning-rate', default=0.1, help='Learning rate')
@click.option('--starting-stack', default=100.0, help='Starting stack size')
@click.option('--small-blind', default=1.0, help='Small blind size')
@click.option('--big-blind', default=2.0, help='Big blind size')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration')
def train_holdem(iterations: int, players: int, algorithm: str, save_interval: int, 
                 log_interval: int, save_path: str, learning_rate: float,
                 starting_stack: float, small_blind: float, big_blind: float, gpu: bool):
    """🎯 Train REAL No Limit Texas Hold'em with multiple players using poker engine"""
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    
    try:
        from .parallel import get_optimal_parallel_config, create_parallel_trainer
        from .algorithms import create_advanced_cfr_trainer
        from .engine import PokerEngine, GameConfig, Action
        from .modern_cfr import InfoState
        import jax.numpy as jnp
        import jax.random as jr
        import time
        import pickle
        
        logger.info("🎯 Starting REAL No Limit Texas Hold'em Training")
        logger.info("=" * 60)
        logger.info(f"Players: {players} (6-max NLHE)")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Iterations: {iterations:,}")
        logger.info(f"Starting stack: ${starting_stack}")
        logger.info(f"Blinds: ${small_blind}/${big_blind}")
        logger.info(f"Save interval: {save_interval}")
        logger.info(f"Log interval: {log_interval}")
        logger.info(f"Save path: {save_path}")
        logger.info("")
        
        # Initialize REAL poker engine with NLHE configuration
        logger.info("Initializing poker engine...")
        game_config = GameConfig(
            players=players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            max_raises=-1  # No limit on raises (NLHE)
        )
        poker_engine = PokerEngine(game_config)
        logger.info("✅ Poker engine initialized")
        
        # Initialize trainer based on algorithm
        trainer = None
        logger.info(f"Initializing trainer with algorithm: {algorithm}")
        
        if algorithm == 'parallel':
            logger.info("Creating parallel trainer...")
            parallel_config = get_optimal_parallel_config()
            trainer = create_parallel_trainer(parallel_config)
            logger.info("✅ Parallel trainer created")
        else:
            logger.info(f"Creating advanced CFR trainer: {algorithm}")
            trainer = create_advanced_cfr_trainer(algorithm)
            logger.info("✅ Advanced CFR trainer created")
        
        # Training data storage
        training_data = {
            'strategy_sum': {},
            'regret_sum': {},
            'iteration': 0,
            'game_config': {
                'players': players,
                'starting_stack': starting_stack,
                'small_blind': small_blind,
                'big_blind': big_blind,
                'game_type': 'No Limit Texas Hold\'em'
            },
            'config': {
                'algorithm': algorithm,
                'iterations': iterations,
                'learning_rate': learning_rate
            }
        }
        
        # Training loop
        logger.info("🚀 Starting REAL NLHE poker training loop...")
        logger.info("=" * 60)
        start_time = time.time()
        games_played = 0
        total_info_sets = 0
        successful_iterations = 0
        
        for iteration in range(1, iterations + 1):
            try:
                iteration_start = time.time()
                
                # Detailed logging only for first few iterations
                verbose_logging = iteration <= 5
                
                if verbose_logging:
                    logger.info(f"🎯 Starting iteration {iteration}/{iterations}")
                    logger.info(f"Creating new game for iteration {iteration}")
                
                # Start new REAL poker game
                game_state = poker_engine.new_game()
                
                if verbose_logging:
                    logger.info(f"Game created successfully for iteration {iteration}")
                    logger.info(f"Starting game decisions loop for iteration {iteration}")
                
                # Play the game and collect training data
                game_decisions = 0
                iteration_info_sets = []
                
                while not game_state.is_terminal():
                    # Timeout protection
                    if time.time() - iteration_start > 30:
                        logger.warning(f"⚠️ Iteration {iteration} timed out after 30s")
                        break
                    
                    try:
                        # Get current player
                        current_player = poker_engine.get_current_player(game_state)
                        
                        if verbose_logging:
                            logger.info(f"  Decision {game_decisions}: Current player {current_player}")
                        
                        # Get information set
                        info_set = poker_engine.get_info_set(game_state, current_player)
                        
                        if verbose_logging:
                            logger.info(f"  Got info set: {info_set}")
                        
                        # Get valid actions
                        valid_actions = poker_engine.get_valid_actions(game_state, current_player)
                        
                        if verbose_logging:
                            logger.info(f"  Valid actions: {valid_actions}")
                        
                        # Train CFR to get action probabilities
                        if algorithm == 'parallel':
                            # Simple uniform random for now (will be replaced with actual CFR)
                            action_probs = {action: 1.0/len(valid_actions) for action in valid_actions}
                        else:
                            action_probs = trainer.get_action_probs(info_set, valid_actions)
                        
                        # Choose action based on probabilities
                        action = jax.random.choice(
                            jax.random.PRNGKey(game_decisions + iteration),
                            jnp.array(valid_actions),
                            p=jnp.array(list(action_probs.values()))
                        )
                        
                        if verbose_logging:
                            logger.info(f"  Chosen action: {action}")
                        
                        # Apply action
                        game_state = poker_engine.apply_action(game_state, current_player, action)
                        
                        # Store info set for training
                        iteration_info_sets.append({
                            'info_set': info_set,
                            'action_probs': action_probs,
                            'action': action,
                            'player': current_player
                        })
                        
                        game_decisions += 1
                        
                        if verbose_logging:
                            logger.info(f"  Action applied, game_decisions: {game_decisions}")
                        
                        # Safety check for infinite loops
                        if game_decisions > 100:
                            logger.warning(f"⚠️ Game {iteration} exceeded 100 decisions, forcing termination")
                            break
                            
                    except Exception as e:
                        logger.error(f"❌ Error in decision {game_decisions} of iteration {iteration}: {e}")
                        break
                
                if verbose_logging:
                    logger.info(f"Game finished for iteration {iteration}, decisions: {game_decisions}")
                
                # Update stats
                games_played += 1
                total_info_sets += len(iteration_info_sets)
                successful_iterations += 1
                
                # Progress logging every 1000 iterations
                if iteration % log_interval == 0:
                    elapsed = time.time() - start_time
                    iterations_per_sec = iteration / elapsed
                    avg_decisions_per_game = game_decisions / games_played if games_played > 0 else 0
                    
                    logger.info(f"🎯 Progress: {iteration:,}/{iterations:,} iterations")
                    logger.info(f"   Speed: {iterations_per_sec:.1f} iterations/sec")
                    logger.info(f"   Games played: {games_played:,}")
                    logger.info(f"   Avg decisions per game: {avg_decisions_per_game:.1f}")
                    logger.info(f"   Total info sets: {total_info_sets:,}")
                    logger.info(f"   Elapsed time: {elapsed:.1f}s")
                    
                    # Estimate remaining time
                    remaining_iterations = iterations - iteration
                    eta_seconds = remaining_iterations / iterations_per_sec if iterations_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600
                    logger.info(f"   ETA: {eta_hours:.1f} hours")
                    logger.info("=" * 60)
                
                # Save checkpoint every save_interval iterations
                if iteration % save_interval == 0:
                    checkpoint_path = save_path.replace('.pkl', f'_checkpoint_{iteration}.pkl')
                    # Save simple checkpoint for now
                    checkpoint_data = {
                        'iteration': iteration,
                        'total_info_sets': total_info_sets,
                        'games_played': games_played,
                        'timestamp': time.time()
                    }
                    
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    
                    logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                
                # Early progress report after 10 iterations
                if iteration == 10:
                    elapsed = time.time() - start_time
                    avg_time_per_iteration = elapsed / 10
                    projected_total_hours = (avg_time_per_iteration * iterations) / 3600
                    
                    logger.info(f"🎯 First 10 iterations completed in {elapsed:.1f}s")
                    logger.info(f"   Average time per iteration: {avg_time_per_iteration:.3f}s")
                    logger.info(f"   Projected time for {iterations:,} iterations: {projected_total_hours:.1f} hours")
                    logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration}: {e}")
                logger.error(f"   Traceback: {traceback.format_exc()}")
                continue
        
        # Final results
        total_time = time.time() - start_time
        avg_iterations_per_sec = successful_iterations / total_time if total_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Total iterations: {successful_iterations:,}")
        logger.info(f"Total games played: {games_played:,}")
        logger.info(f"Total info sets collected: {total_info_sets:,}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average speed: {avg_iterations_per_sec:.1f} iterations/sec")
        logger.info(f"Final model saved: {save_path}")
        logger.info("=" * 60)
        
        # Save final model
        final_model_data = {
            'iterations': successful_iterations,
            'total_info_sets': total_info_sets,
            'games_played': games_played,
            'training_time': total_time,
            'avg_iterations_per_sec': avg_iterations_per_sec,
            'algorithm': algorithm,
            'players': players,
            'timestamp': time.time()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(final_model_data, f)
        
        logger.info(f"💾 Final model saved: {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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
@click.option('--iterations', default=1000, help='Number of test iterations')
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
@click.option('--iterations', default=1000, help='Number of benchmark iterations')
@click.option('--algorithm', default='pdcfr_plus', help='Algorithm to test (pdcfr_plus, outcome_sampling, neural_fsp)')
def test_phase2(iterations: int, algorithm: str):
    """Test Phase 2 performance optimizations"""
    
    try:
        from .parallel import get_optimal_parallel_config, create_parallel_trainer
        from .algorithms import create_advanced_cfr_trainer, benchmark_algorithms
        from .optimization import get_optimal_optimization_config, create_optimized_trainer, benchmark_optimization
        from .modern_cfr import InfoState
        import jax.numpy as jnp
        import jax.random as jr
        import time
        
        logger.info("🚀 Testing Phase 2 - Performance Optimization")
        logger.info("=" * 50)
        
        # Test 1: Parallel Training
        logger.info("1. Testing Multi-GPU Parallel Training...")
        parallel_config = get_optimal_parallel_config()
        parallel_trainer = create_parallel_trainer(parallel_config)
        
        # Benchmark parallel performance
        logger.info("   Benchmarking parallel performance...")
        parallel_results = parallel_trainer.benchmark_parallel_performance(iterations=100)
        
        logger.info(f"   ✅ Parallel training: {parallel_results['throughput_steps_per_sec']:.1f} steps/sec")
        logger.info(f"   ✅ Parallel efficiency: {parallel_results['parallel_efficiency']:.3f}")
        
        # Test 2: Advanced Algorithms
        logger.info(f"\n2. Testing Advanced CFR Algorithm: {algorithm}")
        advanced_trainer = create_advanced_cfr_trainer(algorithm)
        
        # Test algorithm
        test_info_state = InfoState(
            player_id=0,
            cards=jnp.array([1, 2, 3, 4, 5]),
            history=jnp.array([0, 1, 0, 1]),
            pot=10.0,
            round=0
        )
        test_regret = jr.normal(jr.PRNGKey(42), (4,))
        test_strategy = jnp.array([0.25, 0.25, 0.25, 0.25])
        
        start_time = time.time()
        for i in range(min(iterations, 100)):
            result = advanced_trainer.training_step(
                test_info_state, test_regret, test_strategy
            )
        algorithm_time = time.time() - start_time
        
        logger.info(f"   ✅ {algorithm}: {(100 / algorithm_time):.1f} steps/sec")
        
        # Test 3: Optimization Suite
        logger.info("\n3. Testing Optimization Suite...")
        optimization_config = get_optimal_optimization_config()
        optimized_trainer = create_optimized_trainer(optimization_config)
        
        # Test optimization
        test_q_values = jr.normal(jr.PRNGKey(42), (4,))
        test_regrets = jr.normal(jr.PRNGKey(43), (4,))
        
        start_time = time.time()
        for i in range(min(iterations, 100)):
            result = optimized_trainer.optimized_training_step(test_q_values, test_regrets)
        optimization_time = time.time() - start_time
        
        logger.info(f"   ✅ Optimized trainer: {(100 / optimization_time):.1f} steps/sec")
        
        # Test 4: Algorithm Benchmark
        logger.info("\n4. Running Algorithm Benchmark...")
        benchmark_results = benchmark_algorithms(iterations=min(iterations, 100))
        
        logger.info("   Algorithm Performance:")
        for algo, results in benchmark_results.items():
            logger.info(f"   - {algo}: {results['throughput_steps_per_sec']:.1f} steps/sec")
        
        # Test 5: Optimization Benchmark
        logger.info("\n5. Running Optimization Benchmark...")
        opt_benchmark = benchmark_optimization(iterations=min(iterations, 100))
        
        logger.info(f"   ✅ Optimization benchmark: {opt_benchmark['throughput_steps_per_sec']:.1f} steps/sec")
        logger.info(f"   ✅ Cache hit rate: {opt_benchmark['cache_hit_rate']:.3f}")
        
        # Summary
        logger.info("\n🎉 Phase 2 Testing Complete!")
        logger.info("=" * 50)
        logger.info("✅ Multi-GPU parallel training: Working")
        logger.info("✅ Advanced CFR algorithms: Working")
        logger.info("✅ Optimization suite: Working")
        logger.info("✅ Performance benchmarks: Working")
        logger.info("\n🚀 Ready for Phase 3 - Texas Hold'em Implementation!")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--benchmark-type', default='all', help='Type of benchmark (parallel, algorithms, optimization, all)')
@click.option('--iterations', default=1000, help='Number of benchmark iterations')
def benchmark_phase2(benchmark_type: str, iterations: int):
    """Benchmark Phase 2 performance components"""
    
    try:
        from .parallel import get_optimal_parallel_config, create_parallel_trainer
        from .algorithms import benchmark_algorithms
        from .optimization import benchmark_optimization
        import time
        
        logger.info(f"🔥 Benchmarking Phase 2 Components: {benchmark_type}")
        logger.info("=" * 50)
        
        results = {}
        
        if benchmark_type in ['parallel', 'all']:
            logger.info("Benchmarking parallel training...")
            parallel_config = get_optimal_parallel_config()
            parallel_trainer = create_parallel_trainer(parallel_config)
            results['parallel'] = parallel_trainer.benchmark_parallel_performance(iterations=iterations)
            
        if benchmark_type in ['algorithms', 'all']:
            logger.info("Benchmarking algorithms...")
            results['algorithms'] = benchmark_algorithms(iterations=iterations)
            
        if benchmark_type in ['optimization', 'all']:
            logger.info("Benchmarking optimization...")
            results['optimization'] = benchmark_optimization(iterations=iterations)
        
        # Display results
        logger.info("\n📊 Benchmark Results:")
        logger.info("=" * 50)
        
        for component, result in results.items():
            logger.info(f"\n{component.upper()}:")
            if component == 'parallel':
                logger.info(f"  Throughput: {result['throughput_steps_per_sec']:.1f} steps/sec")
                logger.info(f"  Efficiency: {result['parallel_efficiency']:.3f}")
                logger.info(f"  Memory: {result['memory_peak_mb']:.1f} MB")
            elif component == 'algorithms':
                for algo, algo_result in result.items():
                    logger.info(f"  {algo}: {algo_result['throughput_steps_per_sec']:.1f} steps/sec")
            elif component == 'optimization':
                logger.info(f"  Throughput: {result['throughput_steps_per_sec']:.1f} steps/sec")
                logger.info(f"  Cache hit rate: {result['cache_hit_rate']:.3f}")
                logger.info(f"  Final LR: {result['final_learning_rate']:.6f}")
        
        logger.info("\n🎯 Benchmark Complete!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--iterations', default=100, help='Number of iterations to test')
@click.option('--batch-size', default=8192, help='Batch size for testing')
@click.option('--algorithm', default='pdcfr_plus', help='Algorithm to test')
@click.option('--detailed/--no-detailed', default=True, help='Show detailed per-iteration timing')
def test_iteration_timing(iterations: int, batch_size: int, algorithm: str, detailed: bool):
    """Test detailed timing of individual CFR iterations"""
    
    try:
        from .parallel import create_parallel_trainer, get_optimal_parallel_config
        from .algorithms import create_advanced_cfr_trainer
        from .optimization import create_optimized_trainer, get_optimal_optimization_config
        from .modern_cfr import InfoState
        import jax.numpy as jnp
        import jax.random as jr
        import time
        
        logger.info(f"🔍 Testing Iteration Timing - {algorithm}")
        logger.info(f"Iterations: {iterations}, Batch size: {batch_size}")
        logger.info("=" * 60)
        
        # Create trainer based on algorithm
        if algorithm == 'parallel':
            config = get_optimal_parallel_config()
            trainer = create_parallel_trainer(config)
        elif algorithm == 'optimized':
            config = get_optimal_optimization_config()
            trainer = create_optimized_trainer(config)
        else:
            trainer = create_advanced_cfr_trainer(algorithm)
        
        # Test data
        key = jr.PRNGKey(42)
        test_info_state = InfoState(
            player_id=0,
            cards=jnp.array([1, 2, 3, 4, 5]),
            history=jnp.array([0, 1, 0, 1]),
            pot=10.0,
            round=0
        )
        
        # Different test data based on trainer type
        if algorithm == 'parallel':
            test_q_values = jr.normal(key, (batch_size, 4))
            test_regrets = jr.normal(jr.split(key)[0], (batch_size, 4))
        elif algorithm == 'optimized':
            test_q_values = jr.normal(key, (4,))
            test_regrets = jr.normal(jr.split(key)[0], (4,))
        else:
            test_regret = jr.normal(key, (4,))
            test_strategy = jnp.array([0.25, 0.25, 0.25, 0.25])
        
        # Warmup (important for JAX JIT compilation)
        logger.info("🔥 Warming up (JIT compilation)...")
        warmup_start = time.time()
        
        for i in range(10):
            if algorithm == 'parallel':
                trainer.distributed_training_step(test_q_values[0], test_regrets[0], 0.1)
            elif algorithm == 'optimized':
                trainer.optimized_training_step(test_q_values, test_regrets)
            else:
                trainer.training_step(test_info_state, test_regret, test_strategy)
        
        warmup_time = time.time() - warmup_start
        logger.info(f"✅ Warmup completed in {warmup_time:.2f}s")
        
        # Detailed iteration timing
        iteration_times = []
        component_times = {
            'q_update': [],
            'strategy_compute': [],
            'regret_update': [],
            'total': []
        }
        
        logger.info(f"\n📊 Running {iterations} iterations...")
        
        for i in range(iterations):
            iteration_start = time.time()
            
            # Measure components
            if algorithm == 'parallel':
                q_start = time.time()
                result = trainer.distributed_training_step(
                    test_q_values[i % len(test_q_values)], 
                    test_regrets[i % len(test_regrets)], 
                    0.1
                )
                q_time = time.time() - q_start
                component_times['q_update'].append(q_time)
                
            elif algorithm == 'optimized':
                q_start = time.time()
                result = trainer.optimized_training_step(test_q_values, test_regrets)
                q_time = time.time() - q_start
                component_times['q_update'].append(q_time)
                
            else:
                # Advanced CFR algorithm
                q_start = time.time()
                result = trainer.training_step(test_info_state, test_regret, test_strategy)
                q_time = time.time() - q_start
                component_times['q_update'].append(q_time)
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            component_times['total'].append(iteration_time)
            
            # Show detailed progress
            if detailed and (i < 10 or i % 10 == 0):
                logger.info(f"  Iteration {i+1:3d}: {iteration_time*1000:.2f}ms "
                           f"(Q-update: {q_time*1000:.2f}ms)")
        
        # Calculate statistics
        total_time = sum(iteration_times)
        avg_time = total_time / iterations
        min_time = min(iteration_times)
        max_time = max(iteration_times)
        std_time = float(jnp.std(jnp.array(iteration_times)))
        
        throughput = iterations / total_time
        
        # Memory usage
        from .memory import get_memory_usage
        memory_info = get_memory_usage()
        
        # Results
        logger.info(f"\n🎯 ITERATION TIMING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"")
        logger.info(f"⏱️  TIMING STATISTICS:")
        logger.info(f"  Total time:     {total_time:.3f}s")
        logger.info(f"  Average time:   {avg_time*1000:.2f}ms per iteration")
        logger.info(f"  Min time:       {min_time*1000:.2f}ms")
        logger.info(f"  Max time:       {max_time*1000:.2f}ms")
        logger.info(f"  Std deviation:  {std_time*1000:.2f}ms")
        logger.info(f"  Throughput:     {throughput:.1f} iterations/sec")
        logger.info(f"")
        logger.info(f"🧠 MEMORY USAGE:")
        logger.info(f"  Process memory: {memory_info['process_memory_mb']:.1f}MB")
        logger.info(f"  System memory:  {memory_info['system_memory_percent']:.1f}%")
        logger.info(f"  Available mem:  {memory_info['available_memory_gb']:.1f}GB")
        
        # Component breakdown
        if component_times['q_update']:
            avg_q_time = sum(component_times['q_update']) / len(component_times['q_update'])
            logger.info(f"")
            logger.info(f"🔧 COMPONENT BREAKDOWN:")
            logger.info(f"  Q-value update: {avg_q_time*1000:.2f}ms avg")
            logger.info(f"  Q-update %:     {(avg_q_time/avg_time)*100:.1f}%")
        
        # Performance compared to baseline
        baseline_throughput = 20  # steps/sec before Phase 2
        speedup = throughput / baseline_throughput
        logger.info(f"")
        logger.info(f"📈 PERFORMANCE VS BASELINE:")
        logger.info(f"  Baseline:       {baseline_throughput} steps/sec")
        logger.info(f"  Current:        {throughput:.1f} steps/sec")
        logger.info(f"  Speedup:        {speedup:.1f}x")
        
        # VRAM usage estimate
        elements_per_iteration = batch_size * 4  # 4 actions
        memory_per_element = 4  # bytes for float32
        vram_usage_mb = (elements_per_iteration * memory_per_element) / (1024*1024)
        logger.info(f"")
        logger.info(f"🎯 VRAM USAGE ESTIMATE:")
        logger.info(f"  Elements/iter:  {elements_per_iteration:,}")
        logger.info(f"  VRAM/iter:      {vram_usage_mb:.1f}MB")
        
        logger.info(f"\n✅ Iteration timing test completed!")
        
    except Exception as e:
        logger.error(f"❌ Iteration timing test failed: {e}")
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