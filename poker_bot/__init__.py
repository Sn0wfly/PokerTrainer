"""
🎯 PokerTrainer - GPU-Native Poker AI

A high-performance poker bot using JAX + MCCFR for GPU acceleration.
Now with Modern CFR (CFVFP) and GPU optimization!
"""

__version__ = "0.2.0"
__author__ = "PokerTrainer Team"

# Legacy components
from .engine import PokerEngine
from .evaluator import HandEvaluator
from .trainer import SimpleMCCFRTrainer, MCCFRConfig, create_trainer
from .bot import PokerBot

# Modern CFR modules
from .modern_cfr import (
    CFVFPTrainer, 
    CFVFPConfig, 
    InfoState,
    ActionValue,
    create_cfvfp_trainer,
    info_state_from_game,
    batch_update_q_values,
    batch_compute_strategies
)

from .gpu_config import (
    init_gpu_environment,
    get_device_info,
    setup_gpu_optimization
)

from .memory import (
    MemoryMonitor,
    MemoryAwareCache,
    get_memory_usage,
    log_memory_usage,
    checkpoint_wrapper,
    remat_wrapper,
    memory_efficient_cfr_step,
    MemoryEfficientDataLoader
)

__all__ = [
    # Legacy components
    "PokerEngine",
    "HandEvaluator", 
    "SimpleMCCFRTrainer",
    "MCCFRConfig",
    "create_trainer",
    "PokerBot",
    
    # Modern CFR
    "CFVFPTrainer",
    "CFVFPConfig",
    "InfoState",
    "ActionValue",
    "create_cfvfp_trainer",
    "info_state_from_game",
    "batch_update_q_values",
    "batch_compute_strategies",
    
    # GPU Configuration
    "init_gpu_environment",
    "get_device_info",
    "setup_gpu_optimization",
    
    # Memory Management
    "MemoryMonitor",
    "MemoryAwareCache",
    "get_memory_usage",
    "log_memory_usage",
    "checkpoint_wrapper",
    "remat_wrapper",
    "memory_efficient_cfr_step",
    "MemoryEfficientDataLoader",
] 