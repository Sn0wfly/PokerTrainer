"""
🎯 PokerTrainer - GPU-Native Poker AI

A high-performance poker bot using JAX + MCCFR for GPU acceleration.

Phase 1: Modern CFR (CFVFP) and GPU optimization
Phase 2: Performance optimization with multi-GPU, advanced algorithms, and smart caching
"""

__version__ = "0.3.0"
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
    MemoryEfficientDataLoader,
    AdaptiveBatchManager
)

# Phase 2: Performance Optimization Modules
from .parallel import (
    MultiGPUTrainer,
    ParallelConfig,
    DistributedCFRCoordinator,
    GradientCheckpointManager,
    PipelineParallelTrainer,
    create_parallel_trainer,
    get_optimal_parallel_config
)

from .algorithms import (
    PDCFRPlus,
    PDCFRConfig,
    OutcomeSamplingCFR,
    NeuralFictitiousSelfPlay,
    AdvancedCFRSuite,
    create_advanced_cfr_trainer,
    benchmark_algorithms
)

from .optimization import (
    OptimizedCFRTrainer,
    OptimizationConfig,
    GradientAccumulator,
    SmartCache,
    AdaptiveLearningRateScheduler,
    PerformanceProfiler,
    create_optimized_trainer,
    get_optimal_optimization_config,
    benchmark_optimization
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
    "AdaptiveBatchManager",
    
    # Phase 2: Parallel Training
    "MultiGPUTrainer",
    "ParallelConfig",
    "DistributedCFRCoordinator",
    "GradientCheckpointManager",
    "PipelineParallelTrainer",
    "create_parallel_trainer",
    "get_optimal_parallel_config",
    
    # Phase 2: Advanced Algorithms
    "PDCFRPlus",
    "PDCFRConfig",
    "OutcomeSamplingCFR",
    "NeuralFictitiousSelfPlay",
    "AdvancedCFRSuite",
    "create_advanced_cfr_trainer",
    "benchmark_algorithms",
    
    # Phase 2: Optimization
    "OptimizedCFRTrainer",
    "OptimizationConfig",
    "GradientAccumulator",
    "SmartCache",
    "AdaptiveLearningRateScheduler",
    "PerformanceProfiler",
    "create_optimized_trainer",
    "get_optimal_optimization_config",
    "benchmark_optimization",
] 