"""
Advanced Optimization Module for Modern CFR
===========================================

This module implements advanced optimization techniques for maximum CFR performance,
including gradient accumulation, smart caching, and adaptive optimization strategies.

Key Features:
- Gradient Accumulation: Simulate large batch sizes with limited memory
- Smart Caching: Cache JIT-compiled functions for reuse
- Adaptive Optimization: Dynamic learning rate and parameter adjustment
- Performance Profiling: Detailed performance analysis and bottleneck detection
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, value_and_grad
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import logging
import functools
import time
import os
import pickle
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import weakref

from .gpu_config import get_device_info, setup_mixed_precision
from .memory import MemoryMonitor, get_memory_usage, AdaptiveBatchManager
from .modern_cfr import CFVFPConfig

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    gradient_accumulation_steps: int = 8
    max_cache_size: int = 1000
    adaptive_lr_window: int = 100
    performance_profile_interval: int = 1000
    cache_cleanup_threshold: float = 0.8  # Memory usage threshold for cache cleanup
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_learning_rate_scheduling: bool = True
    warmup_steps: int = 1000
    dtype: Any = jnp.bfloat16
    accumulation_dtype: Any = jnp.float32

class GradientAccumulator:
    """Gradient accumulation for simulating large batch sizes"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.accumulated_gradients = {}
        self.accumulation_count = 0
        self.gradient_norms = deque(maxlen=config.adaptive_lr_window)
        
        logger.info(f"GradientAccumulator initialized with {config.gradient_accumulation_steps} steps")
    
    def reset_gradients(self):
        """Reset accumulated gradients"""
        self.accumulated_gradients = {}
        self.accumulation_count = 0
    
    def accumulate_gradients(self, gradients: Dict[str, jnp.ndarray]) -> bool:
        """Accumulate gradients and return True if ready to apply"""
        self.accumulation_count += 1
        
        # Initialize accumulated gradients if first accumulation
        if not self.accumulated_gradients:
            self.accumulated_gradients = {
                key: jnp.zeros_like(grad) for key, grad in gradients.items()
            }
        
        # Add gradients to accumulation
        for key, grad in gradients.items():
            self.accumulated_gradients[key] += grad
        
        # Check if we have enough accumulated gradients
        if self.accumulation_count >= self.config.gradient_accumulation_steps:
            # Average the gradients
            for key in self.accumulated_gradients:
                self.accumulated_gradients[key] /= self.accumulation_count
            
            # Compute gradient norm for adaptive learning rate
            grad_norm = self._compute_gradient_norm(self.accumulated_gradients)
            self.gradient_norms.append(grad_norm)
            
            return True
        
        return False
    
    def _compute_gradient_norm(self, gradients: Dict[str, jnp.ndarray]) -> float:
        """Compute L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += jnp.sum(grad ** 2)
        return float(jnp.sqrt(total_norm))
    
    def apply_gradient_clipping(self, gradients: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Apply gradient clipping to prevent exploding gradients"""
        if not self.config.use_gradient_clipping:
            return gradients
        
        grad_norm = self._compute_gradient_norm(gradients)
        
        if grad_norm > self.config.gradient_clip_norm:
            # Scale gradients to clip norm
            scale_factor = self.config.gradient_clip_norm / grad_norm
            clipped_gradients = {
                key: grad * scale_factor for key, grad in gradients.items()
            }
            logger.debug(f"Gradient clipping applied: norm {grad_norm:.4f} -> {self.config.gradient_clip_norm}")
            return clipped_gradients
        
        return gradients
    
    def get_accumulated_gradients(self) -> Dict[str, jnp.ndarray]:
        """Get the accumulated gradients"""
        return self.accumulated_gradients
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get statistics about gradient accumulation"""
        if not self.gradient_norms:
            return {}
        
        return {
            'mean_gradient_norm': float(jnp.mean(jnp.array(self.gradient_norms))),
            'std_gradient_norm': float(jnp.std(jnp.array(self.gradient_norms))),
            'max_gradient_norm': float(jnp.max(jnp.array(self.gradient_norms))),
            'min_gradient_norm': float(jnp.min(jnp.array(self.gradient_norms)))
        }

class SmartCache:
    """Smart caching system for JIT-compiled functions"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(float)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Use weak references to avoid memory leaks
        self.compiled_functions = weakref.WeakValueDictionary()
        
        logger.info(f"SmartCache initialized with max size {config.max_cache_size}")
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call"""
        # Create a hash of the function signature and arguments
        key_parts = [func_name]
        
        # Add argument shapes and dtypes
        for arg in args:
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                key_parts.append(f"{arg.shape}_{arg.dtype}")
            else:
                key_parts.append(str(type(arg)))
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                key_parts.append(f"{k}:{v.shape}_{v.dtype}")
            else:
                key_parts.append(f"{k}:{type(v)}")
        
        return "_".join(key_parts)
    
    def get_cached_function(self, func_name: str, func: Callable, 
                          args: Tuple, kwargs: Dict) -> Callable:
        """Get cached compiled function or compile and cache"""
        cache_key = self._generate_cache_key(func_name, args, kwargs)
        
        if cache_key in self.compiled_functions:
            self.cache_hits += 1
            self.access_counts[cache_key] += 1
            self.access_times[cache_key] = time.time()
            return self.compiled_functions[cache_key]
        else:
            self.cache_misses += 1
            
            # Compile function
            compiled_func = jit(func)
            
            # Cache management
            if len(self.compiled_functions) >= self.config.max_cache_size:
                self._cleanup_cache()
            
            self.compiled_functions[cache_key] = compiled_func
            self.access_counts[cache_key] = 1
            self.access_times[cache_key] = time.time()
            
            return compiled_func
    
    def _cleanup_cache(self):
        """Clean up cache based on access patterns"""
        # Remove least recently used items
        sorted_items = sorted(
            self.access_times.items(), 
            key=lambda x: x[1]
        )
        
        # Remove oldest 25% of items
        num_to_remove = len(sorted_items) // 4
        for cache_key, _ in sorted_items[:num_to_remove]:
            if cache_key in self.compiled_functions:
                del self.compiled_functions[cache_key]
            if cache_key in self.access_counts:
                del self.access_counts[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
        
        logger.debug(f"Cache cleanup: removed {num_to_remove} items")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.compiled_functions),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'max_cache_size': self.config.max_cache_size
        }

class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler based on training dynamics"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.base_learning_rate = 0.1
        self.current_learning_rate = self.base_learning_rate
        self.step_count = 0
        self.loss_history = deque(maxlen=config.adaptive_lr_window)
        self.plateau_threshold = 0.001
        self.plateau_patience = 50
        self.plateau_counter = 0
        
        logger.info("AdaptiveLearningRateScheduler initialized")
    
    def update_learning_rate(self, loss: float) -> float:
        """Update learning rate based on loss"""
        self.step_count += 1
        self.loss_history.append(loss)
        
        if not self.config.use_learning_rate_scheduling:
            return self.current_learning_rate
        
        # Warmup phase
        if self.step_count < self.config.warmup_steps:
            self.current_learning_rate = (
                self.base_learning_rate * self.step_count / self.config.warmup_steps
            )
            return self.current_learning_rate
        
        # Plateau detection
        if len(self.loss_history) >= self.config.adaptive_lr_window:
            recent_losses = list(self.loss_history)[-self.plateau_patience:]
            if len(recent_losses) >= self.plateau_patience:
                loss_improvement = abs(recent_losses[0] - recent_losses[-1])
                
                if loss_improvement < self.plateau_threshold:
                    self.plateau_counter += 1
                else:
                    self.plateau_counter = 0
                
                # Reduce learning rate if plateau detected
                if self.plateau_counter >= self.plateau_patience:
                    self.current_learning_rate *= 0.5
                    self.plateau_counter = 0
                    logger.info(f"Learning rate reduced to {self.current_learning_rate:.6f}")
        
        # Cosine annealing
        if self.step_count > self.config.warmup_steps:
            progress = (self.step_count - self.config.warmup_steps) / 10000
            cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            self.current_learning_rate = self.base_learning_rate * cosine_factor
        
        return self.current_learning_rate
    
    def get_learning_rate_stats(self) -> Dict[str, float]:
        """Get learning rate statistics"""
        return {
            'current_learning_rate': self.current_learning_rate,
            'base_learning_rate': self.base_learning_rate,
            'step_count': self.step_count,
            'plateau_counter': self.plateau_counter
        }

class PerformanceProfiler:
    """Performance profiler for CFR training"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profile_data = defaultdict(list)
        self.timers = {}
        self.step_count = 0
        
        logger.info("PerformanceProfiler initialized")
    
    def start_timer(self, name: str):
        """Start a timer for profiling"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str):
        """End a timer and record the duration"""
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.profile_data[name].append(duration)
            del self.timers[name]
            return duration
        return 0.0
    
    def profile_step(self, step_data: Dict[str, Any]):
        """Profile a training step"""
        self.step_count += 1
        
        # Record step data
        for key, value in step_data.items():
            self.profile_data[key].append(value)
        
        # Generate report periodically
        if self.step_count % self.config.performance_profile_interval == 0:
            self.generate_profile_report()
    
    def generate_profile_report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance profile report"""
        report = {}
        
        for metric, values in self.profile_data.items():
            if values:
                report[metric] = {
                    'mean': float(jnp.mean(jnp.array(values))),
                    'std': float(jnp.std(jnp.array(values))),
                    'min': float(jnp.min(jnp.array(values))),
                    'max': float(jnp.max(jnp.array(values))),
                    'count': len(values)
                }
        
        # Log report
        logger.info("=== Performance Profile Report ===")
        for metric, stats in report.items():
            logger.info(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        
        return report

class OptimizedCFRTrainer:
    """Optimized CFR trainer with all performance enhancements"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize optimization components
        self.gradient_accumulator = GradientAccumulator(config)
        self.smart_cache = SmartCache(config)
        self.lr_scheduler = AdaptiveLearningRateScheduler(config)
        self.profiler = PerformanceProfiler(config)
        
        # Memory management
        self.memory_monitor = MemoryMonitor("OptimizedCFR")
        self.batch_manager = AdaptiveBatchManager(
            base_batch_size=2048,
            memory_threshold=0.8,
            max_batch_size=16384
        )
        
        # Training state
        self.step_count = 0
        self.total_loss = 0.0
        
        logger.info("OptimizedCFRTrainer initialized with all optimizations")
    
    @functools.partial(jit, static_argnums=(0,))
    def _optimized_q_update(self, q_values: jnp.ndarray, 
                          regrets: jnp.ndarray,
                          learning_rate: float) -> jnp.ndarray:
        """Optimized Q-value update with mixed precision"""
        # Convert to compute dtype for speed
        q_values_compute = q_values.astype(self.config.dtype)
        regrets_compute = regrets.astype(self.config.dtype)
        
        # Compute update
        q_update = learning_rate * regrets_compute
        new_q_values = q_values_compute + q_update
        
        # Convert back to accumulation dtype for stability
        return new_q_values.astype(self.config.accumulation_dtype)
    
    def optimized_training_step(self, q_values: jnp.ndarray,
                              regrets: jnp.ndarray,
                              learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """Execute optimized training step with all enhancements"""
        self.profiler.start_timer("training_step")
        
        with self.memory_monitor:
            # Adaptive learning rate
            if learning_rate is None:
                learning_rate = self.lr_scheduler.update_learning_rate(self.total_loss)
            
            # Get cached optimized function
            cached_update_fn = self.smart_cache.get_cached_function(
                "q_update", self._optimized_q_update, 
                (q_values, regrets, learning_rate), {}
            )
            
            # Execute update
            self.profiler.start_timer("q_update")
            new_q_values = cached_update_fn(q_values, regrets, learning_rate)
            q_update_time = self.profiler.end_timer("q_update")
            
            # Compute loss for adaptive learning rate
            loss = float(jnp.mean((new_q_values - q_values) ** 2))
            self.total_loss += loss
            
            # Update step count
            self.step_count += 1
            
            # Memory management
            current_memory = get_memory_usage()
            if current_memory['system_memory_percent'] > self.config.cache_cleanup_threshold * 100:
                self.smart_cache._cleanup_cache()
        
        step_time = self.profiler.end_timer("training_step")
        
        # Profile step
        step_data = {
            'step_time': step_time,
            'q_update_time': q_update_time,
            'loss': loss,
            'learning_rate': learning_rate,
            'memory_usage': current_memory['system_memory_percent']
        }
        
        self.profiler.profile_step(step_data)
        
        return {
            'new_q_values': new_q_values,
            'loss': loss,
            'learning_rate': learning_rate,
            'step_time': step_time,
            'memory_usage': current_memory,
            'step_count': self.step_count
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        stats = {
            'gradient_accumulation': self.gradient_accumulator.get_gradient_stats(),
            'smart_cache': self.smart_cache.get_cache_stats(),
            'learning_rate_scheduler': self.lr_scheduler.get_learning_rate_stats(),
            'performance_profile': self.profiler.generate_profile_report(),
            'step_count': self.step_count,
            'total_loss': self.total_loss
        }
        
        return stats
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state to file"""
        state = {
            'config': self.config,
            'step_count': self.step_count,
            'total_loss': self.total_loss,
            'lr_scheduler_state': {
                'current_learning_rate': self.lr_scheduler.current_learning_rate,
                'step_count': self.lr_scheduler.step_count,
                'plateau_counter': self.lr_scheduler.plateau_counter
            },
            'cache_stats': self.smart_cache.get_cache_stats()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Optimization state saved to {filepath}")
    
    def load_optimization_state(self, filepath: str):
        """Load optimization state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.step_count = state['step_count']
        self.total_loss = state['total_loss']
        
        # Restore learning rate scheduler
        self.lr_scheduler.current_learning_rate = state['lr_scheduler_state']['current_learning_rate']
        self.lr_scheduler.step_count = state['lr_scheduler_state']['step_count']
        self.lr_scheduler.plateau_counter = state['lr_scheduler_state']['plateau_counter']
        
        logger.info(f"Optimization state loaded from {filepath}")

def create_optimized_trainer(config: OptimizationConfig) -> OptimizedCFRTrainer:
    """Create optimized CFR trainer with all enhancements"""
    return OptimizedCFRTrainer(config)

def get_optimal_optimization_config() -> OptimizationConfig:
    """Get optimal optimization configuration based on hardware"""
    device_info = get_device_info()
    memory_info = get_memory_usage()
    
    # Calculate gradient accumulation steps based on memory
    # With 24GB RTX 3090, we can use smaller accumulation steps
    if memory_info['available_memory_gb'] > 20:
        gradient_accumulation_steps = 2
    elif memory_info['available_memory_gb'] > 16:
        gradient_accumulation_steps = 4
    elif memory_info['available_memory_gb'] > 8:
        gradient_accumulation_steps = 8
    else:
        gradient_accumulation_steps = 16
    
    # Cache size based on available memory - be more aggressive with large VRAM
    cache_size = min(5000, int(memory_info['available_memory_gb'] * 100))
    
    config = OptimizationConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_cache_size=cache_size,
        adaptive_lr_window=100,
        performance_profile_interval=1000,
        cache_cleanup_threshold=0.8,
        use_gradient_clipping=True,
        gradient_clip_norm=1.0,
        use_learning_rate_scheduling=True,
        warmup_steps=1000
    )
    
    logger.info(f"Optimal optimization config: {config}")
    return config

def benchmark_optimization(iterations: int = 1000) -> Dict[str, float]:
    """Benchmark optimization performance"""
    config = get_optimal_optimization_config()
    trainer = create_optimized_trainer(config)
    
    # Test data
    test_q_values = jr.normal(jr.PRNGKey(42), (4,))
    test_regrets = jr.normal(jr.PRNGKey(43), (4,))
    
    # Warmup
    for _ in range(10):
        trainer.optimized_training_step(test_q_values, test_regrets)
    
    # Benchmark
    start_time = time.time()
    for i in range(iterations):
        result = trainer.optimized_training_step(test_q_values, test_regrets)
        
        if i % 100 == 0:
            logger.info(f"Benchmark progress: {i}/{iterations}")
    
    total_time = time.time() - start_time
    
    # Get final stats
    stats = trainer.get_optimization_stats()
    
    benchmark_results = {
        'total_time': total_time,
        'avg_time_per_step': total_time / iterations,
        'throughput_steps_per_sec': iterations / total_time,
        'cache_hit_rate': stats['smart_cache']['hit_rate'],
        'final_learning_rate': stats['learning_rate_scheduler']['current_learning_rate']
    }
    
    logger.info(f"Optimization benchmark results: {benchmark_results}")
    return benchmark_results 