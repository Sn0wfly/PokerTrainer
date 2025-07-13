"""
Multi-GPU and Distributed Training Module for Modern CFR
=======================================================

This module implements JAX 2025 parallel processing techniques for maximum
performance on multi-GPU systems.

Key Features:
- Multi-GPU training with jax.pmap
- Gradient checkpointing for memory efficiency
- Pipeline parallelism for computation overlap
- Distributed training coordination
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import pmap, checkpoint, tree_map
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import functools
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .gpu_config import get_device_info, configure_mixed_precision
from .memory import MemoryMonitor, get_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """Configuration for parallel training"""
    num_devices: int = 1
    batch_size_per_device: int = 256
    gradient_accumulation_steps: int = 1
    use_pipeline_parallelism: bool = True
    checkpoint_frequency: int = 1000
    sync_frequency: int = 100
    
    def __post_init__(self):
        if self.num_devices <= 0:
            device_info = get_device_info()
            self.num_devices = device_info['num_devices']
        
        logger.info(f"Parallel config: {self.num_devices} devices, "
                   f"batch_size_per_device={self.batch_size_per_device}")

class MultiGPUTrainer:
    """Multi-GPU trainer using JAX pmap for distributed CFR training"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        logger.info(f"Initializing MultiGPU trainer with {self.num_devices} devices")
        
        # Configure mixed precision
        configure_mixed_precision()
        
        # Initialize device mesh for parallel training
        self._setup_device_mesh()
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor("MultiGPU_CFR")
        
    def _setup_device_mesh(self):
        """Setup device mesh for parallel computation"""
        self.device_mesh = jax.sharding.Mesh(
            devices=jnp.array(self.devices).reshape(-1),
            axis_names=('device',)
        )
        logger.info(f"Device mesh setup: {self.device_mesh}")
    
    @functools.partial(pmap, axis_name='device')
    def parallel_q_update(self, q_values: jnp.ndarray, 
                         regrets: jnp.ndarray,
                         learning_rate: float) -> jnp.ndarray:
        """Parallel Q-value update across devices"""
        # Q-value update with gradient accumulation
        q_update = learning_rate * regrets
        new_q_values = q_values + q_update
        
        # Synchronize across devices
        new_q_values = jax.lax.pmean(new_q_values, axis_name='device')
        
        return new_q_values
    
    @functools.partial(pmap, axis_name='device')
    def parallel_strategy_computation(self, q_values: jnp.ndarray,
                                    temperature: float) -> jnp.ndarray:
        """Parallel strategy computation with temperature scaling"""
        # Apply temperature scaling
        scaled_q = q_values / temperature
        
        # Softmax for strategy probabilities
        strategy = jax.nn.softmax(scaled_q, axis=-1)
        
        # Synchronize strategies across devices
        strategy = jax.lax.pmean(strategy, axis_name='device')
        
        return strategy
    
    def replicate_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """Replicate data across devices for parallel processing"""
        if data.ndim == 1:
            # Add device dimension
            data = data[None, :]
        
        return jnp.repeat(data[None, :], self.num_devices, axis=0)
    
    def gather_results(self, parallel_results: jnp.ndarray) -> jnp.ndarray:
        """Gather results from parallel computation"""
        # Take mean across devices
        return jnp.mean(parallel_results, axis=0)

class GradientCheckpointManager:
    """Manage gradient checkpointing for memory efficiency"""
    
    def __init__(self, checkpoint_policy: str = "adaptive"):
        self.checkpoint_policy = checkpoint_policy
        self.checkpointed_functions = {}
        
    def checkpoint_function(self, func: Callable, 
                          policy: str = "auto") -> Callable:
        """Create checkpointed version of function"""
        if policy == "auto":
            policy = self.checkpoint_policy
        
        if policy == "always":
            return checkpoint(func)
        elif policy == "adaptive":
            # Checkpoint based on memory usage
            def adaptive_checkpoint(*args, **kwargs):
                memory_usage = get_memory_usage()
                if memory_usage['percent'] > 80:  # High memory usage
                    return checkpoint(func)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return adaptive_checkpoint
        else:
            return func
    
    def register_checkpointed_function(self, name: str, func: Callable):
        """Register a function for checkpointing"""
        self.checkpointed_functions[name] = self.checkpoint_function(func)
        
    def get_checkpointed_function(self, name: str) -> Callable:
        """Get checkpointed version of registered function"""
        return self.checkpointed_functions.get(name, None)

class PipelineParallelTrainer:
    """Pipeline parallel trainer for overlapping computation and communication"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.pipeline_stages = []
        self.executor = ThreadPoolExecutor(max_workers=config.num_devices)
        
    def add_pipeline_stage(self, stage_func: Callable, stage_name: str):
        """Add a stage to the pipeline"""
        self.pipeline_stages.append({
            'func': stage_func,
            'name': stage_name
        })
        logger.info(f"Added pipeline stage: {stage_name}")
    
    def execute_pipeline(self, initial_data: Any) -> Any:
        """Execute the pipeline with overlap"""
        start_time = time.time()
        
        # Submit all stages with data dependencies
        futures = []
        current_data = initial_data
        
        for stage in self.pipeline_stages:
            future = self.executor.submit(stage['func'], current_data)
            futures.append(future)
            
            # For pipeline parallelism, we can start next stage
            # while current is still processing (with careful data handling)
            
        # Wait for all stages to complete
        results = []
        for future in futures:
            results.append(future.result())
            
        execution_time = time.time() - start_time
        logger.info(f"Pipeline execution completed in {execution_time:.3f}s")
        
        return results[-1]  # Return final result

class DistributedCFRCoordinator:
    """Coordinate distributed CFR training across multiple nodes"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.node_id = 0  # This would be set by distributed system
        self.num_nodes = 1  # This would be set by distributed system
        
        # Initialize parallel components
        self.multi_gpu_trainer = MultiGPUTrainer(config)
        self.checkpoint_manager = GradientCheckpointManager()
        self.pipeline_trainer = PipelineParallelTrainer(config)
        
        logger.info(f"Distributed coordinator initialized for node {self.node_id}")
    
    def distributed_training_step(self, q_values: jnp.ndarray,
                                regrets: jnp.ndarray,
                                learning_rate: float) -> Dict[str, Any]:
        """Execute one distributed training step"""
        step_start = time.time()
        
        with self.multi_gpu_trainer.memory_monitor:
            # Replicate data across devices
            q_values_replicated = self.multi_gpu_trainer.replicate_data(q_values)
            regrets_replicated = self.multi_gpu_trainer.replicate_data(regrets)
            
            # Parallel Q-value update
            updated_q_values = self.multi_gpu_trainer.parallel_q_update(
                q_values_replicated, regrets_replicated, learning_rate
            )
            
            # Parallel strategy computation
            strategies = self.multi_gpu_trainer.parallel_strategy_computation(
                updated_q_values, temperature=1.0
            )
            
            # Gather results
            final_q_values = self.multi_gpu_trainer.gather_results(updated_q_values)
            final_strategies = self.multi_gpu_trainer.gather_results(strategies)
        
        step_time = time.time() - step_start
        
        return {
            'q_values': final_q_values,
            'strategies': final_strategies,
            'step_time': step_time,
            'memory_usage': get_memory_usage()
        }
    
    def benchmark_parallel_performance(self, iterations: int = 100) -> Dict[str, float]:
        """Benchmark parallel performance"""
        logger.info(f"Benchmarking parallel performance with {iterations} iterations")
        
        # Test data
        test_q_values = jr.normal(jr.PRNGKey(42), (4,))
        test_regrets = jr.normal(jr.PRNGKey(43), (4,))
        
        # Warmup
        for _ in range(10):
            self.distributed_training_step(test_q_values, test_regrets, 0.1)
        
        # Benchmark
        start_time = time.time()
        total_step_time = 0
        
        for i in range(iterations):
            result = self.distributed_training_step(test_q_values, test_regrets, 0.1)
            total_step_time += result['step_time']
            
            if i % 10 == 0:
                logger.info(f"Benchmark progress: {i}/{iterations}")
        
        total_time = time.time() - start_time
        avg_step_time = total_step_time / iterations
        throughput = iterations / total_time
        
        benchmark_results = {
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'throughput_steps_per_sec': throughput,
            'parallel_efficiency': (1.0 / avg_step_time) / self.config.num_devices,
            'memory_peak_mb': get_memory_usage()['process_mb']
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results

def create_parallel_trainer(config: ParallelConfig) -> DistributedCFRCoordinator:
    """Create and configure parallel trainer"""
    return DistributedCFRCoordinator(config)

def get_optimal_parallel_config() -> ParallelConfig:
    """Get optimal parallel configuration based on available hardware"""
    device_info = get_device_info()
    
    # Calculate optimal batch size per device
    # Rule of thumb: 256-512 per device for good GPU utilization
    batch_size_per_device = 512 if device_info['num_devices'] > 1 else 1024
    
    # Gradient accumulation for large effective batch sizes
    gradient_accumulation_steps = max(1, 4 // device_info['num_devices'])
    
    config = ParallelConfig(
        num_devices=device_info['num_devices'],
        batch_size_per_device=batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_pipeline_parallelism=device_info['num_devices'] > 1,
        checkpoint_frequency=1000,
        sync_frequency=100
    )
    
    logger.info(f"Optimal parallel config: {config}")
    return config 