#!/usr/bin/env python3
"""
CuPy CFR Test - Maximum GPU utilization for RTX 3090
Compare vs JAX performance for CFR operations
"""

import cupy as cp
import numpy as np
import time
from contextlib import contextmanager
import gc

@contextmanager
def gpu_timer(name):
    """GPU timer context manager"""
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    print(f"\n[{name}] Starting...")
    start_event.record()
    start_time = time.time()
    
    yield
    
    end_event.record()
    end_event.synchronize()
    
    gpu_time = cp.cuda.get_elapsed_time(start_event, end_event)
    wall_time = time.time() - start_time
    
    print(f"[{name}] GPU time: {gpu_time:.2f}ms, Wall time: {wall_time:.2f}s")
    print(f"[{name}] Memory usage: {cp.get_default_memory_pool().used_bytes() / 1024**3:.2f} GB")

def cupy_massive_cfr_simulation(batch_size=1000, matrix_size=3000):
    """
    Massive CFR simulation using CuPy - designed to saturate RTX 3090
    """
    with gpu_timer("CuPy CFR Simulation"):
        # MASSIVE strategy matrices (batch_size x matrix_size x matrix_size)
        strategy_matrices = cp.random.random((batch_size, matrix_size, matrix_size), dtype=cp.float32)
        regret_matrices = cp.random.random((batch_size, matrix_size, matrix_size), dtype=cp.float32)
        
        # CFR operations - 5 rounds of massive matrix multiplications
        for round_idx in range(5):
            print(f"CFR Round {round_idx + 1}/5...")
            
            # Massive matrix operations (should saturate GPU)
            updated_strategies = cp.matmul(strategy_matrices, regret_matrices)
            normalized_strategies = updated_strategies / cp.sum(updated_strategies, axis=2, keepdims=True)
            
            # Regret updates with massive reductions
            regret_updates = cp.sum(normalized_strategies * strategy_matrices, axis=2)
            regret_matrices = cp.maximum(regret_matrices + regret_updates[:, :, cp.newaxis], 0)
            
            # Strategy updates
            strategy_matrices = normalized_strategies
        
        # Final CFR calculation
        final_utilities = cp.sum(strategy_matrices * regret_matrices, axis=(1, 2))
        
        return cp.mean(final_utilities)

def cupy_poker_hand_evaluation(num_hands=100000):
    """
    Vectorized poker hand evaluation using CuPy
    """
    with gpu_timer("CuPy Poker Hand Evaluation"):
        # Generate random hands (5 cards each)
        hands = cp.random.randint(0, 52, size=(num_hands, 5), dtype=cp.int32)
        
        # Convert to suits and ranks
        suits = hands % 4
        ranks = hands // 4
        
        # Vectorized hand strength calculation
        # Straight flushes, flushes, straights, etc.
        hand_strengths = cp.zeros(num_hands, dtype=cp.float32)
        
        # Check for flushes (all same suit)
        flush_mask = cp.all(suits == suits[:, 0:1], axis=1)
        hand_strengths[flush_mask] += 5.0
        
        # Check for straights (consecutive ranks)
        sorted_ranks = cp.sort(ranks, axis=1)
        straight_mask = cp.all(cp.diff(sorted_ranks, axis=1) == 1, axis=1)
        hand_strengths[straight_mask] += 4.0
        
        # Pairs, trips, etc.
        for i in range(13):  # 13 ranks
            rank_counts = cp.sum(ranks == i, axis=1)
            hand_strengths += rank_counts * rank_counts * 0.5
        
        return cp.mean(hand_strengths)

def performance_comparison():
    """
    Compare different batch sizes and matrix sizes
    """
    print("=== CuPy CFR Performance Test ===")
    
    # Get GPU device info
    device = cp.cuda.runtime.getDevice()
    device_props = cp.cuda.runtime.getDeviceProperties(device)
    print("GPU:", device_props['name'].decode('utf-8'))
    
    # Get memory info
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print("Memory:", total_mem / 1024**3, "GB total,", free_mem / 1024**3, "GB free")
    
    # Test different configurations (optimized for RTX 3090 24GB memory)
    configs = [
        (100, 1000),    # Small batch, small matrix (~400MB)
        (200, 1000),    # Medium batch, small matrix (~800MB)
        (100, 1500),    # Small batch, medium matrix (~900MB)
        (150, 1500),    # Medium batch, medium matrix (~1.3GB)
        (200, 1500),    # Large batch, medium matrix (~1.8GB)
        (100, 2000),    # Small batch, large matrix (~1.6GB)
    ]
    
    for batch_size, matrix_size in configs:
        print(f"\n--- Testing batch_size={batch_size}, matrix_size={matrix_size} ---")
        
        try:
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            # Run CFR simulation
            result = cupy_massive_cfr_simulation(batch_size, matrix_size)
            print(f"CFR Result: {result:.4f}")
            
            # Run poker evaluation
            poker_result = cupy_poker_hand_evaluation(batch_size * 100)
            print(f"Poker Result: {poker_result:.4f}")
            
        except cp.cuda.memory.OutOfMemoryError:
            print("OUT OF MEMORY - batch size too large")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

if __name__ == "__main__":
    # Check CuPy installation
    print("CuPy version:", cp.__version__)
    print("CUDA version:", cp.cuda.runtime.runtimeGetVersion())
    
    performance_comparison() 