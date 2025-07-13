"""
🎯 PokerTrainer - GPU-Native Poker AI

A high-performance poker bot using JAX + MCCFR for GPU acceleration.
"""

__version__ = "0.1.0"
__author__ = "PokerTrainer Team"

from .engine import PokerEngine
from .evaluator import HandEvaluator
from .trainer import SimpleMCCFRTrainer, MCCFRConfig, create_trainer
from .bot import PokerBot

__all__ = [
    "PokerEngine",
    "HandEvaluator", 
    "SimpleMCCFRTrainer",
    "MCCFRConfig",
    "create_trainer",
    "PokerBot",
] 