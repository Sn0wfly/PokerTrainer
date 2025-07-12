"""
🎯 PokerBot - GPU-Native Poker AI

A high-performance poker bot using JAX + CFRX for GPU acceleration.
"""

__version__ = "0.1.0"
__author__ = "PokerBot Team"

from .engine import PokerEngine
from .evaluator import HandEvaluator
from .trainer import MCCFRTrainer
from .bot import PokerBot

__all__ = [
    "PokerEngine",
    "HandEvaluator", 
    "MCCFRTrainer",
    "PokerBot",
] 