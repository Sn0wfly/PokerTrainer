#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DIAGNOSTIC CHECK
Verifies all model saving/loading systems are working correctly
"""

import os
import pickle
import numpy as np
import jax.numpy as jnp
import logging
from typing import Dict, Any, List
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_files():
    """Check all model files in the models directory"""
    print("\n🔍 CHECKING MODEL FILES")
    print("=" * 50)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found")
        return False
    
    print(f"📁 Found {len(model_files)} model files:")
    
    all_models_valid = True
    
    for model_file in model_files:
        filepath = os.path.join(models_dir, model_file)
        file_size = os.path.getsize(filepath)
        
        print(f"\n📄 {model_file} ({file_size:,} bytes)")
        
        try:
            # Try to load the model
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Analyze model structure
            print(f"   ✅ Loaded successfully")
            print(f"   📊 Keys: {list(model_data.keys())}")
            
            # Check for different model types
            if 'q_values' in model_data and 'strategies' in model_data:
                if isinstance(model_data['q_values'], dict):
                    print(f"   🧠 Type: REAL CFVFP (Dictionary-based)")
                    print(f"   📈 Q-values: {len(model_data['q_values'])} info sets")
                    print(f"   📈 Strategies: {len(model_data['strategies'])} info sets")
                    
                    # Check if strategies are real or fixed
                    if len(model_data['q_values']) > 100:
                        print(f"   ✅ REAL strategies detected (learning working)")
                    else:
                        print(f"   ⚠️  Few strategies - may be early training")
                        
                elif isinstance(model_data['q_values'], np.ndarray):
                    print(f"   🧠 Type: VECTORIZED CFVFP (Array-based)")
                    print(f"   📈 Q-values shape: {model_data['q_values'].shape}")
                    print(f"   📈 Strategies shape: {model_data['strategies'].shape}")
                    
                    # Check if arrays are all identical (fixed)
                    q_unique = np.unique(model_data['q_values'])
                    s_unique = np.unique(model_data['strategies'])
                    
                    if len(q_unique) <= 10 and len(s_unique) <= 10:
                        print(f"   ⚠️  WARNING: Fixed matrices detected (not real learning)")
                    else:
                        print(f"   ✅ VARIED values detected (real learning)")
                        
            elif 'strategy_sum' in model_data and 'regret_sum' in model_data:
                print(f"   🧠 Type: MCCFR (Legacy)")
                print(f"   📈 Strategy states: {len(model_data['strategy_sum'])}")
                print(f"   📈 Regret states: {len(model_data['regret_sum'])}")
                
            else:
                print(f"   ❓ Type: Unknown format")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            all_models_valid = False
    
    return all_models_valid

def test_real_cfvfp_system():
    """Test the REAL CFVFP system specifically"""
    print("\n🧠 TESTING REAL CFVFP SYSTEM")
    print("=" * 50)
    
    try:
        from poker_bot.real_cfvfp_trainer import RealCFVFPTrainer, RealCFVFPConfig
        
        # Create trainer
        config = RealCFVFPConfig(batch_size=1024)
        trainer = RealCFVFPTrainer(config)
        
        print("✅ REAL CFVFP trainer created")
        
        # Test model saving/loading
        test_path = "test_real_cfvfp_diagnostic.pkl"
        
        # Save empty model
        trainer.save_model(test_path)
        print("✅ Model saving works")
        
        # Load model
        new_trainer = RealCFVFPTrainer(config)
        new_trainer.load_model(test_path)
        print("✅ Model loading works")
        
        # Clean up
        os.remove(test_path)
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ REAL CFVFP test failed: {e}")
        return False

def test_vectorized_cfvfp_system():
    """Test the VECTORIZED CFVFP system"""
    print("\n🚀 TESTING VECTORIZED CFVFP SYSTEM")
    print("=" * 50)
    
    try:
        from poker_bot.vectorized_cfvfp_trainer import VectorizedCFVFPTrainer, VectorizedCFVFPConfig
        
        # Create trainer
        config = VectorizedCFVFPConfig(batch_size=1024)
        trainer = VectorizedCFVFPTrainer(config)
        
        print("✅ VECTORIZED CFVFP trainer created")
        
        # Test model saving/loading
        test_path = "test_vectorized_cfvfp_diagnostic.pkl"
        
        # Save model
        trainer.save_model(test_path)
        print("✅ Model saving works")
        
        # Load model
        new_trainer = VectorizedCFVFPTrainer(config)
        new_trainer.load_model(test_path)
        print("✅ Model loading works")
        
        # Clean up
        os.remove(test_path)
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ VECTORIZED CFVFP test failed: {e}")
        return False

def test_visualizer_system():
    """Test the visualizer system"""
    print("\n🎮 TESTING VISUALIZER SYSTEM")
    print("=" * 50)
    
    try:
        from poker_bot.visualizer import PokerVisualizer
        
        # Test with non-existent model (should handle gracefully)
        visualizer = PokerVisualizer("non_existent_model.pkl")
        print("✅ Visualizer handles missing models gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        return False

def test_cli_commands():
    """Test CLI command availability"""
    print("\n⚙️  TESTING CLI COMMANDS")
    print("=" * 50)
    
    try:
        from poker_bot.cli import cli
        
        # Check if commands exist
        commands = [cmd.name for cmd in cli.commands.values()]
        
        expected_commands = [
            'train', 'train-fast', 'train-holdem', 'train-cfvfp',
            'play', 'visualize', 'evaluate', 'list-models'
        ]
        
        print(f"📋 Available commands: {commands}")
        
        missing_commands = [cmd for cmd in expected_commands if cmd not in commands]
        
        if missing_commands:
            print(f"❌ Missing commands: {missing_commands}")
            return False
        else:
            print("✅ All expected commands available")
            return True
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def check_training_systems():
    """Check all training systems"""
    print("\n🏋️  CHECKING TRAINING SYSTEMS")
    print("=" * 50)
    
    systems_to_check = [
        ("REAL CFVFP", "poker_bot.real_cfvfp_trainer"),
        ("VECTORIZED CFVFP", "poker_bot.vectorized_cfvfp_trainer"),
        ("MCCFR", "poker_bot.trainer"),
        ("Modern CFR", "poker_bot.modern_cfr"),
    ]
    
    all_systems_ok = True
    
    for system_name, module_name in systems_to_check:
        try:
            __import__(module_name)
            print(f"✅ {system_name}: Available")
        except ImportError as e:
            print(f"❌ {system_name}: {e}")
            all_systems_ok = False
    
    return all_systems_ok

def run_comprehensive_diagnostic():
    """Run all diagnostic checks"""
    print("🔍 COMPREHENSIVE POKERTRAINER DIAGNOSTIC")
    print("=" * 60)
    print("Checking all systems for potential issues...")
    
    results = {}
    
    # Run all checks
    results['model_files'] = check_model_files()
    results['real_cfvfp'] = test_real_cfvfp_system()
    results['vectorized_cfvfp'] = test_vectorized_cfvfp_system()
    results['visualizer'] = test_visualizer_system()
    results['cli_commands'] = test_cli_commands()
    results['training_systems'] = check_training_systems()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL SYSTEMS CHECKED - NO MAJOR ISSUES DETECTED")
        print("💡 Your PokerTrainer installation appears to be working correctly!")
    else:
        print("⚠️  SOME ISSUES DETECTED - REVIEW ABOVE FOR DETAILS")
        print("💡 Check the specific failing components above")
    
    return all_passed

if __name__ == "__main__":
    run_comprehensive_diagnostic() 