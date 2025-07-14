#!/usr/bin/env python3
"""
🔍 VAST.AI GPU DIAGNOSTIC
Ejecutar en vast.ai para diagnosticar problemas de GPU
"""

import os
import sys
import subprocess
import time
import traceback
from typing import Dict, Any, List

def check_system_gpu():
    """Check if GPU is available at system level"""
    print("🔍 CHECKING SYSTEM GPU")
    print("=" * 40)
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi works")
            print("📊 GPU Info:")
            print(result.stdout[:800] + "..." if len(result.stdout) > 800 else result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no NVIDIA drivers installed")
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    print("\n🔍 CHECKING CUDA INSTALLATION")
    print("=" * 40)
    
    try:
        # Check nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvcc works")
            print(f"📊 CUDA version: {result.stdout.split('release ')[1].split(',')[0]}")
            return True
        else:
            print("❌ nvcc failed")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA not installed")
        return False

def check_jax_gpu():
    """Check JAX GPU detection"""
    print("\n🔍 CHECKING JAX GPU DETECTION")
    print("=" * 40)
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"📦 JAX version: {jax.__version__}")
        print(f"📦 JAXlib version: {jax.lib.__version__}")
        
        # Check devices
        devices = jax.devices()
        print(f"📱 All devices: {devices}")
        
        # Check GPU devices specifically
        try:
            gpu_devices = jax.devices('gpu')
            print(f"🎮 GPU devices: {gpu_devices}")
            
            if len(gpu_devices) > 0:
                print("✅ GPU devices detected by JAX")
                
                # Test GPU operations
                print("🧪 Testing GPU operations...")
                with jax.default_device(gpu_devices[0]):
                    x = jnp.array([1.0, 2.0, 3.0])
                    y = jnp.array([4.0, 5.0, 6.0])
                    z = x + y
                    print(f"✅ GPU operations work: {z}")
                    print(f"✅ Device used: {x.device}")
                
                return True
            else:
                print("❌ No GPU devices detected by JAX")
                return False
                
        except Exception as e:
            print(f"❌ Error checking GPU devices: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX GPU check failed: {e}")
        return False

def check_environment_variables():
    """Check environment variables that affect GPU usage"""
    print("\n🔍 CHECKING ENVIRONMENT VARIABLES")
    print("=" * 40)
    
    relevant_vars = [
        'CUDA_VISIBLE_DEVICES',
        'XLA_FLAGS',
        'JAX_PLATFORMS',
        'JAX_ENABLE_X64',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_ALLOCATOR',
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"📋 {var}: {value}")
    
    # Check if any problematic flags are set
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if 'cpu' in xla_flags.lower():
        print("⚠️  WARNING: XLA_FLAGS contains 'cpu' - this may force CPU usage")
    
    return True

def check_poker_bot_gpu_usage():
    """Check if poker bot is actually using GPU"""
    print("\n🔍 CHECKING POKER BOT GPU USAGE")
    print("=" * 40)
    
    try:
        # Import poker bot modules
        from poker_bot.gpu_config import get_device_info, init_gpu_environment
        
        # Get device info
        device_info = get_device_info()
        print(f"📊 Device info: {device_info}")
        
        # Initialize GPU environment
        env_config = init_gpu_environment()
        print(f"📊 Environment config: {env_config}")
        
        # Test with a simple poker operation
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        
        print("🧪 Testing poker bot GPU usage...")
        
        # Create a simple test
        rng_key = jr.PRNGKey(42)
        x = jnp.array([1.0, 2.0, 3.0])
        
        # Check which device is being used
        print(f"📱 Default device: {jax.default_backend()}")
        print(f"📱 X device: {x.device}")
        
        # Test with poker bot modules
        try:
            from poker_bot.simple_nlhe_engine import simple_nlhe_batch
            
            rng_keys = jr.split(rng_key, 5)
            results = simple_nlhe_batch(rng_keys)
            
            print(f"✅ Poker bot operations work")
            print(f"📊 Results device: {results['hole_cards'].device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Poker bot GPU test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Poker bot GPU check failed: {e}")
        traceback.print_exc()
        return False

def check_jax_installation():
    """Check JAX installation details"""
    print("\n🔍 CHECKING JAX INSTALLATION")
    print("=" * 40)
    
    try:
        import jax
        import jaxlib
        
        print(f"📦 JAX version: {jax.__version__}")
        print(f"📦 JAXlib version: {jaxlib.__version__}")
        
        # Check if JAX was installed with CUDA support
        try:
            import jaxlib.xla_extension
            print("✅ JAXlib XLA extension available")
        except ImportError:
            print("❌ JAXlib XLA extension not available")
        
        # Check JAX installation path
        print(f"📁 JAX path: {jax.__file__}")
        print(f"📁 JAXlib path: {jaxlib.__file__}")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX installation check failed: {e}")
        return False

def check_recent_changes():
    """Check for recent changes that might affect GPU usage"""
    print("\n🔍 CHECKING FOR RECENT CHANGES")
    print("=" * 40)
    
    # Check if gpu_config.py is being imported
    try:
        import poker_bot.gpu_config
        print("✅ gpu_config.py is being imported")
        
        # Check if GPU environment is initialized
        if hasattr(poker_bot.gpu_config, 'init_gpu_environment'):
            print("✅ GPU environment initialization function exists")
        else:
            print("❌ GPU environment initialization function missing")
            
    except Exception as e:
        print(f"❌ Error importing gpu_config: {e}")
    
    # Check for any environment variable overrides
    print("\n📋 Checking for environment overrides...")
    
    # Look for any scripts that might be setting CPU-only flags
    problematic_patterns = [
        'XLA_FLAGS.*cpu',
        'JAX_PLATFORMS.*cpu',
        'CUDA_VISIBLE_DEVICES.*-1',
    ]
    
    for pattern in problematic_patterns:
        print(f"🔍 Checking for pattern: {pattern}")
    
    return True

def run_comprehensive_diagnostic():
    """Run all diagnostic checks"""
    print("🔍 VAST.AI GPU DIAGNOSTIC")
    print("=" * 60)
    print("Checking why GPU is not being used on vast.ai...")
    
    results = {}
    
    # Run all checks
    results['system_gpu'] = check_system_gpu()
    results['cuda_installation'] = check_cuda_installation()
    results['jax_installation'] = check_jax_installation()
    results['jax_gpu'] = check_jax_gpu()
    results['environment_vars'] = check_environment_variables()
    results['poker_bot_gpu'] = check_poker_bot_gpu_usage()
    results['recent_changes'] = check_recent_changes()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    if all_passed:
        print("\n🎉 All checks passed! GPU should be working.")
        print("If GPU is still not being used, check:")
        print("1. Recent code changes that might override GPU settings")
        print("2. Environment variables set in your shell")
        print("3. JAX version compatibility issues")
    else:
        print("\n❌ Some checks failed. Common solutions:")
        print("1. Install/update NVIDIA drivers")
        print("2. Install/update CUDA toolkit")
        print("3. Reinstall JAX with CUDA support")
        print("4. Check environment variables")
        print("5. Run: pip install --upgrade jax jaxlib")
        print("\n🔧 For vast.ai specific fixes:")
        print("1. Run: bash install_jax_cuda_working.sh")
        print("2. Check if CUDA_VISIBLE_DEVICES is set correctly")
        print("3. Restart the Jupyter kernel or Python session")
    
    return results

if __name__ == "__main__":
    run_comprehensive_diagnostic() 