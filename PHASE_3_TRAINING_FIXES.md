# PHASE 3: TRAINING FIXES FOR VAST.AI

## 🎯 **CURRENT STATUS**
- **Project**: Ready for final training phase
- **Issues**: 2 minor configuration problems in vast.ai
- **Expected fix time**: 30 minutes
- **Training ready**: After fixes applied

## ⚠️ **PROBLEM 1: CUDA cuSPARSE Error**

### **Error:**
```
RuntimeError: Unable to load cuSPARSE. Is it installed?
```

### **Solution:**
```bash
# In vast.ai Jupyter terminal with (poker_env) activated:
pip install --upgrade jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install cusparse-cu12 cusolver-cu12 cufft-cu12
```

### **Verify fix:**
```bash
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

## ⚠️ **PROBLEM 2: CLI Command Correction**

### **Incorrect command used:**
```bash
# ❌ This has wrong options:
python -m poker_bot.cli train --algorithm pdcfr_plus --iterations 100000 --batch-size 8192 --save-interval 1000 --gpu
```

### **Correct command:**
```bash
# ✅ Correct options:
python -m poker_bot.cli train --iterations 100000 --batch-size 8192 --save-interval 1000 --gpu
```

### **Available train options:**
- `--iterations`: Number of training iterations (default: 100000)
- `--batch-size`: Batch size for training (default: 1024)
- `--players`: Number of players (default: 2)
- `--learning-rate`: Learning rate (default: 0.1)
- `--exploration`: Exploration rate (default: 0.1)
- `--save-interval`: Save model every N iterations (default: 1000)
- `--log-interval`: Log progress every N iterations (default: 100)
- `--save-path`: Path to save trained model (default: 'models/mccfr_model.pkl')
- `--gpu/--no-gpu`: Use GPU acceleration (default: True)

## 🚀 **COMPLETE TRAINING STARTUP SEQUENCE**

### **Step 1: Fix CUDA (if needed)**
```bash
# Only if cuSPARSE error appears:
pip install --upgrade jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install cusparse-cu12 cusolver-cu12 cufft-cu12
```

### **Step 2: Start training**
```bash
# In vast.ai Jupyter terminal with (poker_env):
cd /workspace/PokerTrainer

# Start training in background:
nohup python -m poker_bot.cli train \
  --iterations 100000 \
  --batch-size 8192 \
  --save-interval 1000 \
  --gpu > training.log 2>&1 &
```

### **Step 3: Monitor progress**
```bash
# Real-time log monitoring:
tail -f training.log

# Check process:
ps aux | grep python

# Monitor GPU usage (new terminal):
watch -n 5 nvidia-smi

# Check saved models:
ls -la models/
```

## 📊 **EXPECTED TRAINING OUTPUT**

### **First few lines should show:**
```
Starting MCCFR training...
Using GPU: GpuDevice(id=0)
Batch size: 8192
Iterations: 100000
Save interval: 1000

Iteration 1/100000: avg_utility=-0.0234, time=2.42ms
Iteration 2/100000: avg_utility=-0.0198, time=2.45ms
Iteration 3/100000: avg_utility=-0.0165, time=2.41ms
...
```

### **Performance indicators:**
- **Time per iteration**: ~2-4ms (good performance)
- **GPU utilization**: 70-80% (nvidia-smi)
- **Memory usage**: ~18GB VRAM
- **Convergence**: avg_utility approaching 0

## 🔧 **TROUBLESHOOTING**

### **If training stops/fails:**
```bash
# Check if process is running:
ps aux | grep python

# Check last error:
tail -50 training.log

# Restart training:
nohup python -m poker_bot.cli train \
  --iterations 100000 \
  --batch-size 8192 \
  --save-interval 1000 \
  --gpu > training.log 2>&1 &
```

### **If GPU not detected:**
```bash
# Check CUDA installation:
nvidia-smi
python -c "import jax; print(jax.devices())"

# Use CPU fallback:
python -m poker_bot.cli train \
  --iterations 100000 \
  --batch-size 1024 \
  --save-interval 1000 \
  --no-gpu > training.log 2>&1 &
```

## 🎯 **SUCCESS METRICS**

### **Training is successful when:**
- ✅ Process runs without errors
- ✅ GPU utilization 70-80%
- ✅ Iteration time ~2-4ms
- ✅ Checkpoints saved every 1000 iterations
- ✅ avg_utility converging toward 0

### **Training completion (~4-8 hours):**
- ✅ 100,000 iterations completed
- ✅ Final model saved to `models/mccfr_model.pkl`
- ✅ Convergence achieved (stable avg_utility)
- ✅ Ready for evaluation and play testing

## 🏆 **NEXT STEPS AFTER TRAINING**

### **Model evaluation:**
```bash
python -m poker_bot.cli evaluate --model models/mccfr_model.pkl
```

### **Test gameplay:**
```bash
python -m poker_bot.cli play --model models/mccfr_model.pkl --hands 10
```

### **Congratulations! Phase 3 complete! 🎉**

---

**Created**: 2025-01-13  
**Project**: PokerTrainer Phase 3 - Texas Hold'em Training  
**Status**: Ready for final training phase after minor fixes 