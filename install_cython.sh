#!/bin/bash
# 🚀 INSTALL_CYTHON.SH - Automated Cython setup for Definitive Hybrid Trainer
# This script installs dependencies and compiles the Cython module

echo "🚀 Installing Cython dependencies..."
pip install cython>=3.0.0 setuptools>=65.0.0

echo "🚀 Compiling Cython module..."
python setup_cython.py build_ext --inplace

echo "✅ Checking compiled module..."
if [ -f "poker_bot/fast_hasher.cpython-*.so" ]; then
    echo "✅ Cython module compiled successfully!"
    ls -la poker_bot/fast_hasher*.so
else
    echo "❌ Cython compilation failed!"
    exit 1
fi

echo "🎉 Cython setup completed successfully!"
echo "🚀 You can now run: python -m poker_bot.cli train-definitive-hybrid" 