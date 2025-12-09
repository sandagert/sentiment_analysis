#!/bin/bash

# ============================================================================
# PART 1: DOWNLOADS - No GPU needed
# Run this while waiting for GPU allocation
# ============================================================================

set -e

echo "========================================================================"
echo "PART 1: DOWNLOADING MODELS"
echo "Time estimate: ~35 minutes"
echo "GPU: NOT NEEDED"
echo "========================================================================"
echo ""

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

echo "STEP 1: Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# ============================================================================
# DOWNLOAD LLAMA MODELS
# ============================================================================

echo "STEP 2: Downloading LLaMA models..."

echo "[1/2] Downloading LLaMA 7B (~10 min, ~14GB)..."
if [ ! -d "models/llama_7b/final" ]; then
    python scripts/setup_llama_7b.py
    echo "✓ LLaMA 7B downloaded"
else
    echo "✓ LLaMA 7B already exists"
fi
echo ""

echo "[2/2] Downloading LLaMA 13B (~25 min, ~26GB)..."
if [ ! -d "models/llama_13b/final" ]; then
    python scripts/setup_llama_13b.py
    echo "✓ LLaMA 13B downloaded"
else
    echo "✓ LLaMA 13B already exists"
fi
echo ""

# ============================================================================
# COMPLETE
# ============================================================================

echo "========================================================================"
echo "PART 1 COMPLETE - DOWNLOADS DONE!"
echo "========================================================================"
echo ""
echo "Downloaded:"
echo "  ✓ LLaMA 7B (14GB)"
echo "  ✓ LLaMA 13B (26GB)"
echo ""
echo "Next: Request GPU and run 2_train_and_evaluate.sh"
echo "========================================================================"