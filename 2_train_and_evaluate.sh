#!/bin/bash

# ============================================================================
# PART 2: TRAINING AND EVALUATIONS - GPU REQUIRED
# Optimized for NVIDIA V100/A100
# ============================================================================

set -e

echo "========================================================================"
echo "PART 2: TRAINING AND EVALUATIONS"
echo "GPU: REQUIRED (V100 or A100 recommended)"
echo "========================================================================"
echo ""
echo "Time estimates:"
echo "  With A100: ~10-14 hours"
echo "  With V100: ~15-20 hours"
echo "  With CPU:  ~46-60 hours (not recommended)"
echo "========================================================================"
echo ""

# Detect GPU
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f'✓ GPU detected: {gpu_name}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  No GPU detected - will use CPU (very slow)')
"
echo ""

# Check that LLaMA models are downloaded
if [ ! -d "models/llama_7b/final" ] || [ ! -d "models/llama_13b/final" ]; then
    echo "❌ ERROR: LLaMA models not found. Run 1_download_models.sh first!"
    exit 1
fi

echo "✓ LLaMA models found"
echo ""

# ============================================================================
# STEP 1: TRAIN BERT AND ROBERTA
# ============================================================================

echo "STEP 1: Training BERT and RoBERTa..."
echo "  V100: ~45-60 min each | A100: ~30-40 min each"
echo ""

echo "[1/2] Training BERT..."
if [ ! -d "models/bert/final" ]; then
    python scripts/train_bert.py
    echo "✓ BERT trained"
else
    echo "✓ BERT already trained"
fi
echo ""

echo "[2/2] Training RoBERTa..."
if [ ! -d "models/roberta/final" ]; then
    python scripts/train_roberta.py
    echo "✓ RoBERTa trained"
else
    echo "✓ RoBERTa already trained"
fi
echo ""

# ============================================================================
# STEP 2: BASELINE EVALUATIONS
# ============================================================================

echo "STEP 2: Running baseline evaluations..."
echo "  Time: ~5-10 min total"
echo ""

echo "[1/2] Baseline BERT..."
python scripts/evaluate_baseline.py --model bert
echo "✓ Baseline BERT complete"
echo ""

echo "[2/2] Baseline RoBERTa..."
python scripts/evaluate_baseline.py --model roberta
echo "✓ Baseline RoBERTa complete"
echo ""

# ============================================================================
# STEP 3: PRETRAINED EVALUATIONS
# ============================================================================

echo "STEP 3: Running pretrained evaluations..."
echo "  Time: ~5-10 min total"
echo ""

echo "[1/2] Pretrained BERT..."
python scripts/evaluate_pretrained.py --model bert
echo "✓ Pretrained BERT complete"
echo ""

echo "[2/2] Pretrained RoBERTa..."
python scripts/evaluate_pretrained.py --model roberta
echo "✓ Pretrained RoBERTa complete"
echo ""

# ============================================================================
# STEP 4: FINE-TUNED EVALUATIONS
# ============================================================================

echo "STEP 4: Running fine-tuned evaluations..."
echo "  Time: ~5-10 min total"
echo ""

echo "[1/2] Fine-tuned BERT..."
python scripts/evaluate.py --model bert
echo "✓ Fine-tuned BERT complete"
echo ""

echo "[2/2] Fine-tuned RoBERTa..."
python scripts/evaluate.py --model roberta
echo "✓ Fine-tuned RoBERTa complete"
echo ""

# ============================================================================
# STEP 5: LLAMA ZERO-SHOT
# ============================================================================

echo "STEP 5: Running LLaMA zero-shot evaluations..."
echo "  V100: ~5-8 hours | A100: ~3.5-5 hours"
echo ""

# Ensure few-shot is disabled
python -c "
with open('scripts/config.py', 'r') as f:
    content = f.read()
content = content.replace(\"'enabled': True\", \"'enabled': False\")
with open('scripts/config.py', 'w') as f:
    f.write(content)
print('✓ Few-shot disabled')
"

echo "[1/2] LLaMA 7B zero-shot..."
echo "  V100: ~2-3 hours | A100: ~1.5-2 hours"
python scripts/evaluate.py --model llama_7b
echo "✓ LLaMA 7B zero-shot complete"
echo ""

echo "[2/2] LLaMA 13B zero-shot..."
echo "  V100: ~3-5 hours | A100: ~2-3 hours"
python scripts/evaluate.py --model llama_13b
echo "✓ LLaMA 13B zero-shot complete"
echo ""

# ============================================================================
# STEP 6: LLAMA FEW-SHOT
# ============================================================================

echo "STEP 6: Running LLaMA few-shot evaluations..."
echo "  V100: ~8-11 hours | A100: ~6.5-8 hours"
echo ""

# Enable few-shot
python -c "
with open('scripts/config.py', 'r') as f:
    content = f.read()
content = content.replace(\"'enabled': False\", \"'enabled': True\")
with open('scripts/config.py', 'w') as f:
    f.write(content)
print('✓ Few-shot enabled (5-shot)')
"

echo "[1/2] LLaMA 7B 5-shot..."
echo "  V100: ~3-4 hours | A100: ~2.5-3 hours"
python scripts/evaluate.py --model llama_7b
echo "✓ LLaMA 7B 5-shot complete"
echo ""

echo "[2/2] LLaMA 13B 5-shot..."
echo "  V100: ~5-7 hours | A100: ~4-5 hours"
python scripts/evaluate.py --model llama_13b
echo "✓ LLaMA 13B 5-shot complete"
echo ""

# ============================================================================
# COMPLETE
# ============================================================================

echo "========================================================================"
echo "ALL TRAINING AND EVALUATIONS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in results/ directory:"
echo "  - baseline_bert_results.csv & baseline_roberta_results.csv"
echo "  - bert_pretrained_results.csv & roberta_pretrained_results.csv"
echo "  - bert_results.csv & roberta_results.csv (fine-tuned)"
echo "  - llama_7b_results.csv & llama_7b_5shot_results.csv"
echo "  - llama_13b_results.csv & llama_13b_5shot_results.csv"
echo "  - All corresponding summary.csv files"
echo ""
echo "========================================================================"