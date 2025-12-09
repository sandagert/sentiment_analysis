# ============================================================================
# Setup LLaMA-2-13B-chat model
# Downloads and saves the model to models/llama_13b/final
# ============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from config import LLAMA_13B_CONFIG

print("=" * 80)
print("SETTING UP LLAMA-2-13B-CHAT")
print("=" * 80)

model_name = LLAMA_13B_CONFIG["model_name"]
output_dir = LLAMA_13B_CONFIG["final_model_dir"]

print(f"\nModel: {model_name}")
print(f"Output directory: {output_dir}")
print("\n⚠️  NOTE: This is a larger model (~26GB)")
print("   Download may take 15-30 minutes depending on connection")

# Create output directory
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Created directory: {output_dir}")

# ============================================================================
# Download Tokenizer
# ============================================================================

print("\n1. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
print("   ✓ Tokenizer saved")

# ============================================================================
# Download Model
# ============================================================================

print("\n2. Downloading model (this will take longer than 7B)...")
print("   Model size: ~26GB")
print("   This may take 15-30 minutes...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"  # For models larger than RAM
)

print("   ✓ Model downloaded")

# ============================================================================
# Save Model
# ============================================================================

print("\n3. Saving model...")
model.save_pretrained(output_dir)
print("   ✓ Model saved")

# ============================================================================
# Verification
# ============================================================================

print("\n4. Verifying installation...")
saved_files = os.listdir(output_dir)
print(f"   Saved {len(saved_files)} files")

essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
missing = [f for f in essential_files if f not in saved_files]

if missing:
    print(f"   ⚠️  Warning: Missing files: {missing}")
else:
    print("   ✓ All essential files present")

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("LLAMA-2-13B SETUP COMPLETE")
print("=" * 80)
print(f"\nModel saved to: {output_dir}")
print("\nYou can now run evaluation with:")
print("  python scrips/evaluate.py --model llama_13b")
print("\n⚠️  Memory Requirements:")
print("   - 7B model: ~14GB")
print("   - 13B model: ~26GB")
print("   - Recommended: Cloud instance with 32GB+ RAM")
print("=" * 80)