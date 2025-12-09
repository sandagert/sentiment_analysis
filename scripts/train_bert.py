# ============================================================================
# Train BERT on IMDB sentiment analysis
# ============================================================================

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from config import BERT_CONFIG, TEST_MODE, TEST_CONFIG

print("="*80)
print("TRAINING: BERT")
if TEST_MODE:
    print("⚠️  RUNNING IN TEST MODE - Using subset of data")
print("="*80)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================

print("\n1. Loading preprocessed data...")

# Load tokenized dataset from disk (fast - no re-tokenizing)
tokenized_datasets = load_from_disk(BERT_CONFIG["tokenized_data_path"])

# Apply test mode subset if enabled
if TEST_MODE:
    print(f"   ⚠️  Test mode: Using {TEST_CONFIG['train_samples']} train, {TEST_CONFIG['test_samples']} test samples")
    tokenized_datasets['train'] = tokenized_datasets['train'].select(range(TEST_CONFIG['train_samples']))
    tokenized_datasets['test'] = tokenized_datasets['test'].select(range(TEST_CONFIG['test_samples']))

print(f"   ✓ Train set: {len(tokenized_datasets['train'])} reviews")
print(f"   ✓ Test set: {len(tokenized_datasets['test'])} reviews")

# ============================================================================
# STEP 2: LOAD MODEL
# ============================================================================

print("\n2. Loading BERT model...")

# Load pre-trained BERT with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    BERT_CONFIG["model_name"],
    num_labels=2  # Binary classification
)

print(f"   ✓ Loaded {BERT_CONFIG['model_name']}")

# ============================================================================
# STEP 3: CONFIGURE TRAINING
# ============================================================================

print("\n3. Configuring training...")

# Override epochs in test mode
num_epochs = TEST_CONFIG["num_epochs"] if TEST_MODE else BERT_CONFIG["num_train_epochs"]

training_args = TrainingArguments(
    output_dir=BERT_CONFIG["output_dir"],
    num_train_epochs=num_epochs,
    per_device_train_batch_size=BERT_CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=BERT_CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=BERT_CONFIG["gradient_accumulation_steps"],
    learning_rate=BERT_CONFIG["learning_rate"],
    warmup_steps=BERT_CONFIG["warmup_steps"],
    weight_decay=BERT_CONFIG["weight_decay"],
    logging_dir='../logs/bert',
    logging_steps=10 if TEST_MODE else 100,  # Log more frequently in test mode
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

print(f"   ✓ Training for {num_epochs} epoch(s)")

# ============================================================================
# STEP 4: INITIALIZE TRAINER
# ============================================================================

print("\n4. Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("   ✓ Trainer ready")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
if TEST_MODE:
    print("Starting TEST training (~5 minutes)...")
else:
    print("Starting FULL training (~1-2 hours)...")
print("="*80 + "\n")

# Train the model
trainer.train()

print("\n" + "="*80)
print("Training complete!")
print("="*80)

# ============================================================================
# STEP 6: SAVE FINAL MODEL
# ============================================================================

print("\n5. Saving final model...")

# Save model and tokenizer together
trainer.save_model(BERT_CONFIG["final_model_dir"])

# Also save tokenizer in same location
tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
tokenizer.save_pretrained(BERT_CONFIG["final_model_dir"])

print(f"   ✓ Model saved to {BERT_CONFIG['final_model_dir']}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*80)
print("BERT TRAINING COMPLETE")
if TEST_MODE:
    print("⚠️  This was a TEST run. Set TEST_MODE=False in config.py for full training.")
print("="*80)
print(f"\nFinal model saved to: {BERT_CONFIG['final_model_dir']}")
print("You can now run: python evaluate.py --model bert")
print("="*80)