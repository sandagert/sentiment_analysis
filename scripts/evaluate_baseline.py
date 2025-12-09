# ============================================================================
# Baseline evaluation - Test ENCODER models WITHOUT prompts
# Purpose: Measure natural performance before prompt engineering
# Note: Only works for BERT/RoBERTa (encoder models with classification heads)
#       LLaMA (generative model) cannot be evaluated without instructions
# ============================================================================

import argparse
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from baseline_config import BERT_CONFIG, ROBERTA_CONFIG, EVAL_CONFIG, TEST_MODE, TEST_CONFIG

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Baseline evaluation without prompts (encoder models only)')
parser.add_argument('--model', type=str, required=True, choices=['bert', 'roberta'],
                    help='Model to evaluate: bert or roberta (LLaMA not supported for baseline)')
args = parser.parse_args()

print("=" * 80)
print(f"BASELINE EVALUATION: {args.model.upper()}")
print("Testing WITHOUT prompts (raw reviews only)")
if TEST_MODE:
    print("⚠️  RUNNING IN TEST MODE - Using subset of data")
print("=" * 80)

# ============================================================================
# LOAD TEST DATA
# ============================================================================

print("\n1. Loading test data...")

# Load appropriate tokenized dataset
if args.model == 'bert':
    tokenized_datasets = load_from_disk(BERT_CONFIG["tokenized_data_path"])
    model_dir = BERT_CONFIG["model_dir"]
elif args.model == 'roberta':
    tokenized_datasets = load_from_disk(ROBERTA_CONFIG["tokenized_data_path"])
    model_dir = ROBERTA_CONFIG["model_dir"]

# Get test data
test_df = tokenized_datasets["test"].to_pandas()

# Use test mode subset or configured test size
if TEST_MODE:
    test_size = TEST_CONFIG["eval_samples"]
    print(f"   ⚠️  Test mode: Evaluating on {test_size} samples")
else:
    test_size = EVAL_CONFIG["test_size"] if EVAL_CONFIG["test_size"] else len(test_df)

test_reviews = test_df['review'][:test_size].tolist()
true_labels = test_df['labels'][:test_size].tolist()

print(f"   ✓ Evaluating on {test_size} reviews")

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\n2. Loading {args.model.upper()} model...")

# Encoder models - use pipeline
classifier = pipeline(
    "sentiment-analysis",
    model=model_dir,
    device=0 if torch.cuda.is_available() else -1
)
print(f"   ✓ Loaded from {model_dir}")


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_encoder_baseline(classifier, reviews):
    """Evaluate BERT/RoBERTa WITHOUT prompts - just raw reviews"""
    predictions = []
    confidences = []

    print("   Evaluating...", end=" ")

    for i, review in enumerate(reviews):
        # NO PROMPT - just feed raw review
        pred = classifier(review, truncation=True, max_length=512)[0]

        pred_label = 1 if pred['label'] == 'LABEL_1' else 0
        predictions.append(pred_label)
        confidences.append(pred['score'])

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(reviews)}...", end=" ")

    print("Done!")
    return predictions, confidences


# ============================================================================
# RUN BASELINE EVALUATION
# ============================================================================

print("\n3. Running baseline evaluation (no prompts)...")
print("=" * 80)

predictions, confidences = evaluate_encoder_baseline(classifier, test_reviews)

# Calculate metrics
acc = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
avg_confidence = np.mean(confidences)
std_confidence = np.std(confidences)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 80)
print(f"BASELINE RESULTS: {args.model.upper()}")
print("=" * 80)
print(f"\nPerformance WITHOUT prompts:")
print(f"  Accuracy:         {acc:.4f} ({acc * 100:.2f}%)")
print(f"  F1 Score:         {f1:.4f}")
print(f"  Avg Confidence:   {avg_confidence:.4f} (±{std_confidence:.4f})")

print(f"\nConfusion Matrix:")
print(f"                   Predicted")
print(f"                  Negative  Positive")
print(f"  Actual Negative   {tn:5d}     {fp:5d}")
print(f"         Positive   {fn:5d}     {tp:5d}")

print(f"\nBreakdown:")
print(f"  True Negatives:  {tn:5d} ({tn / len(true_labels) * 100:.1f}%) - Correctly identified negative")
print(f"  True Positives:  {tp:5d} ({tp / len(true_labels) * 100:.1f}%) - Correctly identified positive")
print(f"  False Negatives: {fn:5d} ({fn / len(true_labels) * 100:.1f}%) - Missed positive (said negative)")
print(f"  False Positives: {fp:5d} ({fp / len(true_labels) * 100:.1f}%) - Missed negative (said positive)")
print("=" * 80)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n4. Saving results...")

# Create results directory if it doesn't exist
os.makedirs(EVAL_CONFIG['results_dir'], exist_ok=True)

# Save metrics CSV
results = {
    'model': args.model,
    'prompt_type': 'baseline',
    'accuracy': acc,
    'f1_score': f1,
    'avg_confidence': avg_confidence,
    'std_confidence': std_confidence,
    'test_samples': len(test_reviews),
    'true_negatives': int(tn),
    'true_positives': int(tp),
    'false_negatives': int(fn),
    'false_positives': int(fp),
}

results_df = pd.DataFrame([results])
output_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_baseline.csv"
results_df.to_csv(output_file, index=False)

print(f"   ✓ Saved metrics to {output_file}")

# Save confusion matrix visualization
os.makedirs(f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.ylabel('Actual Sentiment', fontsize=12)
plt.xlabel('Predicted Sentiment', fontsize=12)
plt.title(f'{args.model.upper()} Baseline - Confusion Matrix\n(No Prompts)', fontsize=14)
plt.tight_layout()

cm_file = f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}/{args.model}_baseline_confusion_matrix.png"
plt.savefig(cm_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved confusion matrix to {cm_file}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(f"{args.model.upper()} BASELINE EVALUATION COMPLETE")
if TEST_MODE:
    print("⚠️  This was a TEST run. Set TEST_MODE=False in baseline_config.py for full evaluation.")
print("=" * 80)
print("\nNote: LLaMA baseline not available (generative models require instructions)")
print("This baseline can be compared against prompted results to measure prompt sensitivity.")
print("=" * 80)