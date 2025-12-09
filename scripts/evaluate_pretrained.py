# ============================================================================
# Evaluate PRE-TRAINED (zero-shot) models with prompt variations
# Tests BERT/RoBERTa straight from Hugging Face without fine-tuning
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
from config import PROMPTS, BERT_CONFIG, ROBERTA_CONFIG, EVAL_CONFIG, TEST_MODE, TEST_CONFIG

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Evaluate pre-trained models (zero-shot)')
parser.add_argument('--model', type=str, required=True, choices=['bert', 'roberta'],
                    help='Model to evaluate: bert or roberta')
args = parser.parse_args()

print("=" * 80)
print(f"ZERO-SHOT EVALUATION: {args.model.upper()}")
print("Testing PRE-TRAINED model (no fine-tuning)")
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
    model_name = BERT_CONFIG["model_name"]  # bert-base-uncased
elif args.model == 'roberta':
    tokenized_datasets = load_from_disk(ROBERTA_CONFIG["tokenized_data_path"])
    model_name = ROBERTA_CONFIG["model_name"]  # roberta-base

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
# LOAD PRE-TRAINED MODEL
# ============================================================================

print(f"\n2. Loading PRE-TRAINED {args.model.upper()} from Hugging Face...")
print(f"   Model: {model_name}")
print("   ⚠️  This is the original pre-trained model (NOT fine-tuned on IMDB)")

# Load pre-trained model directly from Hugging Face
classifier = pipeline(
    "sentiment-analysis",
    model=model_name,
    device=0 if torch.cuda.is_available() else -1
)

print(f"   ✓ Loaded {model_name}")


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_pretrained(classifier, prompt, reviews):
    """Evaluate pre-trained model with a specific prompt"""
    predictions = []
    confidences = []

    for review in reviews:
        prompted_text = prompt + review
        pred = classifier(prompted_text, truncation=True, max_length=512)[0]

        pred_label = 1 if pred['label'] == 'POSITIVE' or pred['label'] == 'LABEL_1' else 0
        predictions.append(pred_label)
        confidences.append(pred['score'])

    return predictions, confidences


# ============================================================================
# RUN EVALUATION
# ============================================================================

print("\n3. Testing prompt variations (zero-shot)...")
print("=" * 80)

all_results = []
all_predictions = []  # Store all predictions for overall confusion matrix

for category, prompts in PROMPTS.items():
    print(f"\n{category.upper()} PROMPTS:")
    print("-" * 40)

    for i, prompt in enumerate(prompts):
        print(f"  Variation {i + 1}/3...", end=" ")

        # Evaluate with prompt
        predictions, confidences = evaluate_pretrained(classifier, prompt, test_reviews)

        # Store for overall confusion matrix
        all_predictions.extend(predictions)

        # Calculate metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        # Store results
        all_results.append({
            'model': f'{args.model}_pretrained',
            'category': category,
            'variation': i + 1,
            'accuracy': acc,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence
        })

        print(f"Acc={acc:.4f}, F1={f1:.4f}, Conf={avg_confidence:.4f}±{std_confidence:.4f}")

# ============================================================================
# OVERALL CONFUSION MATRIX
# ============================================================================

print("\n4. Generating overall confusion matrix...")

# Calculate overall confusion matrix (all prompts combined)
true_labels_repeated = true_labels * 9

cm = confusion_matrix(true_labels_repeated, all_predictions)
tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Confusion Matrix (all prompts combined):")
print(f"                   Predicted")
print(f"                  Negative  Positive")
print(f"  Actual Negative   {tn:5d}     {fp:5d}")
print(f"         Positive   {fn:5d}     {tp:5d}")

print(f"\nBreakdown:")
print(f"  True Negatives:  {tn:5d} ({tn / len(true_labels_repeated) * 100:.1f}%)")
print(f"  True Positives:  {tp:5d} ({tp / len(true_labels_repeated) * 100:.1f}%)")
print(f"  False Negatives: {fn:5d} ({fn / len(true_labels_repeated) * 100:.1f}%)")
print(f"  False Positives: {fp:5d} ({fp / len(true_labels_repeated) * 100:.1f}%)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n5. Saving results...")

# Save metrics CSV
results_df = pd.DataFrame(all_results)
output_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_pretrained_results.csv"
results_df.to_csv(output_file, index=False)

print(f"   ✓ Saved metrics to {output_file}")

# Save confusion matrix visualization
os.makedirs(f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.ylabel('Actual Sentiment', fontsize=12)
plt.xlabel('Predicted Sentiment', fontsize=12)
plt.title(f'{args.model.upper()} PRE-TRAINED (Zero-Shot)\nConfusion Matrix - All Prompts', fontsize=14)
plt.tight_layout()

cm_file = f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}/{args.model}_pretrained_confusion_matrix.png"
plt.savefig(cm_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved confusion matrix to {cm_file}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(f"{args.model.upper()} PRE-TRAINED (ZERO-SHOT) EVALUATION COMPLETE")
if TEST_MODE:
    print("⚠️  This was a TEST run. Set TEST_MODE=False in config.py for full evaluation.")
print("=" * 80)

print(f"\nOverall Performance (zero-shot):")
print(f"  Mean Accuracy: {results_df['accuracy'].mean():.4f} (±{results_df['accuracy'].std():.4f})")
print(f"  Mean F1 Score: {results_df['f1_score'].mean():.4f} (±{results_df['f1_score'].std():.4f})")

print(f"\nBy Category:")
for category in ['neutral', 'descriptive', 'metaphorical']:
    cat_data = results_df[results_df['category'] == category]
    print(f"  {category.capitalize()}: Acc={cat_data['accuracy'].mean():.4f}")

print("\n" + "=" * 80)
print("Compare these zero-shot results to fine-tuned results to see impact of training.")
print("=" * 80)