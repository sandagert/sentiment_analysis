# ============================================================================
# Evaluate models with prompt variations
# Works for BERT, RoBERTa, LLaMA 7B, and LLaMA 13B (zero-shot and few-shot)
# ============================================================================

import argparse
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from datasets import load_from_disk, load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config import (PROMPTS, BERT_CONFIG, ROBERTA_CONFIG,
                    LLAMA_7B_CONFIG, LLAMA_13B_CONFIG,
                    EVAL_CONFIG, TEST_MODE, TEST_CONFIG, FEW_SHOT_CONFIG)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Evaluate sentiment analysis models')
parser.add_argument('--model', type=str, required=True,
                    choices=['bert', 'roberta', 'llama_7b', 'llama_13b'],
                    help='Model to evaluate: bert, roberta, llama_7b, or llama_13b')
args = parser.parse_args()

print("=" * 80)
print(f"EVALUATION: {args.model.upper()}")
if TEST_MODE:
    print("⚠️  RUNNING IN TEST MODE - Using subset of data")
if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    print(f"🎯 FEW-SHOT MODE: Using {FEW_SHOT_CONFIG['n_shots']} examples")
print("=" * 80)

# ============================================================================
# LOAD TEST DATA
# ============================================================================

print("\n1. Loading test data...")

# Load appropriate tokenized dataset
if args.model == 'bert':
    tokenized_datasets = load_from_disk(BERT_CONFIG["tokenized_data_path"])
    model_dir = BERT_CONFIG["final_model_dir"]
elif args.model == 'roberta':
    tokenized_datasets = load_from_disk(ROBERTA_CONFIG["tokenized_data_path"])
    model_dir = ROBERTA_CONFIG["final_model_dir"]
elif args.model == 'llama_7b':
    tokenized_datasets = load_from_disk(BERT_CONFIG["tokenized_data_path"])
    model_dir = LLAMA_7B_CONFIG["final_model_dir"]
else:  # llama_13b
    tokenized_datasets = load_from_disk(BERT_CONFIG["tokenized_data_path"])
    model_dir = LLAMA_13B_CONFIG["final_model_dir"]

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
# LOAD TRAINING DATA (for few-shot learning)
# ============================================================================

few_shot_examples = None

if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    print("\n2a. Loading training data for few-shot learning...")

    # Load full IMDB dataset
    imdb_dataset = load_dataset("imdb")
    train_data = imdb_dataset['train']

    # Select few-shot examples
    n_shots = FEW_SHOT_CONFIG['n_shots']

    # Set random seed for reproducibility
    random.seed(42)

    print(f"   Selecting {n_shots} balanced examples...")

    # Separate by class
    positive_samples = [
        (item['text'], 1)
        for item in train_data
        if item['label'] == 1
    ]
    negative_samples = [
        (item['text'], 0)
        for item in train_data
        if item['label'] == 0
    ]

    # Sample balanced examples
    n_positive = n_shots // 2
    n_negative = n_shots - n_positive

    selected_positive = random.sample(positive_samples, n_positive)
    selected_negative = random.sample(negative_samples, n_negative)

    few_shot_examples = selected_positive + selected_negative
    random.shuffle(few_shot_examples)

    print(f"   ✓ Selected {len(few_shot_examples)} examples:")
    for i, (text, label) in enumerate(few_shot_examples, 1):
        label_text = "positive" if label == 1 else "negative"
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"      {i}. [{label_text}] {preview}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\n{'2b' if few_shot_examples else '2'}. Loading {args.model.upper()} model...")

if args.model in ['bert', 'roberta']:
    classifier = pipeline(
        "sentiment-analysis",
        model=model_dir,
        device=0 if torch.cuda.is_available() else -1
    )
    print(f"   ✓ Loaded from {model_dir}")
elif args.model in ['llama_7b', 'llama_13b']:
    # Load tokenizer from HuggingFace, model from local
    model_name = LLAMA_7B_CONFIG["model_name"] if args.model == 'llama_7b' else LLAMA_13B_CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"   ✓ Loaded from {model_dir}")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_encoder(classifier, prompt, reviews):
    """Evaluate BERT/RoBERTa with a specific prompt"""
    predictions = []
    confidences = []

    for idx, review in enumerate(reviews):
        if idx % 50 == 0 and idx > 0:
            print(f"{idx}/{len(reviews)}", end="...", flush=True)

        prompted_text = prompt + review
        pred = classifier(prompted_text, truncation=True, max_length=512)[0]
        pred_label = 1 if pred['label'] == 'LABEL_1' else 0
        predictions.append(pred_label)
        confidences.append(pred['score'])

    return predictions, confidences


def _build_prompt(tokenizer, base_prompt: str, review: str) -> str:
    """Build proper LLaMA-2-chat prompt (zero-shot)"""
    if "{review}" in base_prompt:
        user_text = base_prompt.format(review=review)
    else:
        user_text = base_prompt + "\n\nReview:\n" + review + "\n\nAnswer with only 'positive' or 'negative'."

    system_text = "You are a sentiment classifier. Given a movie review, decide if it's positive or negative."

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = f"[INST] <<SYS>> {system_text} <</SYS>> {user_text} [/INST]"

    return prompt_text


def _build_prompt_with_examples(tokenizer, base_prompt: str, examples: list, review: str) -> str:
    """Build LLaMA-2-chat prompt with few-shot examples"""
    system_text = (
        "You are a sentiment classifier. "
        "Learn from the examples below, then classify the final review."
    )

    # Build examples section
    examples_text = ""
    if examples:
        examples_text = "Here are some examples:\n\n"
        for i, (review_text, label) in enumerate(examples, 1):
            # Truncate long examples to save tokens
            review_short = review_text[:150] + "..." if len(review_text) > 150 else review_text
            label_text = "positive" if label == 1 else "negative"
            examples_text += f"Example {i}:\n"
            examples_text += f"Review: {review_short}\n"
            examples_text += f"Label: {label_text}\n\n"

    # Add the test review
    user_text = (
        f"{examples_text}"
        f"Now classify this review:\n"
        f"Review: {review}\n\n"
        f"Answer with only 'positive' or 'negative'."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text} [/INST]"

    return prompt_text


def _score_labels(model, tokenizer, prompt_text: str, labels: list) -> list:
    """Compute log-probabilities for each label as prompt continuation"""
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=2000  # Increased for few-shot
    )["input_ids"]

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    logprobs = []

    for label in labels:
        label_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + label_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        prompt_len = len(prompt_ids)
        lp = 0.0
        for j, token_id in enumerate(label_ids):
            pos = prompt_len - 1 + j
            token_logits = logits[0, pos]
            token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            lp += float(token_log_probs[token_id].item())

        logprobs.append(lp)

    return logprobs


def evaluate_llama(model, tokenizer, prompt, reviews, few_shot_examples=None):
    """
    Zero-shot or few-shot classification via label scoring (NO GENERATION)

    Args:
        model: LLaMA model
        tokenizer: LLaMA tokenizer
        prompt: Base prompt template
        reviews: List of reviews to classify
        few_shot_examples: Optional list of (review, label) tuples for few-shot
    """
    model.eval()
    labels = ["negative", "positive"]
    predictions = []
    confidences = []
    log_prob_diffs = []
    start_time = time.time()

    for idx, review in enumerate(reviews):
        if idx % 10 == 0 and idx > 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (len(reviews) - idx)
            print(f"   Progress: {idx}/{len(reviews)} | ETA: {remaining / 60:.1f}m")

        # Build prompt (with or without examples)
        if few_shot_examples:
            prompt_text = _build_prompt_with_examples(tokenizer, prompt, few_shot_examples, review)
        else:
            prompt_text = _build_prompt(tokenizer, prompt, review)

        # Score labels (same for both zero-shot and few-shot)
        logprobs = _score_labels(model, tokenizer, prompt_text, labels)

        logprob_tensor = torch.tensor(logprobs)
        probs = torch.softmax(logprob_tensor, dim=-1).tolist()

        pred_idx = int(torch.argmax(logprob_tensor).item())
        confidence = float(probs[pred_idx])

        # Log-probability difference (decision margin)
        log_prob_diff = abs(logprobs[1] - logprobs[0])

        predictions.append(pred_idx)
        confidences.append(confidence)
        log_prob_diffs.append(log_prob_diff)

    elapsed = time.time() - start_time
    print(f"   ✓ Completed in {elapsed / 60:.1f} minutes")
    return predictions, confidences, log_prob_diffs


# ============================================================================
# RUN EVALUATION
# ============================================================================

print("\n3. Testing prompt variations...")
print("=" * 80)

all_results = []
all_predictions = []

prompt_counter = 0
total_prompts = len(PROMPTS) * 3

for category, prompts in PROMPTS.items():
    print(f"\n{category.upper()} PROMPTS:")
    print("-" * 40)

    for i, prompt in enumerate(prompts):
        prompt_counter += 1
        print(f"\n[{prompt_counter}/{total_prompts}] Variation {i + 1}/3")
        print(f"Prompt: '{prompt[:60]}...'")

        prompt_start = time.time()

        if args.model in ['bert', 'roberta']:
            predictions, confidences = evaluate_encoder(classifier, prompt, test_reviews)
            log_prob_diffs = [None] * len(predictions)  # Placeholder for non-LLaMA
        elif args.model in ['llama_7b', 'llama_13b']:
            # Pass few_shot_examples to evaluate_llama
            predictions, confidences, log_prob_diffs = evaluate_llama(
                model, tokenizer, prompt, test_reviews, few_shot_examples
            )

        all_predictions.extend(predictions)

        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        # Calculate log-prob metrics only for LLaMA
        if log_prob_diffs[0] is not None:
            avg_log_prob_diff = np.mean(log_prob_diffs)
            std_log_prob_diff = np.std(log_prob_diffs)
        else:
            avg_log_prob_diff = None
            std_log_prob_diff = None

        all_results.append({
            'model': args.model,
            'category': category,
            'variation': i + 1,
            'accuracy': acc,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'avg_log_prob_diff': avg_log_prob_diff,
            'std_log_prob_diff': std_log_prob_diff,
        })

        print(f"✓ Acc={acc:.4f}, F1={f1:.4f}, Conf={avg_confidence:.4f}±{std_confidence:.4f}")
        if avg_log_prob_diff is not None:
            print(f"  Log-prob margin: {avg_log_prob_diff:.4f}±{std_log_prob_diff:.4f}")
        print(f"⏱  Time: {(time.time() - prompt_start) / 60:.1f} minutes")

# ============================================================================
# OVERALL CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("4. Generating overall confusion matrix...")

true_labels_repeated = true_labels * 9
cm = confusion_matrix(true_labels_repeated, all_predictions)
tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Confusion Matrix (all prompts combined):")
print(f"                   Predicted")
print(f"                  Negative  Positive")
print(f"  Actual Negative   {tn:5d}     {fp:5d}")
print(f"         Positive   {fn:5d}     {tp:5d}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n5. Saving results...")

os.makedirs(EVAL_CONFIG['results_dir'], exist_ok=True)
results_df = pd.DataFrame(all_results)

# Save detailed results
if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    output_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_{FEW_SHOT_CONFIG['n_shots']}shot_results.csv"
else:
    output_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_results.csv"

results_df.to_csv(output_file, index=False)
print(f"   ✓ Saved detailed results to {output_file}")

# ============================================================================
# CREATE AND SAVE SUMMARY STATISTICS
# ============================================================================

summary_data = {
    'model': [args.model],
    'n_shots': [
        FEW_SHOT_CONFIG['n_shots'] if (args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']) else 0],
    'n_prompts': [len(results_df)],
    'n_samples': [test_size],
    'mean_accuracy': [results_df['accuracy'].mean()],
    'std_accuracy': [results_df['accuracy'].std()],
    'min_accuracy': [results_df['accuracy'].min()],
    'max_accuracy': [results_df['accuracy'].max()],
    'range_accuracy': [results_df['accuracy'].max() - results_df['accuracy'].min()],
    'mean_f1_score': [results_df['f1_score'].mean()],
    'std_f1_score': [results_df['f1_score'].std()],
    'mean_confidence': [results_df['avg_confidence'].mean()],
    'std_confidence': [results_df['avg_confidence'].std()],
}

# Add LLaMA-specific metrics if available
if results_df['avg_log_prob_diff'].notna().any():
    summary_data['mean_log_prob_diff'] = [results_df['avg_log_prob_diff'].mean()]
    summary_data['std_log_prob_diff'] = [results_df['avg_log_prob_diff'].std()]
else:
    summary_data['mean_log_prob_diff'] = [None]
    summary_data['std_log_prob_diff'] = [None]

# Add category-specific accuracy
for category in ['neutral', 'descriptive', 'metaphorical']:
    cat_data = results_df[results_df['category'] == category]
    if len(cat_data) > 0:
        summary_data[f'{category}_accuracy'] = [cat_data['accuracy'].mean()]
    else:
        summary_data[f'{category}_accuracy'] = [None]

summary_df = pd.DataFrame(summary_data)

# Save summary
if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    summary_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_{FEW_SHOT_CONFIG['n_shots']}shot_summary.csv"
else:
    summary_file = f"{EVAL_CONFIG['results_dir']}/{args.model}_summary.csv"

summary_df.to_csv(summary_file, index=False)
print(f"   ✓ Saved summary statistics to {summary_file}")

# ============================================================================
# SAVE CONFUSION MATRIX
# ============================================================================

os.makedirs(f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.ylabel('Actual Sentiment', fontsize=12)
plt.xlabel('Predicted Sentiment', fontsize=12)

# Update title for few-shot
if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    title = f'{args.model.upper()} ({FEW_SHOT_CONFIG["n_shots"]}-shot) - Confusion Matrix\n(All Prompt Variations)'
else:
    title = f'{args.model.upper()} - Overall Confusion Matrix\n(All Prompt Variations Combined)'

plt.title(title, fontsize=14)
plt.tight_layout()

if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    cm_file = f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}/{args.model}_{FEW_SHOT_CONFIG['n_shots']}shot_confusion_matrix.png"
else:
    cm_file = f"{EVAL_CONFIG['results_dir'].replace('results', 'figures')}/{args.model}_confusion_matrix.png"

plt.savefig(cm_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved confusion matrix to {cm_file}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
if args.model in ['llama_7b', 'llama_13b'] and FEW_SHOT_CONFIG['enabled']:
    print(f"{args.model.upper()} {FEW_SHOT_CONFIG['n_shots']}-SHOT EVALUATION COMPLETE")
else:
    print(f"{args.model.upper()} EVALUATION COMPLETE")

if TEST_MODE:
    print("⚠️  This was a TEST run. Set TEST_MODE=False in config.py for full evaluation.")
print("=" * 80)

print(f"\nOverall Performance:")
print(f"  Mean Accuracy: {results_df['accuracy'].mean():.4f} (±{results_df['accuracy'].std():.4f})")
print(f"  Mean F1 Score: {results_df['f1_score'].mean():.4f} (±{results_df['f1_score'].std():.4f})")
print(f"  Accuracy Range: {results_df['accuracy'].min():.4f} to {results_df['accuracy'].max():.4f}")

print(f"\nBy Category:")
for category in ['neutral', 'descriptive', 'metaphorical']:
    cat_data = results_df[results_df['category'] == category]
    if len(cat_data) > 0:
        print(f"  {category.capitalize()}: Acc={cat_data['accuracy'].mean():.4f}")

print("=" * 80)