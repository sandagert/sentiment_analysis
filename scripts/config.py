# ============================================================================
# Configuration file for sentiment analysis experiments
# ============================================================================

# ============================================================================
# TEST MODE TOGGLE
# ============================================================================

# Set to True for quick testing (5-10 min), False for full training (hours)
TEST_MODE = False

TEST_CONFIG = {
    "train_samples": 100,      # Use 100 training samples in test mode
    "test_samples": 20,        # Use 20 test samples in test mode
    "eval_samples": 50,        # Evaluate on 50 samples in test mode
    "num_epochs": 1,           # Just 1 epoch in test mode
}

# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

PROMPTS = {
    "neutral": [
        "Classify the sentiment of the following movie review as positive or negative. Review: ",
        "Determine whether the sentiment in this review is positive or negative. Review: ",
        "Label this review as either positive or negative. Review: "
    ],
    "descriptive": [
    "You are a sentiment classifier. Task: Classify as Positive or Negative. Positive = reviewer liked the movie; Negative = did not. Think step by step: (1) Identify emotional language, (2) Assess overall tone, (3) Determine satisfaction. Output format: [Positive/Negative]. Review: ",
    "Classify this movie review's sentiment. Output only: Positive or Negative. Example: 'Great film, loved it!' = Positive. 'Terrible waste of time' = Negative. Review: ",
    "Role: Professional sentiment analyst. Task: Classify as Positive or Negative. Reason through: (1) What is the overall assessment? (2) What emotional indicators exist? (3) Would they recommend it? Format: [Positive/Negative]. Review: "
    ],
    "metaphorical": [
        "Is this review a ray of sunshine or a stormy cloud? Choose: positive or negative. Review: ",
        "Did the writer cheer or boo the movie? Classify as positive or negative. Review: ",
        "Is this review a love letter to cinema or a breakup text? Decide the sentiment. Review: "
    ]
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

BERT_CONFIG = {
    "model_name": "bert-base-uncased",
    "tokenized_data_path": "data/processed/bert_tokenized",
    "output_dir": "models/bert/checkpoints",
    "final_model_dir": "models/bert/final",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,      # ← Was 4, now 32 (8x faster)
    "per_device_eval_batch_size": 64,       # ← Was 16, now 64 (4x faster)
    "gradient_accumulation_steps": 1,       # ← Was 4, now 1 (no need with larger batches)
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
}

ROBERTA_CONFIG = {
    "model_name": "roberta-base",
    "tokenized_data_path": "data/processed/roberta_tokenized",
    "output_dir": "models/roberta/checkpoints",
    "final_model_dir": "models/roberta/final",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,      # ← Was 4, now 32
    "per_device_eval_batch_size": 64,       # ← Was 16, now 64
    "gradient_accumulation_steps": 1,       # ← Was 4, now 1
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
}

# LLaMA 7B Configuration
LLAMA_7B_CONFIG = {
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "final_model_dir": "models/llama_7b/final",
    "max_new_tokens": 10,
    "temperature": 0.1,
}

# LLaMA 13B Configuration
LLAMA_13B_CONFIG = {
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "final_model_dir": "models/llama_13b/final",
    "max_new_tokens": 10,
    "temperature": 0.1,
}

# For backward compatibility (keep this if you have old scripts)
LLAMA_CONFIG = LLAMA_7B_CONFIG

# Few-Shot Learning Configuration
FEW_SHOT_CONFIG = {
    'enabled': True,  # Set to True to use few-shot
    'n_shots': 5,      # Number of examples
    'strategy': 'balanced'
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVAL_CONFIG = {
    "test_size": None,  # Number of samples to evaluate (set to None for full test set)
    "results_dir": "results",
}