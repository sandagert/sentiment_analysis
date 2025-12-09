# ============================================================================
# Baseline configuration - No prompts, just raw reviews
# ============================================================================

TEST_MODE = False  # Set to False for full evaluation

TEST_CONFIG = {
    "eval_samples": 50,  # Use 50 samples in test mode
}

# ============================================================================
# MODEL PATHS (same as main config)
# ============================================================================

BERT_CONFIG = {
    "model_dir": "models/bert/final",
    "tokenized_data_path": "data/processed/bert_tokenized",
}

ROBERTA_CONFIG = {
    "model_dir": "models/roberta/final",
    "tokenized_data_path": "data/processed/roberta_tokenized",
}

LLAMA_CONFIG = {
    "model_dir": "models/llama/final",
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVAL_CONFIG = {
    "test_size": None,  # Full test size (ignored in TEST_MODE)
    "results_dir": "baseline/results",  # Save baseline results separately
}