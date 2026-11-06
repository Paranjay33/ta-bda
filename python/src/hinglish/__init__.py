"""Hinglish Sentiment Analysis - shared Python package.

Sub-modules:
    config              central config + canonical label mapping loader
    seeding             deterministic seeding across python/numpy/torch
    normalization       shared, model-agnostic dataset-level text normalization
    dataset             dataset loading, schema detection, validation, hashing
    tokenization        model-specific tokenizer wrappers
    train               generic HF Trainer-based fine-tuning entrypoint (E3/E4)
    evaluate            shared metrics / confusion matrix / manifest evaluation
    error_analysis      structured (not auto-labeled) error-analysis scaffolding
    scalability_compare local Python classical baseline for BDA scale comparison
    manifest            run-manifest construction and environment capture
"""
