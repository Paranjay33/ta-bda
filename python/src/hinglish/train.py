"""Generic Transformer fine-tuning entrypoint for E3 (xlm-roberta-base) and
E4 (l3cube-pune/hing-roberta).

Both experiments MUST run through this exact script with only
``--model-name-or-path`` / ``--experiment-id`` differing, so dataset split,
label mapping, training framework, evaluation logic, model-selection metric,
seed policy and max-sequence-length policy are identical (matched conditions).

Data: the frozen processed split (data/processed/*.tsv, built by
``python -m hinglish.prepare_data``), column ``normalized_text``.

Model selection uses validation macro-F1 ONLY. The test split is never read
here unless ``--final-test-eval`` is passed explicitly, which delegates to
``hinglish.evaluate_locked`` AFTER training/selection/calibration are frozen.

Usage:
    python -m hinglish.train --experiment-id E3 --model-name-or-path FacebookAI/xlm-roberta-base
    python -m hinglish.train --experiment-id E4 --model-name-or-path l3cube-pune/hing-roberta
    (add --smoke-test for a tiny CPU-safe pass that writes to a separate run dir)

Changes vs. the midterm version (documented in docs/EXPERIMENTS.md):
  * reads the verified, de-duplicated split (the supplied split leaked);
  * ``warmup_ratio`` from config.yaml is applied again (it was commented out);
  * output goes to artifacts/models/<exp>/<run_tag>/ so the legacy midterm E3
    checkpoints are never overwritten or checkpoint-rotated away;
  * temperature calibration is fitted on validation logits after selection.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .calibration import fit_temperature, probabilistic_metrics, softmax
from .config import load_config
from .evaluate import compute_classification_metrics
from .manifest import Timer, capture_environment
from .prepare_data import load_processed_split, sha256_file, split_identity
from .seeding import set_seed
from .tokenization import load_tokenizer

EXPECTED_CHECKPOINTS = {"E3": "FacebookAI/xlm-roberta-base", "E4": "l3cube-pune/hing-roberta"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-id", required=True, choices=["E3", "E4"])
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--run-tag", default=None, help="defaults to transformer.run_tag in config.yaml")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint in the run dir")
    p.add_argument("--final-test-eval", action="store_true",
                   help="after training, run the LOCKED test evaluation (hinglish.evaluate_locked)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    labels = cfg.labels
    seed = int(cfg.get("seed", default=42))
    set_seed(seed)

    expected = EXPECTED_CHECKPOINTS[args.experiment_id]
    if args.model_name_or_path != expected:
        raise SystemExit(
            f"{args.experiment_id} is defined as {expected!r}; refusing to train it from "
            f"{args.model_name_or_path!r}. Register a separately named experiment instead."
        )

    t_cfg = cfg.get("transformer", default={})
    max_length = int(t_cfg.get("max_seq_length", 128))
    run_tag = args.run_tag or t_cfg.get("run_tag", "run")
    if args.smoke_test:
        run_tag = f"{run_tag}-smoke"
        num_epochs = int(cfg.get("smoke_test", "num_epochs", default=1))
    else:
        num_epochs = int(t_cfg.get("num_epochs", 5))

    train_df = load_processed_split(cfg, "train")
    val_df = load_processed_split(cfg, "validation")
    if args.smoke_test:
        n = int(cfg.get("smoke_test", "max_examples", default=64))
        train_df, val_df = train_df.head(n), val_df.head(max(n // 4, 8))

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tok_bundle = load_tokenizer(args.model_name_or_path, max_length)

    def to_hf(df):
        ds = Dataset.from_dict({"text": df["normalized_text"].tolist(),
                                "label": df["label_id"].astype(int).tolist()})
        return ds.map(lambda b: tok_bundle.encode_batch(b["text"]), batched=True)

    train_ds, val_ds = to_hf(train_df), to_hf(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=labels.num_classes,
        id2label=labels.id_to_label,
        label2id=labels.label_to_id,
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        m = compute_classification_metrics(y_true.tolist(), y_pred.tolist(), labels.id_to_label)
        return {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "weighted_f1": m.weighted_f1}

    run_dir = cfg.path("paths", "models_dir") / args.experiment_id.lower() / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = bool(t_cfg.get("fp16_if_available", True)) and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        seed=seed,
        data_seed=seed,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=int(t_cfg.get("train_batch_size", 16)),
        per_device_eval_batch_size=int(t_cfg.get("eval_batch_size", 32)),
        learning_rate=float(t_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(t_cfg.get("warmup_ratio", 0.06)),
        optim=str(t_cfg.get("optimizer", "adamw_torch")),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        report_to=[],
        save_total_limit=1,
        dataloader_num_workers=0,
        # safetensors 0.8.0 + torch 2.5.1 in this venv segfaults (Windows access
        # violation in safetensors.torch._to_ndarray) when the Trainer saves a
        # checkpoint mid-training. PyTorch .bin serialization is used instead so
        # the user's existing environment does not have to be changed.
        save_safetensors=bool(t_cfg.get("save_safetensors", False)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tok_bundle.tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(t_cfg.get("early_stopping_patience", 2)))],
    )

    started = time.time()
    with Timer() as timer:
        trainer.train(resume_from_checkpoint=True if args.resume else None)

    # --- selection is done (best validation macro-F1 checkpoint is loaded) ---
    # Free optimizer/scheduler state before inference: on a 4 GB laptop GPU the
    # Adam moments otherwise leave no room for the validation forward passes.
    import gc

    trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_out = trainer.predict(val_ds)
    val_logits = np.asarray(val_out.predictions, dtype=np.float64)
    y_val = np.asarray(val_df["label_id"].astype(int).tolist())
    val_pred = val_logits.argmax(axis=1)
    val_metrics = compute_classification_metrics(y_val.tolist(), val_pred.tolist(), labels.id_to_label).to_dict()
    calib = fit_temperature(val_logits, y_val)
    val_metrics["probabilistic_uncalibrated"] = probabilistic_metrics(softmax(val_logits), y_val)
    val_metrics["probabilistic_calibrated"] = probabilistic_metrics(calib.apply(val_logits), y_val)

    best_dir = run_dir / "best"
    trainer.model.save_pretrained(str(best_dir), safe_serialization=training_args.save_safetensors)
    tok_bundle.tokenizer.save_pretrained(str(best_dir))
    calib.save(best_dir / "calibration.json")

    best_ckpt = trainer.state.best_model_checkpoint or ""
    m = re.search(r"checkpoint-(\d+)", best_ckpt)
    best_step = int(m.group(1)) if m else None
    epoch_log = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    selected_epoch = next((h["epoch"] for h in epoch_log if h.get("step") == best_step), None)

    weights = sorted(best_dir.glob("*.safetensors")) or sorted(best_dir.glob("*.bin"))
    artifact_hashes = {w.name: sha256_file(w) for w in weights}
    artifact_hashes["calibration.json"] = sha256_file(best_dir / "calibration.json")
    model_bytes = sum(w.stat().st_size for w in weights)

    hyper = {
        "max_seq_length": max_length,
        "num_epochs": num_epochs,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "early_stopping_patience": int(t_cfg.get("early_stopping_patience", 2)),
        "fp16": use_fp16,
        "optimizer": str(training_args.optim),
        "model_selection_metric": "validation macro_f1",
        "smoke_test": args.smoke_test,
    }

    manifest = {
        "schema_version": 2,
        "experiment_id": args.experiment_id,
        "display_name": {"E3": "XLM-RoBERTa (generic multilingual)",
                         "E4": "HingRoBERTa (Hindi-English domain-adapted)"}[args.experiment_id],
        "family": "transformer",
        "framework": "pytorch+huggingface",
        "run_tag": run_tag,
        "version": f"{args.experiment_id.lower()}-{run_tag}",
        "source_checkpoint": args.model_name_or_path,
        "source_checkpoint_revision": _hub_revision(args.model_name_or_path),
        "seed": seed,
        "dataset": split_identity(cfg),
        "hyperparameters": hyper,
        "training_log": epoch_log,
        "selected_checkpoint_step": best_step,
        "selected_epoch": selected_epoch,
        "validation_metrics": val_metrics,
        "calibration": {"method": calib.method, "temperature": calib.temperature,
                        "fitted_on": "validation", "file": "best/calibration.json"},
        "score_semantics": "calibrated_probability (temperature scaling fitted on validation)",
        "artifact_dir": str(best_dir.relative_to(cfg.root)).replace("\\", "/"),
        "artifact_sha256": artifact_hashes,
        "artifact_size_bytes": model_bytes,
        "training_runtime_seconds": timer.elapsed,
        "environment": capture_environment(cfg.root),
        "status": {"training": "completed", "selection": "validation_macro_f1",
                   "locked_test_evaluation": "not_run"},
        "notes": ["Model selection and calibration used the validation split only; the test split was not read."],
        "created_at_unix": started,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    metrics_dir = cfg.path("paths", "metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke_test:
        with open(metrics_dir / f"{args.experiment_id.lower()}_validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"experiment_id": args.experiment_id, "split": "validation",
                       "model_version": manifest["version"], "metrics": val_metrics}, f, indent=2)
    print(f"[hinglish.train] {args.experiment_id} done: val macro-F1={val_metrics['macro_f1']:.4f} "
          f"(epoch {selected_epoch}), T={calib.temperature:.3f}. Manifest -> {run_dir / 'manifest.json'}")

    if args.final_test_eval:
        from .evaluate_locked import main as eval_main
        return eval_main(["--experiment-id", args.experiment_id, "--run-tag", run_tag])
    return 0


def _hub_revision(model_id: str) -> str | None:
    """Resolve the commit hash of the cached Hub snapshot actually used."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                return revs[0].commit_hash if revs else None
    except Exception:
        return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
