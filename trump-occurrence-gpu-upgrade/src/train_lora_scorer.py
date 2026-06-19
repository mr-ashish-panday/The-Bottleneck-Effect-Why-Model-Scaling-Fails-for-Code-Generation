from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.data_utils import checkpoint_prompt, load_corpus_map, load_frame, load_yaml, prediction_frame, set_seed, write_json
from src.metrics import apply_calibrator, fit_calibrator, metric_block


class OccurrenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        corpus: dict[str, dict[str, Any]],
        tokenizer,
        max_prefix_words: int,
        max_length: int,
        blank_prefix: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_prefix_words = max_prefix_words
        self.max_length = max_length
        self.blank_prefix = blank_prefix

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = checkpoint_prompt(row, self.corpus, self.max_prefix_words, blank_prefix=self.blank_prefix)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        enc["labels"] = int(row["label_occurs_after"])
        return enc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v2.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default="runs/lora_scorer")
    parser.add_argument("--row-cap-train", type=int, default=None)
    parser.add_argument("--row-cap-val", type=int, default=None)
    parser.add_argument("--row-cap-test", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def collate(tokenizer):
    def _collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return _collate


def predict_probs(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    pred = trainer.predict(dataset)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.clip(probs[:, 1], 1e-6, 1 - 1e-6)


def metric_against_optional_baseline(
    labels: pd.DataFrame,
    preds: pd.DataFrame,
    baseline_path: str | None,
    bins: int,
) -> dict[str, Any]:
    out = {"model": metric_block(labels, preds, bins=bins)}
    if baseline_path:
        path = Path(baseline_path)
        if path.exists():
            baseline = pd.read_parquet(path)
            if len(labels) != len(baseline):
                keys = ["transcript_id", "target", "t_pct"]
                baseline = labels[keys].merge(baseline, on=keys, how="left")
            out["timing_baseline"] = metric_block(labels, baseline, bins=bins)
            out["brier_improvement_over_timing"] = (
                out["timing_baseline"]["brier"] - out["model"]["brier"]
            )
    return out


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["seed"])
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    model_name = args.model_name or config["model"]["model_name"]
    train_df = load_frame(config["data"]["train_path"], args.row_cap_train, seed)
    val_df = load_frame(config["data"]["val_path"], args.row_cap_val, seed + 1)
    test_path = config["data"].get("test_path")
    test_df = load_frame(test_path, args.row_cap_test, seed + 2) if test_path else None
    corpus = load_corpus_map(config["data"]["corpus_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    load_in_4bit = bool(config["model"].get("load_in_4bit", True)) and not args.no_4bit
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config["model"]["lora_r"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        target_modules=list(config["model"]["target_modules"]),
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_prefix_words = int(config["data"]["max_prefix_words"])
    max_length = int(config["model"]["max_length"])
    train_ds = OccurrenceDataset(train_df, corpus, tokenizer, max_prefix_words, max_length)
    val_ds = OccurrenceDataset(val_df, corpus, tokenizer, max_prefix_words, max_length)
    val_blanked_ds = OccurrenceDataset(val_df, corpus, tokenizer, max_prefix_words, max_length, blank_prefix=True)
    test_ds = OccurrenceDataset(test_df, corpus, tokenizer, max_prefix_words, max_length) if test_df is not None else None
    test_blanked_ds = (
        OccurrenceDataset(test_df, corpus, tokenizer, max_prefix_words, max_length, blank_prefix=True)
        if test_df is not None
        else None
    )

    eval_strategy = str(config["training"].get("eval_strategy", "steps"))
    save_strategy = str(config["training"].get("save_strategy", "steps"))
    train_args = TrainingArguments(
        output_dir=str(output_dir / "hf_checkpoints"),
        num_train_epochs=float(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        weight_decay=float(config["training"]["weight_decay"]),
        logging_steps=int(config["training"]["logging_steps"]),
        eval_strategy=eval_strategy,
        eval_steps=int(config["training"]["eval_steps"]),
        save_strategy=save_strategy,
        save_steps=int(config["training"]["save_steps"]),
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", False)),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate(tokenizer),
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    raw_val = predict_probs(trainer, val_ds)
    calibrator = fit_calibrator(
        raw_val,
        val_df["label_occurs_after"].astype(int).to_numpy(),
        method=config["evaluation"]["calibration_method"],
    )
    p_val = apply_calibrator(calibrator, raw_val)
    val_preds = prediction_frame(val_df, p_val, "gpu_lora_full")
    val_preds.to_parquet(output_dir / "predictions" / "val_gpu_lora_full.parquet", index=False)
    bins = int(config["evaluation"]["ece_bins"])
    raw_val_blanked = predict_probs(trainer, val_blanked_ds)
    p_val_blanked = apply_calibrator(calibrator, raw_val_blanked)
    val_blanked_preds = prediction_frame(val_df, p_val_blanked, "gpu_lora_blanked")
    val_blanked_preds.to_parquet(output_dir / "predictions" / "val_gpu_lora_blanked.parquet", index=False)
    val_metrics = metric_against_optional_baseline(
        val_df,
        val_preds,
        config["data"].get("val_timing_predictions_path"),
        bins=bins,
    )
    val_blanked_metrics = metric_against_optional_baseline(
        val_df,
        val_blanked_preds,
        config["data"].get("val_timing_predictions_path"),
        bins=bins,
    )
    write_json(
        output_dir / "metrics" / "val_gpu_lora_metrics.json",
        {
            "full": val_metrics,
            "blanked": val_blanked_metrics,
            "content_brier_delta_full_minus_blanked": (
                val_metrics["model"]["brier"] - val_blanked_metrics["model"]["brier"]
            ),
            "note": "Negative content_brier_delta_full_minus_blanked means full has lower Brier than blanked.",
        },
    )

    if test_ds is not None and test_blanked_ds is not None and test_df is not None:
        raw_test = predict_probs(trainer, test_ds)
        p_test = apply_calibrator(calibrator, raw_test)
        test_preds = prediction_frame(test_df, p_test, "gpu_lora_full")
        test_preds.to_parquet(output_dir / "predictions" / "test_gpu_lora_full.parquet", index=False)
        raw_test_blanked = predict_probs(trainer, test_blanked_ds)
        p_test_blanked = apply_calibrator(calibrator, raw_test_blanked)
        test_blanked_preds = prediction_frame(test_df, p_test_blanked, "gpu_lora_blanked")
        test_blanked_preds.to_parquet(output_dir / "predictions" / "test_gpu_lora_blanked.parquet", index=False)
        test_metrics = metric_against_optional_baseline(
            test_df,
            test_preds,
            config["data"].get("test_timing_predictions_path"),
            bins=bins,
        )
        test_blanked_metrics = metric_against_optional_baseline(
            test_df,
            test_blanked_preds,
            config["data"].get("test_timing_predictions_path"),
            bins=bins,
        )
        write_json(
            output_dir / "metrics" / "test_gpu_lora_metrics.json",
            {
                "full": test_metrics,
                "blanked": test_blanked_metrics,
                "content_brier_delta_full_minus_blanked": (
                    test_metrics["model"]["brier"] - test_blanked_metrics["model"]["brier"]
                ),
                "note": "Negative content_brier_delta_full_minus_blanked means full has lower Brier than blanked.",
            },
        )

    trainer.model.save_pretrained(output_dir / "models" / "adapter")
    tokenizer.save_pretrained(output_dir / "models" / "tokenizer")
    with (output_dir / "models" / "calibrator.pkl").open("wb") as f:
        pickle.dump(calibrator, f)

    run_meta = {
        "model_name": model_name,
        "seed": seed,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)) if test_df is not None else 0,
        "train_seconds": train_seconds,
        "load_in_4bit": load_in_4bit,
        "max_prefix_words": max_prefix_words,
        "max_length": max_length,
        "note": "If this uses the V1 validation split only, treat it as exploratory until a fresh sealed V2 test exists.",
    }
    write_json(output_dir / "run_meta.json", run_meta)
    print("val_brier", val_metrics["model"]["brier"])
    if "timing_baseline" in val_metrics:
        print("val_timing_brier", val_metrics["timing_baseline"]["brier"])
        print("val_brier_improvement_over_timing", val_metrics["brier_improvement_over_timing"])
    print("output_dir", output_dir)


if __name__ == "__main__":
    main()
