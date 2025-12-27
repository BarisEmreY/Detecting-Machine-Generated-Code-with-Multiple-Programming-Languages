"""
Task C transformer training: GraphCodeBERT/CodeBERT fine-tuning for 4-class classification.

Why transformer?
- Captures deeper syntax/semantic patterns than n-grams
- Usually improves hybrid/adversarial separation

Designed for Mac MPS constraints:
- subset training
- max_steps control
- max_length default 256
"""

import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def language_wise_macro_f1(langs, y_true, y_pred):
    by_lang = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for l, yt, yp in zip(langs, y_true, y_pred):
        by_lang[l]["y_true"].append(yt)
        by_lang[l]["y_pred"].append(yp)

    out = {}
    for l, d in by_lang.items():
        out[l] = macro_f1(d["y_true"], d["y_pred"])
    return out


def append_results_csv(task_id, model_name, train_size, val_size, macro, lang_scores, notes, out_path="reports/results.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")

    langs = ["Python", "Java", "Go", "JavaScript", "C#", "C++", "C", "PHP"]
    row = [ts, task_id, model_name, str(train_size), str(val_size), f"{macro:.6f}"]
    for l in langs:
        row.append(f"{lang_scores.get(l, float('nan')):.6f}" if l in lang_scores else "")
    row.append(notes)

    header = ["timestamp", "task", "model", "train_size", "val_size", "macro_f1"] + \
             [f"f1_{l}" for l in langs] + ["notes"]

    file_exists = Path(out_path).exists()
    with open(out_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(row) + "\n")


def compute_class_weights(labels, num_labels):
    counts = Counter(labels)
    total = sum(counts.values())
    # Inverse frequency weights
    w = []
    for i in range(num_labels):
        c = counts.get(i, 1)
        w.append(total / (num_labels * c))
    w = torch.tensor(w, dtype=torch.float)
    return w


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_training_args(**kwargs):
    """
    Transformers renamed some args in newer versions.
    We try 'evaluation_strategy' first, fallback to 'eval_strategy' if needed.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if "evaluation_strategy" in str(e) and "unexpected keyword" in str(e):
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
            return TrainingArguments(**kwargs)
        raise


def main():
    p = argparse.ArgumentParser()

    # Data / subset
    p.add_argument("--train_size", type=int, default=30000)
    p.add_argument("--val_size", type=int, default=5000)

    # Balanced sampling (recommended for Task C)
    p.add_argument("--balanced", action="store_true", help="Use class-balanced sampling for training set")
    p.add_argument("--per_class", type=int, default=5000, help="Samples per class when --balanced is enabled")

    # Model
    p.add_argument("--model", type=str, default="microsoft/graphcodebert-base")
    p.add_argument("--max_length", type=int, default=256)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # Loss
    p.add_argument("--weighted_loss", action="store_true", help="Use class-weighted CrossEntropyLoss")

    # Output
    p.add_argument("--out_dir", type=str, default="runs/graphcodebert_taskC")

    args = p.parse_args()
    set_seed(args.seed)

    # Load Task C
    ds = load_dataset("DaniilOr/SemEval-2026-Task13", "C")
    train_ds = ds["train"].select(range(min(args.train_size, len(ds["train"]))))
    val_ds = ds["validation"].select(range(min(args.val_size, len(ds["validation"]))))

    # Optional balanced sampling (uses your src/common/data.py helper)
    if args.balanced:
        from src.common.data import balanced_subset
        train_ds, counts = balanced_subset(train_ds, label_col="label", per_class=args.per_class, seed=args.seed)
        print("[Balanced Sampling] per-class counts:", dict(sorted(counts.items())))
        print("[Balanced Sampling] total train size:", len(train_ds))

    num_labels = 4

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["code"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ("label", "language", "generator")])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=[c for c in val_ds.column_names if c not in ("label", "language", "generator")])

    # Trainer expects 'labels'
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    # Compute class weights if enabled
    class_weights = None
    if args.weighted_loss:
        class_weights = compute_class_weights(train_ds["label"], num_labels=num_labels)
        print("[Weighted Loss] class_weights:", class_weights.tolist())

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": macro_f1(labels, preds)}

    train_args = build_training_args(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps != -1 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        fp16=False,
    )

    TrainerCls = WeightedLossTrainer if args.weighted_loss else Trainer
    trainer = TrainerCls(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights if args.weighted_loss else None,
    )

    trainer.train()

    # Final evaluation + language-wise
    pred_out = trainer.predict(val_tok)
    val_logits = pred_out.predictions
    val_labels = pred_out.label_ids
    val_preds = np.argmax(val_logits, axis=1)

    final_macro = macro_f1(val_labels, val_preds)
    print(f"\nTask C Transformer Macro F1: {final_macro:.4f} (num_labels={num_labels})")

    # Language-wise (from original val_ds)
    lw = language_wise_macro_f1(val_ds["language"], val_ds["label"], val_preds)
    print("\nLanguage-wise Macro F1 scores:")
    for k in sorted(lw.keys()):
        print(f"{k}: Macro F1 = {lw[k]:.4f}")

    # Save model
    trainer.save_model(args.out_dir)
    print(f"Saved model to: {args.out_dir}")

    append_results_csv(
        task_id="C",
        model_name=f"{args.model}_transformer",
        train_size=len(train_ds),
        val_size=len(val_ds),
        macro=final_macro,
        lang_scores=lw,
        notes=f"max_length={args.max_length}, max_steps={args.max_steps}, lr={args.lr}, balanced={args.balanced}, per_class={args.per_class}, weighted_loss={args.weighted_loss}",
    )
    print("Saved results to reports/results.csv")


if __name__ == "__main__":
    main()
