import argparse
import numpy as np
from collections import defaultdict
import torch
from collections import Counter
from src.common.trainer import WeightedLossTrainer
from src.common.data import balanced_subset


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.common.data import load_task
from src.common.eval import macro_f1
from src.common.logging import log_results
from src.tasks.task_b import TASK_ID, TEXT_COLUMN, LABEL_COLUMN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="microsoft/codebert-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--train_size", type=int, default=30000)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--train_bs", type=int, default=4)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--balanced", action="store_true", help="Use class-balanced sampling for training set")
    p.add_argument("--per_class", type=int, default=3000, help="Samples per class when --balanced is enabled")
    return p.parse_args()


def main():
    args_cli = parse_args()

    MODEL_NAME = args_cli.model
    MAX_LENGTH = args_cli.max_length
    TRAIN_SIZE = args_cli.train_size
    VAL_SIZE = args_cli.val_size
    EPOCHS = args_cli.epochs
    SEED = args_cli.seed

    # 1) Load data
    train_ds, val_ds, _ = load_task(TASK_ID, train_size=TRAIN_SIZE, val_size=VAL_SIZE)
    if args_cli.balanced:
        train_ds, counts = balanced_subset(train_ds, label_col="label", per_class=args_cli.per_class, seed=args_cli.seed)
        print("[Balanced Sampling] per-class counts:", dict(sorted(counts.items())))
        print("[Balanced Sampling] total train size:", len(train_ds))
    else:
    # if you already use train_size slicing, keep it here as before
     pass

     if args_cli.balanced and args_cli.train_size != -1:
        print("[Info] --train_size is ignored because --balanced is enabled.")

    # 2) Infer num_labels from data (labels are int64, not ClassLabel)
    train_labels = train_ds[LABEL_COLUMN]
    num_labels = int(max(train_labels)) + 1
    # Compute class weights from label frequencies (inverse frequency)
    counts = Counter(train_labels)
    freq = torch.tensor([counts.get(i, 0) for i in range(num_labels)], dtype=torch.float)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()  # normalize for stability


    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    # 4) Tokenize datasets
    train_tok = train_ds.map(tokenize_batch, batched=True)
    val_tok = val_ds.map(tokenize_batch, batched=True)

    # Keep only what Trainer needs
    keep_cols = ["input_ids", "attention_mask", LABEL_COLUMN, "language"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in keep_cols])

    train_tok = train_tok.rename_column(LABEL_COLUMN, "labels")
    val_tok = val_tok.rename_column(LABEL_COLUMN, "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # 6) Training config
    out_dir = f"checkpoints/task_b_{MODEL_NAME.replace('/', '_')}_len{MAX_LENGTH}_seed{SEED}"
    train_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=args_cli.train_bs,
        per_device_eval_batch_size=args_cli.eval_bs,
        gradient_accumulation_steps=args_cli.grad_accum,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        max_steps=(args_cli.max_steps if args_cli.max_steps and args_cli.max_steps > 0 else -1),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": macro_f1(labels, preds)}

    trainer = WeightedLossTrainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=weights,
)


    # 7) Train
    trainer.train()

    # 8) Evaluate overall (full val once)
    eval_out = trainer.predict(val_tok)
    logits = eval_out.predictions
    y_true = eval_out.label_ids
    y_pred = np.argmax(logits, axis=-1)
    overall = macro_f1(y_true, y_pred)

    print(f"\nTask B Transformer Macro F1: {overall:.4f} (num_labels={num_labels})")

    # 9) Language-wise (no extra predict calls; reuse y_true/y_pred)
    val_langs = val_ds["language"]
    idx_map = defaultdict(list)
    for i, lang in enumerate(val_langs):
        idx_map[lang].append(i)

    lang_scores = {}
    print("\nLanguage-wise Macro F1 scores:")
    for lang, indices in idx_map.items():
        score = macro_f1(y_true[indices], y_pred[indices])
        lang_scores[lang] = (score, len(indices))
        print(f"{lang}: Macro F1 = {score:.4f} (n={len(indices)})")

    # 10) Log results
    log_results(
        csv_path="reports/results.csv",
        task_id=TASK_ID,
        model_name=f"transformer_{MODEL_NAME.replace('/', '_')}_maxlen{MAX_LENGTH}_seed{SEED}",
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        overall_f1=overall,
        lang_f1=lang_scores,
        notes=f"epochs={EPOCHS} lr=2e-5 bs={args_cli.train_bs} grad_accum={args_cli.grad_accum} max_length={MAX_LENGTH} num_labels={num_labels}"
    )
    print("\nSaved results to reports/results.csv")


if __name__ == "__main__":
    main()
