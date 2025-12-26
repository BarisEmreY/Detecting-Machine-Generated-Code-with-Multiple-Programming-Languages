import numpy as np
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset

from src.common.data import load_task
from src.common.eval import macro_f1, language_wise_f1
from src.common.logging import log_results
from src.tasks.task_a import TASK_ID, TEXT_COLUMN, LABEL_COLUMN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="microsoft/codebert-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--train_size", type=int, default=30000)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1)
    return p.parse_args()

args_cli = parse_args()

MODEL_NAME = args_cli.model
MAX_LENGTH = args_cli.max_length
TRAIN_SIZE = args_cli.train_size
VAL_SIZE = args_cli.val_size
EPOCHS = args_cli.epochs
SEED = args_cli.seed


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": macro_f1(labels, preds)}


def main():
    # 1) Load data
    train_ds, val_ds, _ = load_task(TASK_ID, train_size=TRAIN_SIZE, val_size=VAL_SIZE)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # 3) Tokenize datasets
    train_tok = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    val_tok = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    # Keep only needed columns for Trainer
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in ["input_ids", "attention_mask", LABEL_COLUMN, "language"]])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in ["input_ids", "attention_mask", LABEL_COLUMN, "language"]])

    train_tok = train_tok.rename_column(LABEL_COLUMN, "labels")
    val_tok = val_tok.rename_column(LABEL_COLUMN, "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 5) Training config
    args = TrainingArguments(
    max_steps=args_cli.max_steps,
    output_dir=f"checkpoints/task_a_{MODEL_NAME.replace('/', '_')}_len{MAX_LENGTH}_seed{SEED}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    seed=SEED,
)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) Train
    trainer.train()

    # 7) Evaluate overall
    eval_out = trainer.predict(val_tok)
    logits = eval_out.predictions
    y_true = eval_out.label_ids
    y_pred = np.argmax(logits, axis=-1)
    overall = macro_f1(y_true, y_pred)

    print(f"\nTask A Transformer Macro F1: {overall:.4f}")

    # 8) Language-wise eval
    # We need "languages" aligned with val order
    val_langs = val_ds["language"]  # original val order matches val_tok order
    # Build a small wrapper so we can reuse language_wise_f1(model, X, y, langs)
    class _PredictWrapper:
        def __init__(self, trainer, dataset):
            self.trainer = trainer
            self.dataset = dataset
        def predict(self, sliced_dataset):
            out = self.trainer.predict(sliced_dataset)
            return np.argmax(out.predictions, axis=-1)

    wrapper = _PredictWrapper(trainer, val_tok)

    # language_wise_f1 expects slicing like X_val[indices]
    # HuggingFace datasets support select(indices)
    lang_scores = {}
    from collections import defaultdict
    idx_map = defaultdict(list)
    for i, lang in enumerate(val_langs):
        idx_map[lang].append(i)

    print("\nLanguage-wise Macro F1 scores:")
    val_langs = val_ds["language"]

    # We already computed y_pred for full val above (reuse it!)
    # y_true and y_pred are aligned with val order
    from collections import defaultdict
    idx_map = defaultdict(list)
    for i, lang in enumerate(val_langs):
        idx_map[lang].append(i)

    lang_scores = {}
    for lang, indices in idx_map.items():
        score = macro_f1(y_true[indices], y_pred[indices])
        lang_scores[lang] = (score, len(indices))
        print(f"{lang}: Macro F1 = {score:.4f} (n={len(indices)})")


    # 9) Log results
    log_results(
    csv_path="reports/results.csv",
    task_id=TASK_ID,
    model_name=f"transformer_{MODEL_NAME.replace('/', '_')}_maxlen{MAX_LENGTH}_seed{SEED}",
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    overall_f1=overall,
    lang_f1=lang_scores,
    notes=f"epochs={EPOCHS} lr=2e-5 bs=8 max_length={MAX_LENGTH}"
)

    print("\nSaved results to reports/results.csv")


if __name__ == "__main__":
    main()
