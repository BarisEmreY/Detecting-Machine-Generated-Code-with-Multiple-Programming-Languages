import numpy as np
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

MODEL_NAME = "microsoft/codebert-base"  # good starting point
MAX_LENGTH = 256                        # start small for speed, can increase later

TRAIN_SIZE = 30000
VAL_SIZE = 5000


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
    output_dir="checkpoints/task_a_codebert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
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
    for lang, indices in idx_map.items():
        subset = val_tok.select(indices)
        out = trainer.predict(subset)
        preds = np.argmax(out.predictions, axis=-1)
        score = macro_f1(out.label_ids, preds)
        lang_scores[lang] = (score, len(indices))
        print(f"{lang}: Macro F1 = {score:.4f} (n={len(indices)})")

    # 9) Log results
    log_results(
        csv_path="reports/results.csv",
        task_id=TASK_ID,
        model_name=f"transformer_{MODEL_NAME.replace('/', '_')}_maxlen{MAX_LENGTH}",
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        overall_f1=overall,
        lang_f1=lang_scores,
        notes=f"epochs=2 lr=2e-5 bs=8 max_length={MAX_LENGTH}"
    )
    print("\nSaved results to reports/results.csv")


if __name__ == "__main__":
    main()
