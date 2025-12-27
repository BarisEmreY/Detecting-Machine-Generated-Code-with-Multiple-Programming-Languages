"""
Task C baseline: TF-IDF character n-grams + Logistic Regression.

Why this baseline?
- Very fast
- Strong for code-style patterns (indentation, symbols, keywords)
- Provides a solid reference before transformer fine-tuning
"""

import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def append_results_csv(
    task_id: str,
    model_name: str,
    train_size: int,
    val_size: int,
    macro: float,
    lang_scores: dict,
    notes: str,
    out_path: str = "reports/results.csv",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")

    # Keep the same "one-row, language columns" style you had before.
    # If some languages missing, they will stay empty.
    langs = ["Python", "Java", "Go", "JavaScript", "C#", "C++", "C", "PHP"]

    row = [
        ts,
        task_id,
        model_name,
        str(train_size),
        str(val_size),
        f"{macro:.6f}",
    ]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=30000, help="Subset size from train split")
    p.add_argument("--val_size", type=int, default=5000, help="Subset size from validation split")
    p.add_argument("--max_features", type=int, default=200000)
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--ngram_min", type=int, default=3)
    p.add_argument("--ngram_max", type=int, default=5)
    args = p.parse_args()

    ds = load_dataset("DaniilOr/SemEval-2026-Task13", "C")
    train_ds = ds["train"].select(range(min(args.train_size, len(ds["train"]))))
    val_ds = ds["validation"].select(range(min(args.val_size, len(ds["validation"]))))

    X_train_text = train_ds["code"]
    y_train = train_ds["label"]

    X_val_text = val_ds["code"]
    y_val = val_ds["label"]
    val_langs = val_ds["language"]

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
    )
    X_train = vec.fit_transform(X_train_text)
    X_val = vec.transform(X_val_text)

    # class_weight="balanced" helps with imbalanced Task C labels
    clf = LogisticRegression(max_iter=3000, n_jobs=1, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)

    score = macro_f1(y_val, pred)
    print(f"Task C Char n-gram Baseline Macro F1: {score:.4f}")

    lw = language_wise_macro_f1(val_langs, y_val, pred)
    print("\nLanguage-wise Macro F1 scores:")
    for k in sorted(lw.keys()):
        print(f"{k}: Macro F1 = {lw[k]:.4f}")

    append_results_csv(
        task_id="C",
        model_name="char_ngram_tfidf_logreg",
        train_size=len(train_ds),
        val_size=len(val_ds),
        macro=score,
        lang_scores=lw,
        notes=f"TF-IDF char {args.ngram_min}-{args.ngram_max}grams, max_features={args.max_features}, min_df={args.min_df}, class_weight=balanced",
    )
    print("Saved results to reports/results.csv")


if __name__ == "__main__":
    main()
