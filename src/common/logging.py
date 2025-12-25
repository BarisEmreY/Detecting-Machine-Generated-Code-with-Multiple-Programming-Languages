import csv
import os
from datetime import datetime

def log_results(
    csv_path: str,
    task_id: str,
    model_name: str,
    train_size: int,
    val_size: int,
    overall_f1: float,
    lang_f1=None,
    notes: str = ""
):
    """
    Append experiment results to a CSV file.
    lang_f1 can be:
      - {"Python": (f1, n), "C++": (f1, n), ...}
      - {"Python": f1, "C++": f1, ...}
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Normalize language scores to plain floats
    lang_scores = {}
    if lang_f1:
        for k, v in lang_f1.items():
            if isinstance(v, tuple):
                lang_scores[k] = float(v[0])
            else:
                lang_scores[k] = float(v)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_id,
        "model": model_name,
        "train_size": train_size,
        "val_size": val_size,
        "macro_f1": round(float(overall_f1), 6),
        "python_f1": round(float(lang_scores.get("Python", "")), 6) if "Python" in lang_scores else "",
        "cpp_f1": round(float(lang_scores.get("C++", "")), 6) if "C++" in lang_scores else "",
        "java_f1": round(float(lang_scores.get("Java", "")), 6) if "Java" in lang_scores else "",
        "notes": notes,
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
