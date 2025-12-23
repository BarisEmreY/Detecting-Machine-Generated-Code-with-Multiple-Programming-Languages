from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

# Load Subtask A dataset
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

# Use a small subset for the first baseline (speed + simplicity)
train_ds = ds["train"].select(range(20000))
val_ds = ds["validation"].select(range(5000))

def extract_features(code):
    """Extract simple surface-level features from code."""
    return [
        len(code),                    # total characters
        code.count("\n"),             # number of lines
        code.count(" "),              # spaces
        sum(code.count(c) for c in "{}();,")  # punctuation count
    ]

# Build feature matrices
X_train = np.array([extract_features(x["code"]) for x in train_ds])
y_train = np.array(train_ds["label"])

X_val = np.array([extract_features(x["code"]) for x in val_ds])
y_val = np.array(val_ds["label"])

# Train a simple classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate on validation set
y_pred = clf.predict(X_val)
f1 = f1_score(y_val, y_pred, average="macro")

print(f"Baseline Macro F1: {f1:.4f}")
