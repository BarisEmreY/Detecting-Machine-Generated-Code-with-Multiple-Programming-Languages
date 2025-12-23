from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

# Load Subtask A dataset
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

# Use subsets for a fast baseline
train_ds = ds["train"].select(range(30000))
val_ds = ds["validation"].select(range(5000))

X_train_texts = train_ds["code"]
y_train = np.array(train_ds["label"])

X_val_texts = val_ds["code"]
y_val = np.array(val_ds["label"])

# Character n-gram TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),      # 3- to 5-character grams
    min_df=5,                # ignore very rare patterns
    max_features=100000      # cap feature size
)

# Transform text into numeric feature vectors
X_train = vectorizer.fit_transform(X_train_texts)
X_val = vectorizer.transform(X_val_texts)

# Train classifier
clf = LogisticRegression(max_iter=1000, n_jobs=1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
f1 = f1_score(y_val, y_pred, average="macro")

print(f"Char n-gram Baseline Macro F1: {f1:.4f}")
