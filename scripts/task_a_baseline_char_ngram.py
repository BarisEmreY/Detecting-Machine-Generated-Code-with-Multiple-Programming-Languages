import numpy as np
from sklearn.linear_model import LogisticRegression
from src.common.logging import log_results
from src.common.data import load_task
from src.common.vectorizers import char_ngram_vectorizer
from src.common.eval import macro_f1, language_wise_f1
from src.tasks.task_a import TASK_ID, TEXT_COLUMN, LABEL_COLUMN

train_ds, val_ds, _ = load_task(TASK_ID, train_size=30000, val_size=5000)

X_train_texts = train_ds[TEXT_COLUMN]
y_train = np.array(train_ds[LABEL_COLUMN])

X_val_texts = val_ds[TEXT_COLUMN]
y_val = np.array(val_ds[LABEL_COLUMN])
val_langs = val_ds["language"]

vectorizer = char_ngram_vectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_val = vectorizer.transform(X_val_texts)

clf = LogisticRegression(max_iter=1000, n_jobs=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
f1 = macro_f1(y_val, y_pred)
print(f"Task A Char n-gram Baseline Macro F1: {f1:.4f}")

lang_results = language_wise_f1(clf, X_val, y_val, val_langs)
print("\nLanguage-wise Macro F1 scores:")
for lang, (score, n) in lang_results.items():
    print(f"{lang}: Macro F1 = {score:.4f} (n={n})")
    
log_results(
    csv_path="reports/results.csv",
    task_id=TASK_ID,
    model_name="char_ngram_tfidf_logreg",
    train_size=30000,
    val_size=5000,
    overall_f1=f1,
    lang_f1=lang_results,
    notes="TF-IDF char 3-5grams, max_features=100k, min_df=5"
)
print("\nSaved results to reports/results.csv")
