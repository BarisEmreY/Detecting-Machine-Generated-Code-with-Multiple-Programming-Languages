from collections import defaultdict
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    """Compute Macro F1 score."""
    return f1_score(y_true, y_pred, average="macro")


def language_wise_f1(model, X_val, y_val, languages):
    """
    Compute Macro F1 per language.

    Parameters
    ----------
    model : fitted sklearn-like model with .predict()
    X_val : vectorized validation features (supports slicing with indices)
    y_val : numpy array of labels aligned with X_val
    languages : list/array of language strings aligned with X_val
    """
    lang_to_indices = defaultdict(list)
    for idx, lang in enumerate(languages):
        lang_to_indices[lang].append(idx)

    results = {}
    for lang, indices in lang_to_indices.items():
        y_pred_lang = model.predict(X_val[indices])
        results[lang] = (macro_f1(y_val[indices], y_pred_lang), len(indices))

    return results
