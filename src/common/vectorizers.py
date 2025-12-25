from sklearn.feature_extraction.text import TfidfVectorizer


def char_ngram_vectorizer(
    ngram_range=(3, 5),
    max_features=100_000,
    min_df=5
):
    """
    Character-level TF-IDF vectorizer for source code.
    Uses char n-grams to capture formatting and stylistic patterns.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=False
    )
