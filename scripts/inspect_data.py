from datasets import load_dataset
from collections import Counter

# Load Subtask A dataset (Human vs Machine)
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

# Print dataset structure and splits
print(ds)

# Print the first training example to inspect raw data
example = ds["train"][0]
print("\nFirst training example:")
for k, v in example.items():
    print(f"{k}: {str(v)[:200]}")

# 1) Label distribution in the training set
# This shows how many human (0) vs machine-generated (1) samples we have
train_labels = ds["train"]["label"]
print("\nLabel distribution (train):")
print(Counter(train_labels))

# 2) Generator distribution
# This shows which source (human or which LLM) produced the code
train_gens = ds["train"]["generator"]
print("\nGenerator distribution (train):")
print(Counter(train_gens))

# 3) Sanity check: generator-label consistency
# Verify that 'human' corresponds to label 0 and all LLMs correspond to label 1
print("\nSanity check: (generator, label) pairs from first 50 samples:")
pairs = [(ds["train"][i]["generator"], ds["train"][i]["label"]) for i in range(50)]
print(Counter(pairs))

# 4) Programming language distribution (sampled from first 10k examples)
# This gives an idea of which languages dominate the training data
langs = [ds["train"][i]["language"] for i in range(10000)]
print("\nLanguage distribution (first 10k train samples):")
print(Counter(langs))
