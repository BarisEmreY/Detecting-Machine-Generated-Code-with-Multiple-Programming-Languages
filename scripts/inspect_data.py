import argparse
from datasets import load_dataset
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["A", "B", "C"],
                        help="Which task to inspect: A, B, or C")
    parser.add_argument("--lang_sample", type=int, default=10000,
                        help="How many training examples to sample for language distribution")
    parser.add_argument("--sanity_n", type=int, default=50,
                        help="How many samples to use for sanity checks (Task A only)")
    args = parser.parse_args()

    # Load dataset for the chosen task
    ds = load_dataset("DaniilOr/SemEval-2026-Task13", args.task)

    # Print dataset structure and splits
    print(ds)

    # Print the first training example
    example = ds["train"][0]
    print("\nFirst training example:")
    for k, v in example.items():
        print(f"{k}: {str(v)[:200]}")

    # 1) Label distribution
    train_labels = ds["train"]["label"]
    print("\nLabel distribution (train):")
    print(Counter(train_labels))

    # 2) Generator distribution (if exists)
    if "generator" in ds["train"].features:
        train_gens = ds["train"]["generator"]
        print("\nGenerator distribution (train):")
        print(Counter(train_gens))
    else:
        print("\nGenerator distribution (train): <no generator field in this task>")

    # 3) Sanity check (Task A only)
    if args.task == "A" and "generator" in ds["train"].features:
        print(f"\nSanity check: (generator, label) pairs from first {args.sanity_n} samples:")
        pairs = [(ds["train"][i]["generator"], ds["train"][i]["label"]) for i in range(args.sanity_n)]
        print(Counter(pairs))
    else:
        print("\nSanity check skipped (not applicable for this task).")

    # 4) Language distribution
    n = min(args.lang_sample, len(ds["train"]))
    langs = [ds["train"][i]["language"] for i in range(n)]
    print(f"\nLanguage distribution (first {n} train samples):")
    print(Counter(langs))

if __name__ == "__main__":
    main()
