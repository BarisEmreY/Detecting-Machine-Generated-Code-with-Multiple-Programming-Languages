"""
Chunked (sliding-window) evaluation for PHP on Task C validation.

Why:
- PHP samples are much longer than other languages.
- With max_length=256, truncation may remove useful signal.
- Chunked inference keeps max_length fixed (memory-safe on MPS) and aggregates logits.
"""

import argparse
from collections import Counter
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


@torch.no_grad()
def predict_chunked(model, tokenizer, text, max_length=256, stride=128, device="cpu"):
    """
    Sliding-window inference using token IDs directly (no decode->encode).
    This avoids exceeding model max_position_embeddings (512).
    """
    # Get full token ids (no special tokens)
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]

    # Window length excluding special tokens
    window = max_length - 2
    if window <= 0:
        raise ValueError("max_length must be >= 3")

    # Short case
    if len(ids) <= window:
        input_ids = tokenizer.build_inputs_with_special_tokens(ids)
        attention_mask = [1] * len(input_ids)

        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attention_mask], device=device)

        out = model(input_ids=input_ids_t, attention_mask=attn_t)
        return out.logits.squeeze(0).detach().cpu().numpy()

    logits_list = []
    start = 0
    while start < len(ids):
        end = min(start + window, len(ids))
        chunk_ids = ids[start:end]

        input_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        attention_mask = [1] * len(input_ids)

        # 2D tensors: [batch, seq_len]
        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attention_mask], device=device)

        out = model(input_ids=input_ids_t, attention_mask=attn_t)
        logits_list.append(out.logits.squeeze(0).detach().cpu().numpy())

        if end == len(ids):
            break
        start += stride

    return np.mean(np.stack(logits_list, axis=0), axis=0)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="runs/graphcodebert_taskC")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--limit", type=int, default=2000, help="Max PHP samples to evaluate (speed control)")
    args = p.parse_args()

    ds = load_dataset("DaniilOr/SemEval-2026-Task13", "C")
    val = ds["validation"]

    php_idx = [i for i, lang in enumerate(val["language"]) if lang == "PHP"]
    php = val.select(php_idx)

    if args.limit > 0:
        php = php.select(range(min(args.limit, len(php))))

    print(f"PHP validation eval size: {len(php)}")
    print("PHP label distribution:", Counter(php["label"]))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    y_true = php["label"]
    y_pred = []

    for i in range(len(php)):
        text = php[i]["code"]
        logits = predict_chunked(
            model, tokenizer, text,
            max_length=args.max_length,
            stride=args.stride,
            device=device
        )
        y_pred.append(int(np.argmax(logits)))

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(php)}")

    score = macro_f1(y_true, y_pred)
    print(f"\nTask C PHP Macro F1 (chunked eval): {score:.4f}")


if __name__ == "__main__":
    main()
