import json
import os
import random
from datasets import load_dataset


def download_and_save_msmarco(output_dir="data", n_samples=50000, seed=42):
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/processed", exist_ok=True)
    os.makedirs(f"{output_dir}/pairs", exist_ok=True)

    print("Downloading MS MARCO...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")  # ← fixed

    print("Processing pairs...")
    pairs = []

    for item in ds:
        query = item["query"].strip()
        passages = item["passages"]

        positive = None
        negatives = []

        for i, passage in enumerate(passages["passage_text"]):
            if passages["is_selected"][i] == 1:
                positive = passage.strip()
            else:
                negatives.append(passage.strip())

        if positive is None:
            continue
        if len(negatives) == 0:
            continue

        pairs.append({
            "query": query,
            "positive": positive,
            "negative": random.choice(negatives)  # random negative for now
        })

        if len(pairs) >= n_samples:
            break

    # save raw
    with open(f"{output_dir}/raw/msmarco_raw.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} raw pairs")

    # shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    with open(f"{output_dir}/pairs/train.json", "w") as f:
        json.dump(train, f, indent=2)

    with open(f"{output_dir}/pairs/val.json", "w") as f:
        json.dump(val, f, indent=2)

    with open(f"{output_dir}/pairs/test.json", "w") as f:
        json.dump(test, f, indent=2)

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    print("Done. Files saved to data/pairs/")


if __name__ == "__main__":
    download_and_save_msmarco()