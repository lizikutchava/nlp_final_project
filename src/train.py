import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from loss import InfoNCELoss, TripletLoss
from model import TextEncoder
from search import evaluate_retrieval, get_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data/pairs/train.json"
VAL_PATH = PROJECT_ROOT / "data/pairs/val.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


class TripletDataset(Dataset):

    def __init__(self, path: Path, max_samples: int = None) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.data = data[:max_samples] if max_samples else data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        item = self.data[idx]
        return item["query"], item["positive"], item["negative"]


def collate(batch) -> Tuple[List[str], List[str], List[str]]:
    queries, positives, negatives = zip(*batch)
    return list(queries), list(positives), list(negatives)


def train(
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    loss_type: str = "infonce",
    max_train_samples: int = None,
    max_val_samples: int = 1000,
    max_length: int = 256,
    model_name: str = "distilbert-base-uncased",
    log_every: int = 50,
) -> Dict:
    device = get_device()

    train_ds = TripletDataset(TRAIN_PATH, max_samples=max_train_samples)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    with open(VAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    if max_val_samples:
        val_data = val_data[:max_val_samples]

    model = TextEncoder(model_name=model_name).to(device)

    if loss_type == "infonce":
        criterion = InfoNCELoss(scale=20.0)
    elif loss_type == "triplet":
        criterion = TripletLoss(margin=0.2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = {"step": [], "train_loss": [], "val": []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for i, (queries, positives, negatives) in enumerate(train_loader, start=1):
            q = model.tokenize(queries, max_length=max_length)
            p = model.tokenize(positives, max_length=max_length)
            n = model.tokenize(negatives, max_length=max_length)

            q_emb = model(q["input_ids"].to(device), q["attention_mask"].to(device))
            p_emb = model(p["input_ids"].to(device), p["attention_mask"].to(device))
            n_emb = model(n["input_ids"].to(device), n["attention_mask"].to(device))

            loss = criterion(q_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running += loss.item()
            if i % log_every == 0:
                avg = running / log_every
                running = 0.0
                history["step"].append(global_step)
                history["train_loss"].append(avg)
                print(f"epoch {epoch} | step {i}/{len(train_loader)} | loss {avg:.4f}")

        val_metrics = evaluate_retrieval(model, val_data, k_values=(1, 5, 10), device=device)
        val_metrics["epoch"] = epoch
        history["val"].append(val_metrics)
        printable = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if k != "epoch")
        print(f"[val] epoch {epoch}: {printable}")

        ckpt = CHECKPOINT_DIR / f"encoder_{loss_type}_epoch{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            ckpt,
        )

    torch.save(
        {"model_state_dict": model.state_dict(), "model_name": model_name},
        CHECKPOINT_DIR / f"encoder_{loss_type}_final.pt",
    )
    with open(CHECKPOINT_DIR / f"history_{loss_type}.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss", dest="loss_type", default="infonce",
                        choices=["infonce", "triplet"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
