from typing import Dict, List, Optional

import torch

from model import TextEncoder


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_texts(
    model: TextEncoder,
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or get_device()
    model.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = model.tokenize(batch, max_length=max_length)
        emb = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def evaluate_retrieval(
    model: TextEncoder,
    data: List[Dict],
    k_values=(1, 5, 10),
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or get_device()
    queries = [d["query"] for d in data]
    documents = [d["positive"] for d in data]

    query_emb = encode_texts(model, queries, batch_size, max_length, device)
    doc_emb = encode_texts(model, documents, batch_size, max_length, device)

    scores = torch.matmul(query_emb, doc_emb.t())
    correct = scores.diag().unsqueeze(1)
    ranks = (scores > correct).sum(dim=1) + 1
    ranks = ranks.float()

    metrics = {f"recall@{k}": (ranks <= k).float().mean().item() for k in k_values}
    metrics["mrr"] = (1.0 / ranks).mean().item()
    return metrics


def load_encoder(
    checkpoint_path: Optional[str] = None,
    model_name: str = "distilbert-base-uncased",
    device: Optional[torch.device] = None,
) -> TextEncoder:
    device = device or get_device()
    model = TextEncoder(model_name=model_name)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


class NeuralSearchEngine:

    def __init__(
        self,
        model: TextEncoder,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        self.model = model
        self.device = device or get_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.to(self.device)

        self.document_embeddings: Optional[torch.Tensor] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []

    def fit(self, documents: List[str], metadata: List[Dict]) -> None:
        self.documents = documents
        self.metadata = metadata
        self.document_embeddings = encode_texts(
            self.model, documents,
            batch_size=self.batch_size, max_length=self.max_length, device=self.device,
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.document_embeddings is None:
            raise ValueError("Search engine is not fitted yet. Call fit() first.")

        query_emb = encode_texts(
            self.model, [query],
            batch_size=1, max_length=self.max_length, device=self.device,
        )
        scores = torch.matmul(query_emb, self.document_embeddings.t()).flatten()

        k = min(top_k, scores.size(0))
        top_scores, top_indices = torch.topk(scores, k)

        results = []
        for rank, (idx, score) in enumerate(
            zip(top_indices.tolist(), top_scores.tolist()), start=1
        ):
            item = self.metadata[idx].copy()
            item["rank"] = rank
            item["score"] = float(score)
            results.append(item)
        return results

    def save_index(self, path: str) -> None:
        torch.save(
            {
                "embeddings": self.document_embeddings,
                "metadata": self.metadata,
                "documents": self.documents,
            },
            path,
        )

    def load_index(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.document_embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.documents = data["documents"]


def demo_search(checkpoint_path: Optional[str] = None) -> None:
    from baseline import build_corpus, load_json, TRAIN_PATH

    train_data = load_json(TRAIN_PATH)
    documents, metadata = build_corpus(train_data)

    model = load_encoder(checkpoint_path)
    engine = NeuralSearchEngine(model)
    engine.fit(documents, metadata)

    while True:
        query = input("\nEnter query, or type 'exit': ").strip()
        if query.lower() == "exit":
            break
        for result in engine.search(query, top_k=5):
            print(f"\nRank: {result['rank']}  Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    demo_search()
