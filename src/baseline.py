import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data/pairs/train.json"
VAL_PATH = PROJECT_ROOT / "data/pairs/val.json"
TEST_PATH = PROJECT_ROOT / "data/pairs/test.json"


def load_json(path: Path) -> List[Dict]:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_document_text(item: Dict) -> str:

    if "positive_doc" in item:
        return item["positive_doc"]

    if "positive" in item:
        return item["positive"]

    raise KeyError("No positive document field found. Expected 'positive_doc' or 'positive'.")


def build_corpus(data: List[Dict]) -> Tuple[List[str], List[Dict]]:

    documents = []
    metadata = []

    for idx, item in enumerate(data):
        doc_text = get_document_text(item).strip()

        if not doc_text:
            continue

        documents.append(doc_text)
        metadata.append(
            {
                "doc_id": idx,
                "query": item.get("query", ""),
                "text": doc_text,
                "source": item.get("source", "unknown"),
            }
        )

    return documents, metadata


class TfidfSearchEngine:

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
        )
        self.document_vectors = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []

    def fit(self, documents: List[str], metadata: List[Dict]) -> None:

        self.documents = documents
        self.metadata = metadata
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:

        if self.document_vectors is None:
            raise ValueError("Search engine is not fitted yet. Call fit() first.")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_vectors).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            item = self.metadata[idx].copy()
            item["rank"] = rank
            item["score"] = float(scores[idx])
            results.append(item)

        return results


def demo_search() -> None:

    from corpus import build_book_corpus

    documents, metadata = build_book_corpus()

    search_engine = TfidfSearchEngine()
    search_engine.fit(documents, metadata)

    while True:
        query = input("\nEnter query, or type 'exit': ").strip()

        if query.lower() == "exit":
            break

        results = search_engine.search(query, top_k=5)

        print("\nTop results:")

        for result in results:
            print(f"\nRank: {result['rank']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    demo_search()