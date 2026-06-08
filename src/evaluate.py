import json
from pathlib import Path
from typing import Dict, List

from baseline import TfidfSearchEngine, build_corpus, get_document_text, load_json


TRAIN_PATH = Path("data/pairs/train.json")
TEST_PATH = Path("data/pairs/test.json")


def recall_at_k(ranks: List[int], k: int) -> float:
    """
    Computes Recall@K.

    If the correct document appears within top-k, it counts as 1.
    Otherwise, it counts as 0.
    """
    if not ranks:
        return 0.0

    hits = sum(1 for rank in ranks if rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: List[int]) -> float:
    """
    Computes Mean Reciprocal Rank.
    """
    if not ranks:
        return 0.0

    reciprocal_ranks = [1 / rank for rank in ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def find_correct_doc_rank(
    results: List[Dict],
    correct_doc: str,
) -> int:
    """
    Finds the rank of the correct document in retrieved results.

    Returns a large rank if the correct document is not found.
    """
    correct_doc = correct_doc.strip()

    for result in results:
        retrieved_text = result["text"].strip()

        if retrieved_text == correct_doc:
            return result["rank"]

    return 10**9


def evaluate_tfidf_baseline(top_k: int = 10, max_eval_samples: int = 1000) -> None:
    """
    Evaluates TF-IDF retrieval.

    For each test query, the correct document is the paired positive document.
    We search over a corpus built from train + test positive documents.
    """

    print("Loading data...")
    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)

    if max_eval_samples is not None:
        test_data = test_data[:max_eval_samples]

    print("Building retrieval corpus...")
    corpus_data = train_data + test_data
    documents, metadata = build_corpus(corpus_data)

    print(f"Corpus size: {len(documents)}")
    print(f"Evaluation queries: {len(test_data)}")

    print("Fitting TF-IDF search engine...")
    search_engine = TfidfSearchEngine()
    search_engine.fit(documents, metadata)

    ranks = []

    print("Evaluating...")
    for item in test_data:
        query = item["query"]
        correct_doc = get_document_text(item)

        results = search_engine.search(query, top_k=top_k)
        rank = find_correct_doc_rank(results, correct_doc)
        ranks.append(rank)

    r1 = recall_at_k(ranks, k=1)
    r5 = recall_at_k(ranks, k=5)
    r10 = recall_at_k(ranks, k=10)
    mrr = mean_reciprocal_rank(ranks)

    print("\nTF-IDF Baseline Results")
    print("-----------------------")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"MRR:       {mrr:.4f}")


if __name__ == "__main__":
    evaluate_tfidf_baseline(top_k=10, max_eval_samples=1000)