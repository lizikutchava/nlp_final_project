import argparse
from pathlib import Path
from typing import Dict, List

from baseline import (
    TfidfSearchEngine, build_corpus, get_document_text, load_json,
)
from search import NeuralSearchEngine, load_encoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data/pairs/train.json"
TEST_PATH = PROJECT_ROOT / "data/pairs/test.json"


def recall_at_k(ranks: List[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1 / rank for rank in ranks) / len(ranks)


def find_correct_doc_rank(results: List[Dict], correct_doc: str) -> int:
    correct_doc = correct_doc.strip()
    for result in results:
        if result["text"].strip() == correct_doc:
            return result["rank"]
    return 10 ** 9


def evaluate_engine(engine, test_data: List[Dict], top_k: int = 10) -> List[int]:
    ranks = []
    for item in test_data:
        results = engine.search(item["query"], top_k=top_k)
        ranks.append(find_correct_doc_rank(results, get_document_text(item)))
    return ranks


def summarize(ranks: List[int]) -> Dict[str, float]:
    return {
        "Recall@1": recall_at_k(ranks, 1),
        "Recall@5": recall_at_k(ranks, 5),
        "Recall@10": recall_at_k(ranks, 10),
        "MRR": mean_reciprocal_rank(ranks),
    }


def build_eval_corpus(test_data: List[Dict], train_data: List[Dict], n_distractors: int):
    corpus_data = list(test_data)
    if n_distractors:
        corpus_data += train_data[:n_distractors]
    return build_corpus(corpus_data)


def compare_models(
    checkpoint_path: str,
    top_k: int = 10,
    max_eval_samples: int = 500,
    n_distractors: int = 5000,
) -> Dict[str, Dict[str, float]]:
    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)[:max_eval_samples]

    documents, metadata = build_eval_corpus(test_data, train_data, n_distractors)

    tfidf = TfidfSearchEngine()
    tfidf.fit(documents, metadata)
    tfidf_metrics = summarize(evaluate_engine(tfidf, test_data, top_k))

    model = load_encoder(checkpoint_path)
    neural = NeuralSearchEngine(model)
    neural.fit(documents, metadata)
    neural_metrics = summarize(evaluate_engine(neural, test_data, top_k))

    header = f"{'Metric':<12}{'TF-IDF':>10}{'Neural':>10}"
    print("\nModel comparison")
    print(header)
    print("-" * len(header))
    for key in ["Recall@1", "Recall@5", "Recall@10", "MRR"]:
        print(f"{key:<12}{tfidf_metrics[key]:>10.4f}{neural_metrics[key]:>10.4f}")

    return {"tfidf": tfidf_metrics, "neural": neural_metrics}


def evaluate_tfidf_baseline(top_k: int = 10, max_eval_samples: int = 1000) -> Dict[str, float]:
    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)[:max_eval_samples]

    documents, metadata = build_corpus(train_data + test_data)

    engine = TfidfSearchEngine()
    engine.fit(documents, metadata)
    metrics = summarize(evaluate_engine(engine, test_data, top_k))

    print("\nTF-IDF Baseline Results")
    print("-----------------------")
    for key, value in metrics.items():
        print(f"{key:<10} {value:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained encoder checkpoint. If set, runs the comparison.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    args = parser.parse_args()

    if args.checkpoint:
        compare_models(args.checkpoint, top_k=args.top_k, max_eval_samples=args.max_eval_samples)
    else:
        evaluate_tfidf_baseline(top_k=args.top_k, max_eval_samples=args.max_eval_samples)
