import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corpus import build_book_corpus


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def build_vectors(docs):
    doc_tf = [Counter(tokens(d)) for d in docs]
    df = Counter()
    for tf in doc_tf:
        for word in tf:
            df[word] += 1
    n = len(docs)
    idf = {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}

    def vectorize(tf):
        vec = {word: freq * idf.get(word, 0.0) for word, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {word: v / norm for word, v in vec.items()}

    return [vectorize(tf) for tf in doc_tf], vectorize


def main():
    docs, meta = build_book_corpus()
    print(f"corpus: {len(docs)} chunks; metadata keys: {sorted(meta[0])}")
    assert all("text" in m for m in meta), "every metadata item needs a text field"

    doc_vecs, vectorize = build_vectors(docs)

    def search(query, top_k=3):
        qv = vectorize(Counter(tokens(query)))
        scored = sorted(
            ((sum(qv[w] * dv.get(w, 0.0) for w in qv), i) for i, dv in enumerate(doc_vecs)),
            reverse=True,
        )
        results = []
        for rank, (score, idx) in enumerate(scored[:top_k], 1):
            item = dict(meta[idx])
            item["rank"] = rank
            item["score"] = float(score)
            results.append(item)
        return results

    queries = [
        "how does contrastive learning use negative examples?",
        "what metric checks if the correct document is in the top results?",
    ]
    for query in queries:
        print(f"\nQUERY: {query}")
        for r in search(query):
            assert {"rank", "score", "text"} <= set(r), "result missing demo fields"
            preview = r["text"][:90]
            print(f"  rank {r['rank']} | score {r['score']:.3f} | src={r['source']} | {preview}...")

    print("\nOK: build_book_corpus -> fit -> search -> {rank, score, text} flow verified.")


if __name__ == "__main__":
    main()
