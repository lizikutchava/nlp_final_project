import re
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = PROJECT_ROOT / "data/book"

MIN_WORDS = 200
MAX_WORDS = 300
TARGET_WORDS = 250

ABBREVIATIONS = r"\b(e\.g|i\.e|et al|vs|cf|Fig|Eq|Dr|Mr|Mrs|Ms|St)\."

STAND_IN_TEXT = """
Natural language processing studies how computers can be made to understand and
generate human language. Early systems relied on hand-written rules and curated
lexicons, which were brittle and expensive to maintain. Modern systems instead
learn statistical patterns directly from large text corpora, which lets them
generalise to inputs that their designers never explicitly anticipated.

A central idea in modern NLP is the distributional hypothesis: words that occur in
similar contexts tend to have similar meanings. This motivates representing words
and longer texts as dense vectors, or embeddings, where geometric closeness in the
vector space reflects similarity in meaning. Embeddings can be learned by predicting
a word from its neighbours, or with contrastive objectives that pull related texts
together and push unrelated texts apart.

Information retrieval is the task of finding documents relevant to a user query.
Classical methods such as TF-IDF and BM25 score documents by exact term overlap,
weighting rare terms more heavily than common ones. They are fast and surprisingly
strong, but they fail when the query and the relevant document use different words
for the same concept, a problem known as the vocabulary mismatch.

Neural retrieval addresses vocabulary mismatch by encoding the query and each
document into the same embedding space and ranking documents by cosine similarity.
Because the encoder is trained to place paraphrases near one another, a neural
retriever can match a question to its answer even when they share few surface words.
In practice neural and lexical methods are complementary, and the strongest systems
often combine them.

Contrastive learning is the workhorse behind most modern text encoders. Given a
query, a relevant passage, and one or more irrelevant passages, the model is trained
so that the query embedding is closer to the positive than to the negatives. Triplet
loss and InfoNCE formalise this objective. InfoNCE treats retrieval as
classification over a batch of candidates, reusing the other positives in a batch as
in-batch negatives.

Evaluating a retrieval system requires a held-out set of queries with known relevant
documents. Recall at k measures whether the correct document appears among the top k
results, while mean reciprocal rank rewards placing it near the top. Reporting
several metrics together gives a fuller picture than any single number.

Transformer encoders read a whole passage at once and produce a contextual vector
for every token. To turn those token vectors into a single passage embedding we need
a pooling step. Mean pooling averages the token vectors, weighted by the attention
mask so that padding is ignored, and tends to work well for retrieval. Taking the
representation of a special classification token is another common choice, though it
usually needs more fine-tuning before it becomes useful for similarity search.

The quality of a contrastive model depends heavily on the negatives it sees during
training. Random negatives are easy to mine but often trivially different from the
query, so the model learns little from them. Hard negatives, passages that look
relevant but are not, force the encoder to make finer distinctions and usually lead
to large gains. Mining good hard negatives, for example with an earlier version of
the model itself, is one of the most effective tricks in modern dense retrieval.

Once an encoder is trained, every document in the collection is embedded once and
stored in an index. At query time only the query has to be encoded, and the index
returns the nearest document vectors. For small collections an exact search by brute
force is perfectly fine, but for millions of documents approximate nearest neighbour
structures trade a little accuracy for a large speed-up, which is what makes neural
search practical at scale.
""".strip()


def normalize_whitespace(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    cleaned = []
    for paragraph in paragraphs:
        collapsed = re.sub(r"\s+", " ", paragraph).strip()
        if collapsed:
            cleaned.append(collapsed)
    return "\n\n".join(cleaned)


def split_sentences(text: str) -> List[str]:
    protected = re.sub(ABBREVIATIONS, lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
    target_words: int = TARGET_WORDS,
) -> List[str]:
    text = normalize_whitespace(text)
    sentences: List[str] = []
    for paragraph in text.split("\n\n"):
        sentences.extend(split_sentences(paragraph))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current_len + n > max_words and current_len >= min_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0

        current.append(sentence)
        current_len += n

        if current_len >= target_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0

    if current:
        if chunks and current_len < min_words:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))

    return chunks


def load_book_texts(book_dir: Path = BOOK_DIR) -> List[Tuple[str, str]]:

    if book_dir.exists():
        texts = []
        for path in sorted(book_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                texts.append((path.stem, content))
        if texts:
            return texts

    return [("stand_in", STAND_IN_TEXT)]


def build_book_corpus(
    book_dir: Path = BOOK_DIR,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
    target_words: int = TARGET_WORDS,
) -> Tuple[List[str], List[Dict]]:
    documents: List[str] = []
    metadata: List[Dict] = []

    doc_id = 0
    for source_name, text in load_book_texts(book_dir):
        for chunk_id, chunk in enumerate(chunk_text(text, min_words, max_words, target_words)):
            documents.append(chunk)
            metadata.append(
                {
                    "doc_id": doc_id,
                    "text": chunk,
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "n_words": len(chunk.split()),
                }
            )
            doc_id += 1

    return documents, metadata


def corpus_stats(metadata: List[Dict]) -> Dict[str, float]:
    if not metadata:
        return {"n_chunks": 0}

    lengths = [item["n_words"] for item in metadata]
    sources = {item["source"] for item in metadata}
    return {
        "n_chunks": len(metadata),
        "n_sources": len(sources),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "mean_words": sum(lengths) / len(lengths),
        "within_target": sum(1 for n in lengths if MIN_WORDS <= n <= MAX_WORDS) / len(lengths),
    }


def main() -> None:
    documents, metadata = build_book_corpus()
    stats = corpus_stats(metadata)

    print("\nBook corpus built")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:<14} {value:.3f}")
        else:
            print(f"{key:<14} {value}")

    if documents:
        print("\nExample chunk (first):")
        print(documents[0][:600], "...")


if __name__ == "__main__":
    main()
