import re
from collections import Counter
from typing import Dict, List

PAD = "<pad>"
UNK = "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_text(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_id = stoi[PAD]
        self.unk_id = stoi[UNK]

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, texts: List[str], max_size: int = 30000, min_freq: int = 2) -> "Vocab":
        counter = Counter()
        for text in texts:
            counter.update(tokenize_text(text))

        stoi = {PAD: 0, UNK: 1}
        for word, freq in counter.most_common():
            if freq < min_freq or len(stoi) >= max_size:
                break
            stoi[word] = len(stoi)
        return cls(stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokenize_text(text)]

    def to_dict(self) -> Dict[str, int]:
        return self.stoi

    @classmethod
    def from_dict(cls, stoi: Dict[str, int]) -> "Vocab":
        return cls(stoi)
