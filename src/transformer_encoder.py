import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vocab import Vocab


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        q = self.query(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.key(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.value(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.out(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_len": max_len,
            "dropout": dropout,
        }

        self.token_embedding = nn.Embedding(len(vocab), d_model, padding_idx=vocab.pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(x * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        limit = self.max_len if max_length is None else min(max_length, self.max_len)
        encoded = [self.vocab.encode(text)[:limit] for text in texts]
        batch_len = max((len(ids) for ids in encoded), default=1) or 1

        input_ids = []
        attention_mask = []
        for ids in encoded:
            padding = batch_len - len(ids)
            input_ids.append(ids + [self.vocab.pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
