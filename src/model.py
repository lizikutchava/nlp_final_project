from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for neural retrieval.

    It takes input text, passes it through a pretrained transformer,
    then applies mean pooling over token embeddings to produce
    a single dense vector for the whole text.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling with attention mask.

        token_embeddings shape:
            [batch_size, seq_len, hidden_dim]

        attention_mask shape:
            [batch_size, seq_len]
        """

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        input_mask_expanded = input_mask_expanded.float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            sentence/document embeddings with shape [batch_size, hidden_dim]
        """

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state

        embeddings = self.mean_pooling(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def tokenize(
        self,
        texts,
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts.
        """

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )


def test_encoder() -> None:
    """
    Small test to verify that the encoder works.
    """

    model = TextEncoder()

    texts = [
        "What is a neural network?",
        "A neural network is a machine learning model inspired by the brain.",
    ]

    batch = model.tokenize(texts)

    with torch.no_grad():
        embeddings = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    print("Embeddings shape:", embeddings.shape)
    print("First embedding norm:", torch.norm(embeddings[0]).item())


if __name__ == "__main__":
    test_encoder()