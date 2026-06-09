import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        if negative_emb is not None:
            candidates = torch.cat([positive_emb, negative_emb], dim=0)
        else:
            candidates = positive_emb

        scores = torch.matmul(query_emb, candidates.t()) * self.scale
        labels = torch.arange(query_emb.size(0), device=query_emb.device)

        return self.cross_entropy(scores, labels)


class TripletLoss(nn.Module):

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = torch.sum(query_emb * positive_emb, dim=1)
        neg_sim = torch.sum(query_emb * negative_emb, dim=1)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return loss.mean()
