from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.retrievaler import RetrievalerConfig


@dataclass
class RetrievalerChunks:
    query_indices: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    targets: torch.Tensor
    positive_positions: torch.Tensor
    is_positive_chunk: torch.Tensor
    chunk_kinds: torch.Tensor


class RetrievalerChunkBuilder:
    POSITIVE = 1
    HARD_EMPTY = 2
    RANDOM_EMPTY = 3

    def __init__(self, cfg: RetrievalerConfig):
        self.cfg = cfg

    def build(
        self,
        embeddings: torch.Tensor,
        positives_mask: torch.Tensor,
        negatives_mask: torch.Tensor,
    ) -> RetrievalerChunks:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        chunk_size = self.cfg.chunk_size

        positives_mask = positives_mask.to(device=device, dtype=torch.bool).clone()
        negatives_mask = negatives_mask.to(device=device, dtype=torch.bool).clone()
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        positives_mask.masked_fill_(eye, False)
        negatives_mask.masked_fill_(eye, False)

        valid_queries = torch.where(
            (positives_mask.sum(dim=1) > 0) & (negatives_mask.sum(dim=1) > 0)
        )[0]
        if valid_queries.numel() == 0:
            return self._empty(device)

        if valid_queries.numel() > self.cfg.train_queries_per_batch:
            perm = torch.randperm(valid_queries.numel(), device=device)
            valid_queries = valid_queries[perm[:self.cfg.train_queries_per_batch]]

        with torch.no_grad():
            norm_embeddings = F.normalize(embeddings.detach(), dim=1)
            sim = norm_embeddings @ norm_embeddings.t()

        query_indices = []
        candidate_indices = []
        candidate_masks = []
        targets = []
        positive_positions = []
        is_positive = []
        chunk_kinds = []

        for query_idx in valid_queries.tolist():
            pos_idx = torch.where(positives_mask[query_idx])[0]
            neg_idx = torch.where(negatives_mask[query_idx])[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            neg_scores = sim[query_idx, neg_idx]
            hard_order = torch.argsort(neg_scores, descending=True)
            hard_neg_idx = neg_idx[hard_order]

            pos_choice = pos_idx[
                torch.randint(pos_idx.numel(), (1,), device=device).item()
            ]
            pos_candidates = torch.cat(
                [pos_choice.view(1), hard_neg_idx[: max(chunk_size - 1, 0)]],
                dim=0,
            )
            self._append_chunk(
                query_idx=query_idx,
                indices=pos_candidates,
                positive_index=pos_choice.item(),
                kind=self.POSITIVE,
                device=device,
                query_indices=query_indices,
                candidate_indices=candidate_indices,
                candidate_masks=candidate_masks,
                targets=targets,
                positive_positions=positive_positions,
                is_positive=is_positive,
                chunk_kinds=chunk_kinds,
            )

            for n in range(self.cfg.hard_empty_chunks_per_query):
                start = n * chunk_size
                hard_candidates = hard_neg_idx[start:start + chunk_size]
                if hard_candidates.numel() == 0:
                    hard_candidates = hard_neg_idx[:chunk_size]
                self._append_chunk(
                    query_idx=query_idx,
                    indices=hard_candidates,
                    positive_index=None,
                    kind=self.HARD_EMPTY,
                    device=device,
                    query_indices=query_indices,
                    candidate_indices=candidate_indices,
                    candidate_masks=candidate_masks,
                    targets=targets,
                    positive_positions=positive_positions,
                    is_positive=is_positive,
                    chunk_kinds=chunk_kinds,
                )

            for _ in range(self.cfg.random_empty_chunks_per_query):
                perm = torch.randperm(neg_idx.numel(), device=device)
                random_candidates = neg_idx[perm[:chunk_size]]
                self._append_chunk(
                    query_idx=query_idx,
                    indices=random_candidates,
                    positive_index=None,
                    kind=self.RANDOM_EMPTY,
                    device=device,
                    query_indices=query_indices,
                    candidate_indices=candidate_indices,
                    candidate_masks=candidate_masks,
                    targets=targets,
                    positive_positions=positive_positions,
                    is_positive=is_positive,
                    chunk_kinds=chunk_kinds,
                )

        if len(query_indices) == 0:
            return self._empty(device)

        return RetrievalerChunks(
            query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
            candidate_indices=torch.stack(candidate_indices, dim=0),
            candidate_mask=torch.stack(candidate_masks, dim=0),
            targets=torch.tensor(targets, dtype=torch.long, device=device),
            positive_positions=torch.tensor(
                positive_positions, dtype=torch.long, device=device
            ),
            is_positive_chunk=torch.tensor(is_positive, dtype=torch.bool, device=device),
            chunk_kinds=torch.tensor(chunk_kinds, dtype=torch.long, device=device),
        )

    def _append_chunk(
        self,
        query_idx,
        indices,
        positive_index,
        kind,
        device,
        query_indices,
        candidate_indices,
        candidate_masks,
        targets,
        positive_positions,
        is_positive,
        chunk_kinds,
    ):
        chunk_size = self.cfg.chunk_size
        indices = indices.to(device=device, dtype=torch.long)
        valid_count = min(indices.numel(), chunk_size)
        indices = indices[:valid_count]
        mask = torch.zeros(chunk_size, dtype=torch.bool, device=device)
        padded = torch.zeros(chunk_size, dtype=torch.long, device=device)
        if valid_count > 0:
            padded[:valid_count] = indices
            mask[:valid_count] = True

        if positive_index is not None:
            perm = torch.randperm(valid_count, device=device)
            shuffled_valid = padded[:valid_count][perm]
            padded[:valid_count] = shuffled_valid
            positive_matches = torch.where(shuffled_valid == positive_index)[0]
            assert positive_matches.numel() == 1
            positive_pos = positive_matches.item()
            target = positive_pos + 1
            positive_flag = True
        else:
            if valid_count > 1:
                perm = torch.randperm(valid_count, device=device)
                padded[:valid_count] = padded[:valid_count][perm]
            positive_pos = -1
            target = 0
            positive_flag = False

        query_indices.append(query_idx)
        candidate_indices.append(padded)
        candidate_masks.append(mask)
        targets.append(target)
        positive_positions.append(positive_pos)
        is_positive.append(positive_flag)
        chunk_kinds.append(kind)

    def _empty(self, device) -> RetrievalerChunks:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return RetrievalerChunks(
            query_indices=empty_long,
            candidate_indices=torch.empty(
                0, self.cfg.chunk_size, dtype=torch.long, device=device
            ),
            candidate_mask=torch.empty(
                0, self.cfg.chunk_size, dtype=torch.bool, device=device
            ),
            targets=empty_long,
            positive_positions=empty_long,
            is_positive_chunk=empty_bool,
            chunk_kinds=empty_long,
        )


class RetrievalerLoss(nn.Module):
    def __init__(self, cfg: RetrievalerConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_builder = RetrievalerChunkBuilder(cfg)

    def forward(
        self,
        retrievaler: nn.Module,
        embeddings: torch.Tensor,
        positives_mask: torch.Tensor,
        negatives_mask: torch.Tensor,
    ):
        chunks = self.chunk_builder.build(embeddings, positives_mask, negatives_mask)
        if chunks.query_indices.numel() == 0:
            zero = embeddings.sum() * 0.0
            return zero, self._zero_stats()

        query = embeddings[chunks.query_indices]
        candidates = embeddings[chunks.candidate_indices]
        logits = retrievaler(query, candidates, chunks.candidate_mask)

        ce_loss = F.cross_entropy(logits, chunks.targets)
        candidate_logits = logits[:, 1:]
        dustbin_logits = logits[:, 0]
        deltas = candidate_logits - dustbin_logits.unsqueeze(1)

        positive_mask = chunks.is_positive_chunk
        empty_mask = torch.logical_not(positive_mask)

        db_loss = embeddings.sum() * 0.0
        rank_loss = embeddings.sum() * 0.0

        if positive_mask.any():
            pos_rows = torch.where(positive_mask)[0]
            pos_cols = chunks.positive_positions[pos_rows]
            pos_delta = deltas[pos_rows, pos_cols]
            db_loss = db_loss + F.relu(self.cfg.margin - pos_delta).mean()

            neg_mask = chunks.candidate_mask[pos_rows].clone()
            neg_mask[torch.arange(pos_rows.numel(), device=embeddings.device), pos_cols] = False
            neg_logits = candidate_logits[pos_rows].masked_fill(
                torch.logical_not(neg_mask), float('-inf')
            )
            max_neg = neg_logits.max(dim=1).values
            pos_logits = candidate_logits[pos_rows, pos_cols]
            valid_rank = torch.isfinite(max_neg)
            if valid_rank.any():
                rank_loss = F.relu(
                    self.cfg.margin + max_neg[valid_rank] - pos_logits[valid_rank]
                ).mean()

        if empty_mask.any():
            empty_rows = torch.where(empty_mask)[0]
            empty_delta = deltas[empty_rows].masked_fill(
                torch.logical_not(chunks.candidate_mask[empty_rows]), float('-inf')
            )
            max_empty_delta = empty_delta.max(dim=1).values
            valid_empty = torch.isfinite(max_empty_delta)
            if valid_empty.any():
                db_loss = db_loss + F.relu(
                    self.cfg.margin + max_empty_delta[valid_empty]
                ).mean()

        loss = (
            ce_loss
            + self.cfg.lambda_db * db_loss
            + self.cfg.lambda_rank * rank_loss
        )
        stats = self._stats(
            loss=loss,
            ce_loss=ce_loss,
            db_loss=db_loss,
            rank_loss=rank_loss,
            logits=logits,
            deltas=deltas,
            chunks=chunks,
        )
        return loss, stats

    def _stats(
        self,
        loss,
        ce_loss,
        db_loss,
        rank_loss,
        logits,
        deltas,
        chunks: RetrievalerChunks,
    ) -> Dict[str, float]:
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            positive_mask = chunks.is_positive_chunk
            empty_mask = torch.logical_not(positive_mask)

            positive_acc = torch.tensor(0.0, device=logits.device)
            empty_acc = torch.tensor(0.0, device=logits.device)
            avg_delta_pos = torch.tensor(0.0, device=logits.device)
            avg_delta_empty = torch.tensor(0.0, device=logits.device)

            if positive_mask.any():
                pos_rows = torch.where(positive_mask)[0]
                positive_acc = (
                    predictions[pos_rows] == chunks.targets[pos_rows]
                ).float().mean()
                pos_cols = chunks.positive_positions[pos_rows]
                avg_delta_pos = deltas[pos_rows, pos_cols].mean()

            if empty_mask.any():
                empty_rows = torch.where(empty_mask)[0]
                empty_acc = (predictions[empty_rows] == 0).float().mean()
                empty_delta = deltas[empty_rows].masked_fill(
                    torch.logical_not(chunks.candidate_mask[empty_rows]),
                    float('-inf'),
                )
                max_empty_delta = empty_delta.max(dim=1).values
                valid_empty = torch.isfinite(max_empty_delta)
                if valid_empty.any():
                    avg_delta_empty = max_empty_delta[valid_empty].mean()

            return {
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'db_margin_loss': db_loss.item(),
                'rank_loss': rank_loss.item(),
                'positive_chunk_acc': positive_acc.item(),
                'empty_chunk_acc': empty_acc.item(),
                'avg_delta_pos': avg_delta_pos.item(),
                'avg_delta_empty_max': avg_delta_empty.item(),
                'num_chunks': float(chunks.targets.numel()),
                'num_positive_chunks': float(positive_mask.sum().item()),
                'num_empty_chunks': float(empty_mask.sum().item()),
            }

    def _zero_stats(self):
        return {
            'loss': 0.0,
            'ce_loss': 0.0,
            'db_margin_loss': 0.0,
            'rank_loss': 0.0,
            'positive_chunk_acc': 0.0,
            'empty_chunk_acc': 0.0,
            'avg_delta_pos': 0.0,
            'avg_delta_empty_max': 0.0,
            'num_chunks': 0.0,
            'num_positive_chunks': 0.0,
            'num_empty_chunks': 0.0,
        }
