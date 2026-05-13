import unittest

import torch

from models.losses.retrievaler_loss import RetrievalerChunkBuilder, RetrievalerLoss
from models.retrievaler import DustbinRetrievaler, RetrievalerConfig


class DustbinRetrievalerTest(unittest.TestCase):
    def test_forward_shape_and_padding_mask(self):
        model = DustbinRetrievaler(
            descriptor_dim=16,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
        )
        query = torch.randn(2, 16)
        candidates = torch.randn(2, 5, 16)
        candidate_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        logits = model(query, candidates, candidate_mask)
        self.assertEqual(logits.shape, (2, 6))
        self.assertTrue(torch.isneginf(logits[0, 4]))
        self.assertTrue(torch.isneginf(logits[0, 5]))

    def test_star_attention_mask(self):
        mask = DustbinRetrievaler.build_star_attention_mask(
            seq_len=6, device=torch.device('cpu')
        )
        self.assertFalse(mask[0].any())  # query attends all
        self.assertFalse(mask[1].any())  # dustbin attends all
        self.assertFalse(mask[2, 0])
        self.assertFalse(mask[2, 1])
        self.assertFalse(mask[2, 2])
        self.assertTrue(mask[2, 3])
        self.assertTrue(mask[3, 2])


class RetrievalerLossTest(unittest.TestCase):
    def _masks(self, batch_size):
        positives_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        positives_mask[0, 1] = True
        positives_mask[1, 0] = True
        positives_mask[2, 3] = True
        positives_mask[3, 2] = True
        negatives_mask = torch.logical_not(positives_mask)
        negatives_mask.fill_diagonal_(False)
        return positives_mask, negatives_mask

    def test_chunk_builder_creates_positive_and_empty_chunks(self):
        cfg = RetrievalerConfig(
            chunk_size=4,
            train_queries_per_batch=4,
            hard_empty_chunks_per_query=1,
            random_empty_chunks_per_query=1,
        )
        embeddings = torch.randn(8, 16)
        positives_mask, negatives_mask = self._masks(8)
        chunks = RetrievalerChunkBuilder(cfg).build(
            embeddings, positives_mask, negatives_mask
        )
        self.assertGreater(chunks.query_indices.numel(), 0)
        self.assertEqual(chunks.candidate_indices.shape[1], 4)
        self.assertTrue(chunks.is_positive_chunk.any())
        self.assertTrue(torch.logical_not(chunks.is_positive_chunk).any())
        self.assertTrue((chunks.targets[chunks.is_positive_chunk] > 0).all())
        self.assertTrue((chunks.targets[~chunks.is_positive_chunk] == 0).all())

    def test_retrievaler_loss_backpropagates_to_embeddings_and_model(self):
        torch.manual_seed(1)
        cfg = RetrievalerConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
            chunk_size=4,
            train_queries_per_batch=4,
            hard_empty_chunks_per_query=1,
            random_empty_chunks_per_query=1,
        )
        embeddings = torch.randn(8, 16, requires_grad=True)
        positives_mask, negatives_mask = self._masks(8)
        retrievaler = DustbinRetrievaler(
            descriptor_dim=16,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )
        loss, stats = RetrievalerLoss(cfg)(
            retrievaler, embeddings, positives_mask, negatives_mask
        )
        loss.backward()
        self.assertGreater(stats['num_chunks'], 0)
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        grad_norm = sum(
            p.grad.abs().sum().item()
            for p in retrievaler.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == '__main__':
    unittest.main()
