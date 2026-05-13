from dataclasses import dataclass
import configparser

import torch
import torch.nn as nn


@dataclass
class RetrievalerConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    chunk_size: int = 32
    train_queries_per_batch: int = 256
    hard_empty_chunks_per_query: int = 2
    random_empty_chunks_per_query: int = 1
    margin: float = 0.1
    lambda_db: float = 0.1
    lambda_rank: float = 0.1
    top_m: int = 2
    tau: float = 0.0
    stage2_max_candidates: int = 256


def load_retrievaler_config(config_path: str) -> RetrievalerConfig:
    config = configparser.ConfigParser()
    config.read(config_path)
    cfg = RetrievalerConfig()
    if 'RETRIEVALER' not in config:
        return cfg

    section = config['RETRIEVALER']
    cfg.hidden_dim = section.getint('hidden_dim', cfg.hidden_dim)
    cfg.num_layers = section.getint('num_layers', cfg.num_layers)
    cfg.num_heads = section.getint('num_heads', cfg.num_heads)
    cfg.ff_dim = section.getint('ff_dim', cfg.ff_dim)
    cfg.dropout = section.getfloat('dropout', cfg.dropout)
    cfg.chunk_size = section.getint('chunk_size', cfg.chunk_size)
    cfg.train_queries_per_batch = section.getint(
        'train_queries_per_batch', cfg.train_queries_per_batch
    )
    cfg.hard_empty_chunks_per_query = section.getint(
        'hard_empty_chunks_per_query', cfg.hard_empty_chunks_per_query
    )
    cfg.random_empty_chunks_per_query = section.getint(
        'random_empty_chunks_per_query', cfg.random_empty_chunks_per_query
    )
    cfg.margin = section.getfloat('margin', cfg.margin)
    cfg.lambda_db = section.getfloat('lambda_db', cfg.lambda_db)
    cfg.lambda_rank = section.getfloat('lambda_rank', cfg.lambda_rank)
    cfg.top_m = section.getint('top_m', cfg.top_m)
    cfg.tau = section.getfloat('tau', cfg.tau)
    cfg.stage2_max_candidates = section.getint(
        'stage2_max_candidates', cfg.stage2_max_candidates
    )
    return cfg


class DustbinRetrievaler(nn.Module):
    """
    Set-wise cross-encoder over one query and a chunk of global descriptors.

    Token order inside the transformer is [query, dustbin, candidate_1, ...].
    The returned logit order is [dustbin, candidate_1, ...].
    """

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(descriptor_dim, hidden_dim)
        self.candidate_proj = nn.Linear(descriptor_dim, hidden_dim)
        self.dustbin_token = nn.Parameter(torch.zeros(hidden_dim))
        self.type_embeddings = nn.Embedding(3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.dustbin_token, std=0.02)

    @staticmethod
    def build_star_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build a bool attention mask for [query, dustbin, candidates...].
        True entries are disallowed by nn.TransformerEncoder.
        """
        idx = torch.arange(seq_len, device=device)
        target = idx[:, None]
        source = idx[None, :]
        allowed = (target <= 1) | (source <= 1) | (target == source)
        return torch.logical_not(allowed)

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (B, C).
            candidates: Tensor of shape (B, K, C).
            candidate_mask: Optional bool tensor of shape (B, K), where True
                marks a valid candidate and False marks padding.

        Returns:
            Tensor of shape (B, K + 1), ordered as [dustbin, candidates...].
        """
        assert query.dim() == 2, f'Expected query shape (B,C), got {query.shape}'
        assert candidates.dim() == 3, (
            f'Expected candidates shape (B,K,C), got {candidates.shape}'
        )
        assert query.shape[0] == candidates.shape[0]
        assert query.shape[1] == candidates.shape[2]

        batch_size, chunk_size, _ = candidates.shape
        device = query.device

        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch_size, chunk_size, dtype=torch.bool, device=device
            )
        else:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
            assert candidate_mask.shape == (batch_size, chunk_size)

        query_token = self.query_proj(query).unsqueeze(1)
        dustbin_token = self.dustbin_token.view(1, 1, -1).expand(batch_size, 1, -1)
        candidate_tokens = self.candidate_proj(candidates)

        tokens = torch.cat([query_token, dustbin_token, candidate_tokens], dim=1)
        type_ids = torch.empty(
            tokens.shape[:2], dtype=torch.long, device=device
        )
        type_ids[:, 0] = 0
        type_ids[:, 1] = 1
        type_ids[:, 2:] = 2
        tokens = tokens + self.type_embeddings(type_ids)

        seq_len = tokens.shape[1]
        attn_mask = self.build_star_attention_mask(seq_len, device)
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 2, dtype=torch.bool, device=device),
                torch.logical_not(candidate_mask),
            ],
            dim=1,
        )

        encoded = self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )
        dustbin_logits = self.score_head(encoded[:, 1:2]).squeeze(-1)
        candidate_logits = self.score_head(encoded[:, 2:]).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(
            torch.logical_not(candidate_mask), float('-inf')
        )
        return torch.cat([dustbin_logits, candidate_logits], dim=1)


def build_retrievaler(descriptor_dim: int, cfg: RetrievalerConfig) -> DustbinRetrievaler:
    return DustbinRetrievaler(
        descriptor_dim=descriptor_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
