"""ProstT5 encoder embedding extraction and query pooling utilities."""

from __future__ import annotations

import logging
import math
import re
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency in CI
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:  # pragma: no cover
    T5EncoderModel = None
    T5Tokenizer = None

DEFAULT_3DI_TOKENS = list("ABCDEFGHIJKLMNOPQRST")
_SPIECE_PREFIX = "\u2581"
_FALLBACK_3DI_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

log = logging.getLogger(__name__)
_NONCANONICAL_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_sequence(sequence: str) -> str:
    """Replace non-canonical amino acids with X."""
    return _NONCANONICAL_RE.sub("X", sequence.strip().upper())


def slice_logits_to_3di(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Slice logits to the 20 3Di token columns."""
    if logits.ndim != 3:
        raise ValueError("logits must be (batch, length, vocab).")
    if len(token_ids) != 20:
        raise ValueError("token_ids must have length 20.")
    index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    return logits.index_select(dim=-1, index=index)


def compute_entropy_confidence(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute confidence = 1 - entropy/log(20) from probabilities."""
    if probs.ndim != 3 or probs.size(-1) != 20:
        raise ValueError("probs must be (batch, length, 20).")
    probs_fp32 = probs.float().clamp(min=eps)
    entropy = -torch.sum(probs_fp32 * torch.log(probs_fp32), dim=-1)
    confidence = 1.0 - entropy / math.log(20.0)
    return confidence


class ProstT53DiEmbedder(nn.Module):
    """Generate per-residue encoder embeddings using ProstT5.

    Args:
        model_name: HuggingFace model ID for ProstT5.
        device: Torch device string or torch.device.
        cache_dir: Optional cache directory for model weights.
        prefix_token: Unused for encoder embeddings (kept for config compatibility).
        three_di_tokens: Optional list of 20 tokens for the 3Di alphabet.
        three_di_token_ids: Optional list of 20 token IDs (overrides tokens).
        skip_second_fwd: Unused for encoder embeddings.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/ProstT5",
        device: str | torch.device = "cpu",
        cache_dir: Optional[str] = None,
        prefix_token: str = "<AA2fold>",
        three_di_tokens: Optional[Sequence[str]] = None,
        three_di_token_ids: Optional[Sequence[int]] = None,
        skip_second_fwd: bool = False,
    ) -> None:
        super().__init__()
        if T5EncoderModel is None or T5Tokenizer is None:
            raise ImportError("transformers with T5EncoderModel is required.")
        self.device = torch.device(device)
        tokenizer_kwargs = {}
        model_kwargs = {"use_safetensors": True}
        if cache_dir is not None:
            model_kwargs["cache_dir"] = str(cache_dir)
            tokenizer_kwargs["cache_dir"] = str(cache_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, do_lower_case=False, legacy=True, **tokenizer_kwargs
        )
        self.model = T5EncoderModel.from_pretrained(model_name, **model_kwargs)
        if self.device.type == "cpu":
            self.model.float()
        else:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.prefix_token = prefix_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.three_di_token_ids = self._resolve_token_ids(three_di_tokens, three_di_token_ids)
        self.skip_second_fwd = bool(skip_second_fwd)

    def _resolve_token_ids(
        self,
        three_di_tokens: Optional[Sequence[str]],
        three_di_token_ids: Optional[Sequence[int]],
    ) -> list[int]:
        def _token_to_id(token: str) -> tuple[Optional[int], str]:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0 or token_id == self.tokenizer.unk_token_id:
                if len(token) == 1 and token.isalpha() and token.isupper():
                    prefixed = f"{_SPIECE_PREFIX}{token}"
                    token_id = self.tokenizer.convert_tokens_to_ids(prefixed)
                    return token_id, prefixed
            return token_id, token

        def _validate_ids(ids: list[int]) -> None:
            if len(set(ids)) != 20:
                raise ValueError("3Di token IDs must be unique.")
            if self.tokenizer.unk_token_id is not None and any(
                val == self.tokenizer.unk_token_id for val in ids
            ):
                raise ValueError("3Di token IDs must not map to <unk>.")

        if three_di_token_ids is not None:
            ids = [int(val) for val in three_di_token_ids]
            if len(ids) != 20:
                raise ValueError("three_di_token_ids must have length 20.")
            _validate_ids(ids)
            return ids
        tokens = list(three_di_tokens) if three_di_tokens is not None else DEFAULT_3DI_TOKENS
        if len(tokens) != 20:
            raise ValueError("three_di_tokens must have length 20.")
        resolved_tokens: list[str] = []
        ids: list[int] = []
        for token in tokens:
            token_id, resolved = _token_to_id(token)
            if token_id is None or token_id < 0 or token_id == self.tokenizer.unk_token_id:
                continue
            if token_id in ids:
                continue
            resolved_tokens.append(resolved)
            ids.append(token_id)
        if len(ids) < 20:
            for token in _FALLBACK_3DI_POOL:
                token_id, resolved = _token_to_id(token)
                if token_id is None or token_id < 0 or token_id == self.tokenizer.unk_token_id:
                    continue
                if token_id in ids:
                    continue
                resolved_tokens.append(resolved)
                ids.append(token_id)
                if len(ids) == 20:
                    break
        if len(ids) != 20:
            raise ValueError(
                "3Di token IDs could not be resolved; provide three_di_token_ids."
            )
        if resolved_tokens != tokens:
            log.warning(
                "Resolved 3Di tokens to %s based on tokenizer availability.",
                resolved_tokens,
            )
        _validate_ids(ids)
        return ids

    @torch.inference_mode()
    def forward(
        self,
        sequences: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-residue encoder embeddings.

        Args:
            sequences: Sequence of amino-acid strings.
            lengths: Optional sequence lengths for truncation.

        Returns:
            Tuple of embeddings with shape (batch, max_length, hidden_dim) and
            residue lengths as a tensor.
        """
        if not sequences:
            raise ValueError("sequences must be non-empty.")
        cleaned = [clean_sequence(seq) for seq in sequences]
        if lengths is not None:
            trimmed = []
            for seq, length in zip(cleaned, lengths):
                trimmed.append(seq[: int(length)])
            cleaned = trimmed
        lengths_tensor = torch.tensor([len(seq) for seq in cleaned], dtype=torch.long)
        max_len = int(lengths_tensor.max().item()) if lengths_tensor.numel() > 0 else 0

        # Tokenize with spaced residues; add_special_tokens appends a single EOS token.
        spaced = [" ".join(list(seq)) for seq in cleaned]
        ids = self.tokenizer.batch_encode_plus(
            spaced,
            add_special_tokens=True,
            padding="longest",
            return_attention_mask=True,
        )
        input_ids = torch.tensor(ids["input_ids"], device=self.device)
        attention_mask = torch.tensor(ids["attention_mask"], device=self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if max_len <= 0:
            empty = hidden.new_zeros((len(cleaned), 0, hidden.size(-1)))
            return empty.detach().cpu(), lengths_tensor

        embeddings = hidden.new_zeros((len(cleaned), max_len, hidden.size(-1)))
        for idx, length in enumerate(lengths_tensor.tolist()):
            if length <= 0:
                continue
            embeddings[idx, :length] = hidden[idx, :length]
        return embeddings.detach().cpu(), lengths_tensor


class ProstT5QueryEncoder(nn.Module):
    """Compute per-head ProstT5 query vectors from encoder embeddings.

    Args:
        heads: Number of attention heads.
        head_dim: Per-head dimension (d).
        input_dim: ProstT5 encoder hidden size.
    """

    def __init__(self, heads: int, head_dim: int, input_dim: int = 1024) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.input_dim = int(input_dim)
        self.projections = nn.ModuleList(
            [nn.Linear(self.input_dim, head_dim, bias=False) for _ in range(heads)]
        )
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
        self.norms = nn.ModuleList([nn.LayerNorm(head_dim) for _ in range(heads)])
        self.pool_weights = nn.Parameter(torch.zeros(heads, head_dim))
        nn.init.normal_(self.pool_weights, mean=0.0, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention-pooled query vectors.

        Args:
            embeddings: Tensor with shape (batch, length, input_dim).
            mask: Optional boolean tensor with shape (batch, length).
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of queries with shape (batch, heads, head_dim) and optional
            weights with shape (batch, heads, length).
        """
        if embeddings.ndim != 3 or embeddings.size(-1) != self.input_dim:
            raise ValueError(
                "embeddings must be (batch, length, input_dim)."
            )
        if mask is not None and mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must match (batch, length).")

        queries: list[torch.Tensor] = []
        attn_weights: list[torch.Tensor] = []
        embed_fp32 = embeddings.float()
        for idx in range(self.heads):
            projected = self.projections[idx](embed_fp32)
            projected = self.norms[idx](projected)
            scores = torch.einsum("bld,d->bl", projected, self.pool_weights[idx])
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores.float(), dim=1).to(projected.dtype)
            weights = torch.nan_to_num(weights, nan=0.0)
            query = torch.sum(projected * weights.unsqueeze(-1), dim=1)
            queries.append(query)
            attn_weights.append(weights)

        stacked_queries = torch.stack(queries, dim=1)
        stacked_weights = torch.stack(attn_weights, dim=1)
        if return_weights:
            return stacked_queries, stacked_weights
        return stacked_queries, None
