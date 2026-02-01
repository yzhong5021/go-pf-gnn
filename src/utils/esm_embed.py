"""Utilities for loading pretrained ESM embeddings for PF-AGCN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmModel, EsmTokenizer


class ESM_Embed(nn.Module):
    """Thin wrapper around HuggingFace ESM models with frozen weights."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | torch.device = "cpu",
        cache_dir: Optional[str | Path] = None,
        chunk_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        pretrained_kwargs = {}
        if self.cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(self.cache_dir)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, **pretrained_kwargs)
        self.model = EsmModel.from_pretrained(model_name, **pretrained_kwargs)
        self.model.eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        config = getattr(self.model, "config", None)
        max_positions = getattr(config, "max_position_embeddings", None)
        default_chunk = max(1, int(max_positions) - 2) if max_positions else 1022
        self.chunk_len = int(chunk_len) if chunk_len is not None else default_chunk
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Unable to determine ESM hidden size for empty inputs.")
        self.hidden_size = int(hidden_size)
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _embed_single(self, sequence: str) -> torch.Tensor:
        if not sequence:
            return torch.zeros((0, self.hidden_size), device=self.device)
        chunks = [
            sequence[start : start + self.chunk_len]
            for start in range(0, len(sequence), self.chunk_len)
        ]
        embeddings: list[torch.Tensor] = []
        for chunk in chunks:
            batch = self.tokenizer(
                [chunk],
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            input_ids = batch["input_ids"]
            residue_mask = (input_ids != self.pad_id) & (input_ids != self.cls_id) & (
                input_ids != self.eos_id
            )
            embeddings.append(outputs.last_hidden_state[0][residue_mask[0]])
        return torch.cat(embeddings, dim=0)

    @torch.inference_mode()
    def get_esm_embed(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return residue-level embeddings and masks for provided sequences."""
        embeddings = [self._embed_single(seq) for seq in seqs]
        if not embeddings:
            empty = torch.empty((0, 0, self.hidden_size), device=self.device)
            return empty, torch.empty((0, 0), dtype=torch.bool, device=self.device)
        lengths = torch.tensor([emb.size(0) for emb in embeddings], device=self.device)
        padded = pad_sequence(embeddings, batch_first=True)
        mask = torch.arange(padded.size(1), device=self.device)[None, :] < lengths[:, None]
        return padded, mask

    @torch.inference_mode()
    def forward(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """nn.Module forward passthrough for unified embedder interface."""
        return self.get_esm_embed(seqs)

    @torch.inference_mode()
    def get_embeddings(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias mirroring Prost wrapper interface."""
        return self.get_esm_embed(seqs)
