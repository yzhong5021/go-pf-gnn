"""Utilities for loading pretrained ProstT5 embeddings for PF-AGCN."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class ProstEmbed(nn.Module):
    """Thin wrapper around ProstT5 providing residue-level embeddings and masks."""

    def __init__(
        self,
        model_name: str = "Rostlab/ProstT5",
        device: str | torch.device = "cpu",
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        pretrained_kwargs = {}
        if self.cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(self.cache_dir)

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            legacy=True,
            **pretrained_kwargs,
        )
        self.model = T5EncoderModel.from_pretrained(model_name, **pretrained_kwargs)
        if self.device.type == "cpu":
            self.model.float()
        else:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.prefix_token = "<AA2fold>"
        self.prefix_token_id = self.tokenizer.convert_tokens_to_ids(self.prefix_token)
        self.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def forward(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """nn.Module forward passthrough for unified embedder interface."""
        return self.get_embeddings(sequences)

    @torch.inference_mode()
    def get_embeddings(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ProstT5 per-residue embeddings and the corresponding attention mask."""
        cleaned = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        prefixed = [f"{self.prefix_token} {sequence}" for sequence in cleaned]

        batch = self.tokenizer(
            prefixed,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        outputs = self.model(**batch)
        embeddings = outputs.last_hidden_state
        residue_mask = batch["attention_mask"].bool()
        input_ids = batch["input_ids"]
        if self.prefix_token_id is not None and self.prefix_token_id >= 0:
            residue_mask &= input_ids != self.prefix_token_id
        if self.pad_token_id is not None:
            residue_mask &= input_ids != self.pad_token_id
        return embeddings, residue_mask
