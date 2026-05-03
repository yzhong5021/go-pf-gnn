from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.utils import prost_embed


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self._prefix_id = 5
        self.last_sequences: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<AA2fold>":
            return self._prefix_id
        return -1

    def __call__(self, sequences, **kwargs):
        self.last_sequences = list(sequences)
        input_ids = torch.tensor([[self._prefix_id, 7, 8, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, **batch):
        input_ids = batch["input_ids"]
        bsz, seqlen = input_ids.shape
        hidden = torch.arange(seqlen, dtype=torch.float32).repeat(bsz, 1).view(bsz, seqlen, 1)
        hidden = hidden.expand(bsz, seqlen, 2)
        return SimpleNamespace(last_hidden_state=hidden)

    def eval(self):
        return self

    def to(self, device):
        return self

    def half(self):
        return self

    def float(self):
        return self


def test_prost_embed_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(
        prost_embed.T5Tokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(
        prost_embed.T5EncoderModel, "from_pretrained", lambda *args, **kwargs: DummyModel()
    )

    embedder = prost_embed.ProstEmbed(model_name="dummy")

    embeddings, mask = embedder(["AC"])

    assert tokenizer.last_sequences[0].startswith("<AA2fold>")
    assert embeddings.shape == (1, 4, 2)
    assert mask.shape == (1, 4)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0], torch.tensor([False, True, True, False]))
    assert all(not param.requires_grad for param in embedder.model.parameters())
