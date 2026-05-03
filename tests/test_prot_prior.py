from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.utils.prot_prior import (
    ESMProjector,
    generate_prior,
    prot_prior_blast,
    prot_prior_esm,
)


def test_prot_prior_esm_mean_reduce() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(3, 4, 8)
    projector = ESMProjector(in_dim=8, out_dim=6)

    prior = prot_prior_esm(embeddings, projector=projector, reduce="mean")

    assert prior.shape == (3, 3)
    assert torch.allclose(prior, prior.T, atol=1e-6)
    assert torch.allclose(prior.diag(), torch.zeros(3), atol=1e-6)


def test_prot_prior_esm_cls_reduce() -> None:
    torch.manual_seed(1)
    embeddings = torch.randn(2, 5, 4)

    prior = prot_prior_esm(embeddings, reduce="cls")

    assert prior.shape == (2, 2)
    assert torch.allclose(prior, prior.T, atol=1e-6)


def test_prot_prior_esm_invalid_reduce() -> None:
    embeddings = torch.randn(1, 2, 3)
    with pytest.raises(ValueError):
        prot_prior_esm(embeddings, reduce="median")


def test_generate_prior_invalid_method() -> None:
    with pytest.raises(ValueError):
        generate_prior(torch.randn(2, 2, 2), method="unknown")


def _read_fasta_sequence(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    return "".join(line for line in lines if not line.startswith(">"))


def test_prot_prior_blast_builds_symmetric_adjacency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class DummyCompletedProcess:
        def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        query_path = command[command.index("-query") + 1]
        subject_path = command[command.index("-subject") + 1]
        query_seq = _read_fasta_sequence(query_path)
        subject_seq = _read_fasta_sequence(subject_path)
        if query_seq[-1] == subject_seq[-1]:
            return DummyCompletedProcess(stdout="1e-08\n")
        return DummyCompletedProcess(stdout="")

    monkeypatch.setattr("src.utils.prot_prior.subprocess.run", fake_run)
    monkeypatch.setattr(
        "src.utils.prot_prior._resolve_blastp_executable",
        lambda *args, **kwargs: "blastp",
    )

    seqs = ["AAAQ", "TTTQ", "GGGC"]
    prior = prot_prior_blast(seqs, evalue_threshold=1e-3)

    assert prior.shape == (3, 3)
    assert torch.allclose(prior, prior.T)
    assert torch.all(prior.diag() == 1.0)
    assert prior[0, 1] == prior[1, 0] == 1.0
    assert prior[0, 2] == prior[2, 0] == 0.0
    assert prior[1, 2] == prior[2, 1] == 0.0
    assert len(calls) == 3


def test_prot_prior_blast_raises_on_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyCompletedProcess:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, check, capture_output, text):
        return DummyCompletedProcess(returncode=2, stderr="crashed")

    monkeypatch.setattr("src.utils.prot_prior.subprocess.run", fake_run)
    monkeypatch.setattr(
        "src.utils.prot_prior._resolve_blastp_executable",
        lambda *args, **kwargs: "blastp",
    )

    with pytest.raises(RuntimeError, match="blastp execution failed"):
        prot_prior_blast(["AAAA", "TTTT"], evalue_threshold=1e-3)


def test_generate_prior_blast_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.ones(2, 2)
    captured: dict[str, tuple[tuple[str, ...], dict[str, float]]] = {}

    def fake_blast(data, **kwargs):
        captured["blast"] = (tuple(data), kwargs)
        return expected

    monkeypatch.setattr("src.utils.prot_prior.prot_prior_blast", fake_blast)

    result = generate_prior(["AA", "BB"], method="blast", evalue_threshold=1e-4)

    assert torch.equal(result, expected)
    assert captured["blast"][0] == ("AA", "BB")
    assert captured["blast"][1]["evalue_threshold"] == 1e-4
