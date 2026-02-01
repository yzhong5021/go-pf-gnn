"""PF-AGCN architecture definition.

Implements the dilated-conv → sequence projection → adaptive graph workflow
using cached sequence embeddings and priors supplied at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
from types import SimpleNamespace

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:
    from model.config import PFAGCNConfig  # type: ignore
except ImportError:  # pragma: no cover
    from typing import Any as PFAGCNConfig  # type: ignore
from src.modules.adaptive_function import AdaptiveFunctionBlock
from src.modules.adaptive_protein import AdaptiveProteinBlock
from src.modules.dccn import DCCN_1D
from src.modules.head import ClassificationHead
from src.modules.pooling import AdaptivePooling
from src.modules.seq_final import SeqFinal
from src.modules.seq_gating import SeqGating


@dataclass
class Output:
    """Container returned by the model forward pass."""

    logits: torch.Tensor
    protein_embeddings: torch.Tensor
    function_embeddings: torch.Tensor
    protein_adjacency: Optional[torch.Tensor]
    function_adjacency: Optional[torch.Tensor]


class PFAGCN(nn.Module):
    """Protein Function Adaptive Graph Convolutional Network.

    Complete architecture:
    Sequence embeddings -> DCCN -> ESM + DCCN representational gating -> adaptive protein/function blocks -> classification head
    """

    def __init__(self, config: PFAGCNConfig) -> None:
        super().__init__()
        self.config = config
        model_cfg = config.model
        self.num_functions = config.task.num_functions

        seq_cfg = model_cfg.seq_embeddings
        seq_dim = seq_cfg.feature_dim
        backend_value = getattr(seq_cfg, "backend", "esm") or "esm"
        seq_backend = str(backend_value).lower()
        raw_dim = getattr(seq_cfg, "raw_dim", None)
        if seq_backend == "esm":
            raw_dim = int(raw_dim) if raw_dim is not None else 1280
        self.seq_input_dim = seq_dim
        if seq_backend == "esm" and raw_dim != seq_dim:
            self.seq_input_dim = int(raw_dim)
            self.seq_projector = nn.Linear(self.seq_input_dim, seq_dim, bias=False)
            nn.init.xavier_uniform_(self.seq_projector.weight)
        else:
            self.seq_projector = nn.Identity()

        dccn_cfg = model_cfg.dccn
        self.use_dccn = bool(getattr(dccn_cfg, "enabled", True))  # ablation study
        dccn_channels = dccn_cfg.channels
        graph_dim = model_cfg.seq_final.graph_dim
        shared_dim = model_cfg.seq_gating.shared_dim
        prot_prior_cfg = getattr(model_cfg, "prot_prior", SimpleNamespace())
        self.use_protein_prior = bool(getattr(prot_prior_cfg, "enabled", False))
        self.protein_prior_method = getattr(prot_prior_cfg, "method", "cosine")
        sim_threshold = getattr(prot_prior_cfg, "sim_threshold", 0.3)
        self.protein_prior_sim_threshold = (
            float(sim_threshold) if sim_threshold is not None else None
        )
        self.use_go_prior = bool(getattr(model_cfg.go_prior, "enabled", True))

        if shared_dim != graph_dim:
            raise ValueError("seq_gating.shared_dim must equal seq_final.graph_dim")

        if seq_dim != dccn_channels:
            self.dccn_input = nn.Linear(seq_dim, dccn_channels, bias=False)
            nn.init.xavier_uniform_(self.dccn_input.weight)
        else:
            self.dccn_input = nn.Identity()

        if self.use_dccn:  # ablation study
            self.dccn = DCCN_1D(
                embed_len=dccn_channels,
                k_size=dccn_cfg.kernel_size,
                dilation=dccn_cfg.dilation,
                dropout=dccn_cfg.dropout,
            )
        else:
            self.dccn = nn.Identity()  # ablation study

        self.seq_gating = SeqGating(
            d_shared=shared_dim,
            d_esm=seq_dim,
            c_dcc=dccn_channels,
            attn_hidden=model_cfg.seq_gating.attn_hidden,
            dropout=model_cfg.seq_gating.dropout,
        )

        seq_final_cfg = model_cfg.seq_final
        graph_cfg = getattr(model_cfg, "graph", None) or SimpleNamespace()
        self.graph_structure = str(getattr(graph_cfg, "structure", "decoupled")).lower()
        if self.graph_structure not in {"decoupled", "alternating"}:
            raise ValueError("graph.structure must be either 'decoupled' or 'alternating'.")

        self.protein_stack_depth = int(getattr(graph_cfg, "protein_stacks", 2))
        self.function_stack_depth = int(getattr(graph_cfg, "function_stacks", 2))
        self.alternating_depth = int(getattr(graph_cfg, "alternating_depth", 4))
        if self.graph_structure == "alternating" and self.alternating_depth <= 0:
            raise ValueError("alternating graph structure requires alternating_depth > 0.")

        self.function_feature_dim = getattr(seq_final_cfg, "function_dim", graph_dim)
        self.protein_feature_dim = getattr(seq_final_cfg, "protein_dim", graph_dim)

        self.seq_final = SeqFinal(
            in_dim=shared_dim,
            N_C=self.num_functions,
            proj=seq_final_cfg.metric_dim,
            out_ch=graph_dim,
            decoupled=self.graph_structure == "decoupled",
            function_dim=self.function_feature_dim,
            protein_dim=self.protein_feature_dim,
        )

        head_cfg = model_cfg.head
        head_dim = getattr(head_cfg, "feature_dim", graph_dim)
        head_attn_hidden = getattr(head_cfg, "attn_hidden", model_cfg.seq_gating.attn_hidden)

        self.head: Optional[ClassificationHead]
        self.alternating_classifier: Optional[nn.Linear]
        self.protein_summary_pool: Optional[AdaptivePooling]
        self.function_summary_pool: Optional[AdaptivePooling]

        if self.graph_structure == "decoupled":
            self.function_blocks = nn.ModuleList(
                [
                    AdaptiveFunctionBlock(
                        d_in=self.function_feature_dim,
                        d_attn=model_cfg.adaptive_function.attention_dim,
                        steps=model_cfg.adaptive_function.steps,
                        p=model_cfg.adaptive_function.top_p_mass,
                        tau=model_cfg.adaptive_function.temperature,
                        dropout=model_cfg.adaptive_function.dropout,
                    )
                    for _ in range(self.function_stack_depth)
                ]
            )
            self.protein_blocks = nn.ModuleList(
                [
                    AdaptiveProteinBlock(
                        d_in=self.protein_feature_dim,
                        d_attn=model_cfg.adaptive_protein.attention_dim,
                        steps=model_cfg.adaptive_protein.steps,
                        p=model_cfg.adaptive_protein.top_p_mass,
                        tau=model_cfg.adaptive_protein.temperature,
                        dropout=model_cfg.adaptive_protein.dropout,
                    )
                    for _ in range(self.protein_stack_depth)
                ]
            )
            self.alternating_blocks = nn.ModuleList()
            self.alternating_classifier = None
            self.protein_summary_pool = None
            self.function_summary_pool = None
            self.head = ClassificationHead(
                N_C=self.num_functions,
                d_in=head_dim,
                protein_input_dim=self.protein_feature_dim,
                function_input_dim=self.function_feature_dim,
                dropout=head_cfg.dropout,
                attn_hidden=head_attn_hidden,
            )
        else:
            self.function_blocks = nn.ModuleList()
            self.protein_blocks = nn.ModuleList()
            self.head = None
            self.alternating_blocks = nn.ModuleList(
                [
                    (
                        AdaptiveFunctionBlock(
                            d_in=graph_dim,
                            d_attn=model_cfg.adaptive_function.attention_dim,
                            steps=model_cfg.adaptive_function.steps,
                            p=model_cfg.adaptive_function.top_p_mass,
                            tau=model_cfg.adaptive_function.temperature,
                            dropout=model_cfg.adaptive_function.dropout,
                        )
                        if idx % 2 == 0
                        else AdaptiveProteinBlock(
                            d_in=graph_dim,
                            d_attn=model_cfg.adaptive_protein.attention_dim,
                            steps=model_cfg.adaptive_protein.steps,
                            p=model_cfg.adaptive_protein.top_p_mass,
                            tau=model_cfg.adaptive_protein.temperature,
                            dropout=model_cfg.adaptive_protein.dropout,
                        )
                    )
                    for idx in range(self.alternating_depth)
                ]
            )
            self.alternating_classifier = nn.Linear(graph_dim, 1)
            nn.init.xavier_uniform_(self.alternating_classifier.weight)
            nn.init.zeros_(self.alternating_classifier.bias)
            self.protein_summary_pool = AdaptivePooling(
                embed_dim=graph_dim,
                attn_hidden=head_attn_hidden,
                dropout=head_cfg.dropout,
            )
            self.function_summary_pool = AdaptivePooling(
                embed_dim=graph_dim,
                attn_hidden=head_attn_hidden,
                dropout=head_cfg.dropout,
            )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        protein_prior: Optional[torch.Tensor] = None,
        go_prior: Optional[torch.Tensor] = None,
    ) -> Output:
        """Run end-to-end inference with cached sequence embeddings."""

        self._log_cuda_memory("forward:start")
        if seq_embeddings.ndim != 3:
            raise ValueError("seq_embeddings must be a 3D tensor (batch, length, dim).")

        batch, seqlen, feat_dim = seq_embeddings.shape
        # print("batch, seqlen, features:", [batch, seqlen, feat_dim])
        
        expected_dim = self.seq_input_dim
        if feat_dim != expected_dim:
            raise ValueError(
                f"Expected seq_embeddings feature dim {expected_dim}, got {feat_dim}."
            )

        if mask is None and lengths is None:
            raise ValueError("Either lengths or mask must be provided.")

        seq_embeddings = self.seq_projector(seq_embeddings)
        self._log_cuda_memory("after seq_projector")
        device = seq_embeddings.device
        mask_bool, lengths_tensor = self._normalise_mask(lengths, mask, batch, seqlen, device)

        if not self.use_go_prior:
            go_prior = None
        elif go_prior is None:
            log.debug("GO prior enabled but missing; proceeding without it for this batch.")

        embeddings_projected = self.dccn_input(seq_embeddings)
        self._log_cuda_memory("after dccn_input")
        # print("\n\n\nEmbeddings_projected", embeddings_projected.shape)
        if self.use_dccn:  # ablation study
            conv_features = self.dccn(embeddings_projected, mask=mask_bool)
        else:
            conv_features = embeddings_projected  # ablation study
        self._log_cuda_memory("after dccn")
        # print("\n\n\nConv_features", conv_features.shape)

        
        gating_repr = self.seq_gating(
            H_esm=seq_embeddings,
            H_dcc=conv_features,
            lengths=lengths_tensor,
            mask=mask_bool,
        )
        self._log_cuda_memory("after seq_gating")

        # print("\n\n\nGating_repr", gating_repr.shape)

        if self.graph_structure == "decoupled":
            function_init, protein_init = self.seq_final(
                gating_repr,
                mode="decoupled",
            )
            self._log_cuda_memory("after seq_final")
            protein_prior_matrix = self._resolve_protein_prior(protein_init, protein_prior)
            function_prior_matrix = self._resolve_function_prior(
                go_prior, function_init.device, function_init.dtype
            )
            protein_features, protein_adj = self._apply_stack(
                self.protein_blocks, protein_init, protein_prior_matrix
            )
            function_features, function_adj = self._apply_stack(
                self.function_blocks, function_init, function_prior_matrix
            )
            logits, protein_embeddings, function_embeddings = self.head(
                function_features,
                protein_features,
            )
            self._log_cuda_memory("after head")
        else:
            shared_features = self.seq_final(
                gating_repr,
                mode="alternating",
            )
            self._log_cuda_memory("after seq_final")
            protein_prior_matrix = self._resolve_protein_prior(shared_features, protein_prior)
            function_prior_matrix = self._resolve_function_prior(
                go_prior, shared_features.device, shared_features.dtype
            )
            shared_features, protein_adj, function_adj = self._run_alternating_blocks(
                shared_features, protein_prior_matrix, function_prior_matrix
            )
            logits = self.alternating_classifier(shared_features).squeeze(-1)
            self._log_cuda_memory("after alternating_classifier")
            protein_embeddings = self.protein_summary_pool(shared_features)
            self._log_cuda_memory("after protein_summary_pool")
            function_embeddings = self.function_summary_pool(
                shared_features.permute(1, 0, 2)
            )
            self._log_cuda_memory("after function_summary_pool")

        protein_adj = self._ensure_adjacency(
            protein_adj,
            protein_embeddings.size(0),
            protein_embeddings.device,
            protein_embeddings.dtype,
        )
        function_adj = self._ensure_adjacency(
            function_adj,
            self.num_functions,
            function_embeddings.device,
            function_embeddings.dtype,
        )

        return Output(
            logits=logits,
            protein_embeddings=protein_embeddings,
            function_embeddings=function_embeddings,
            protein_adjacency=protein_adj,
            function_adjacency=function_adj,
        )

    def _log_cuda_memory(self, label: str) -> None:  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        if not torch.cuda.is_available():  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
            log.info("CUDA not available; skipping memory log.")  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
            return  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        mem_mb = torch.cuda.memory_allocated() / 1e6  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        log.info(  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
            "CUDA memory %s at %s: %.2f MB", label, timestamp, mem_mb  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        )

    def _normalise_mask(
        self,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        batch: int,
        seqlen: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            if mask.shape != (batch, seqlen):
                raise ValueError("mask must match the batch and sequence dimensions.")
            mask_bool = mask.to(device=device, dtype=torch.bool)
            lengths_tensor = mask_bool.sum(dim=1, dtype=torch.long)
            if lengths is not None:
                lengths = lengths.to(device=device, dtype=torch.long)
                if not torch.equal(lengths_tensor, lengths):
                    raise ValueError("mask and lengths encode conflicting values.")
        else:
            if lengths is None or lengths.ndim != 1:
                raise ValueError("lengths must be a 1D tensor when mask is absent.")
            lengths_tensor = lengths.to(device=device, dtype=torch.long)
            mask_bool = (
                torch.arange(seqlen, device=device)[None, :]
                < lengths_tensor[:, None]
            )
        return mask_bool, lengths_tensor

    def _apply_stack(
        self,
        blocks: nn.ModuleList,
        features: torch.Tensor,
        prior: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        adjacency: Optional[torch.Tensor] = None
        current = features
        if not blocks:
            return current, None

        idx = 0
        for block in blocks:
            current = block(current, prior=prior)
            self._log_cuda_memory(f"{block.__class__.__name__}[{idx}]")
            idx += 1
            adjacency = getattr(block, "last_attention", None)
        return current, adjacency

    def _run_alternating_blocks(
        self,
        features: torch.Tensor,
        protein_prior: Optional[torch.Tensor],
        function_prior: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        protein_adj: Optional[torch.Tensor] = None
        function_adj: Optional[torch.Tensor] = None
        current = features

        idx = 0
        for block in self.alternating_blocks:
            prior = protein_prior if block.axis == "protein" else function_prior
            current = block(current, prior=prior)
            self._log_cuda_memory(f"{block.__class__.__name__}[{idx}]")
            idx += 1
            if block.axis == "protein":
                protein_adj = getattr(block, "last_attention", None)
            else:
                function_adj = getattr(block, "last_attention", None)

        return current, protein_adj, function_adj

    def _resolve_protein_prior(
        self,
        tensor: torch.Tensor,
        explicit_prior: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_protein_prior:
            return None

        nodes = tensor.size(0)
        if explicit_prior is not None:
            return self._validate_prior_matrix(
                explicit_prior, nodes, "protein_prior", tensor.device, tensor.dtype
            )

        pooled = tensor.mean(dim=1)
        return self._build_protein_prior_from_summary(pooled)

    def _resolve_function_prior(
        self,
        go_prior: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_go_prior or go_prior is None:
            return None
        return self._validate_prior_matrix(
            go_prior, self.num_functions, "go_prior", device, dtype
        )

    def _validate_prior_matrix(
        self,
        prior: torch.Tensor,
        size: int,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
            raise ValueError(f"{name} must be a square 2D tensor.")
        if prior.shape[0] != size:
            raise ValueError(f"{name} size mismatch: expected {size}, got {prior.shape[0]}.")
        return prior.to(device=device, dtype=dtype)

    def _build_protein_prior_from_summary(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.size(0) == 0:
            return torch.zeros((0, 0), device=pooled.device, dtype=pooled.dtype)

        method = str(self.protein_prior_method).lower()
        apply_threshold = method == "cosine"
        if method == "identity":
            return torch.eye(pooled.size(0), device=pooled.device, dtype=pooled.dtype)
        if method != "cosine":
            log.warning("Unknown protein prior method '%s'; defaulting to cosine similarity.", method)
            apply_threshold = True

        normed = F.normalize(pooled, dim=1)
        cosine = torch.matmul(normed, normed.T)
        cosine = torch.clamp(cosine, min=0.0)

        threshold = self.protein_prior_sim_threshold if apply_threshold else None
        if threshold is not None:
            cosine = torch.where(cosine >= threshold, cosine, torch.zeros_like(cosine))
        return cosine

    def _ensure_adjacency(
        self,
        adj: Optional[torch.Tensor],
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if adj is None:
            return None
        return adj.to(device=device, dtype=dtype)
