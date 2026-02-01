from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch
import torch.nn as nn

from src.train.training import PFAGCNLightningModule, build_model

# ablation study: hydra-backed config helper for ablation checks.


def _compose_cfg(overrides: list[str]) -> object:  # ablation study
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="kaggle_config", overrides=overrides)


# ablation study
def test_ablation_study_decoupled_flags() -> None:
    cfg = _compose_cfg(
        [
            "task.num_functions=4",
            "model.graph.structure=decoupled",
            "model.dccn.enabled=false",
            "model.graph.protein_stacks=0",
            "model.graph.function_stacks=0",
        ]
    )
    model = build_model(cfg)
    assert model.use_dccn is False
    assert isinstance(model.dccn, nn.Identity)
    assert model.graph_structure == "decoupled"
    assert len(model.protein_blocks) == 0
    assert len(model.function_blocks) == 0


# ablation study
def test_ablation_study_metric_summary() -> None:
    cfg = _compose_cfg(["task.num_functions=4"])
    module = PFAGCNLightningModule(cfg=cfg, thresholds=[0.5], ia_weights=None)
    module._update_metric_summary("train/loss", 0.5)
    module._update_metric_summary("train/loss", 0.4)
    module._update_metric_summary("cafa/pr_auc", 0.2)
    module._update_metric_summary("cafa/pr_auc", 0.3)
    summary = module.summary_metrics()
    assert summary["train/loss_best"] == 0.4
    assert summary["train/loss_final"] == 0.4
    assert summary["cafa/pr_auc_best"] == 0.3
    assert summary["cafa/pr_auc_final"] == 0.3


def test_configure_gradient_clipping_signature_compatibility() -> None:
    cfg = _compose_cfg(["task.num_functions=4"])
    module = PFAGCNLightningModule(cfg=cfg, thresholds=[0.5], ia_weights=None)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    param = next(module.parameters())
    param.grad = torch.ones_like(param) * 10.0
    module.configure_gradient_clipping(optimizer, 0.5, "norm")
    assert param.grad.norm().item() <= 0.5 + 1.0e-3

    param.grad = torch.ones_like(param) * 10.0
    module.configure_gradient_clipping(optimizer, 0, 0.25, "norm")
    assert param.grad.norm().item() <= 0.25 + 1.0e-3
