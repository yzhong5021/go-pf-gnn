import optuna
import pytest
from optuna.study import StudyDirection
from omegaconf import OmegaConf

from src import hpo


def test_return_objective_applies_trial_suggestions(monkeypatch) -> None:
    recorded = {}

    def fake_run_training(cfg, extra_callbacks=None):
        recorded["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        recorded["callbacks"] = list(extra_callbacks or [])
        return 0.42

    monkeypatch.setattr(hpo, "run_training", fake_run_training)

    created_callbacks = []

    class DummyPruningCallback:
        def __init__(self, trial, monitor):
            created_callbacks.append({"trial": trial, "monitor": monitor})

    monkeypatch.setattr(hpo, "PyTorchLightningPruningCallback", DummyPruningCallback)

    base_cfg = OmegaConf.create({"training": {"max_epochs": 1}})
    search_space = [
        {"name": "optimizer.lr", "type": "float", "low": 1e-5, "high": 5e-3},
        {"name": "scheduler.name", "type": "categorical", "choices": ["step"]},
    ]
    trial = optuna.trial.FixedTrial({"optimizer.lr": 1e-4, "scheduler.name": "step"})
    objective = hpo.return_objective(base_cfg, search_space, max_epochs=3)

    result = objective(trial)

    assert result == 0.42
    cfg = recorded["cfg"]
    assert cfg["training"]["max_epochs"] == 3
    assert cfg["optimizer"]["lr"] == pytest.approx(1e-4, rel=1e-6)
    assert cfg["scheduler"]["name"] == "step"
    assert created_callbacks[0]["monitor"] == "cafa/ia_fmax"


def test_run_hpo_command_uses_cli_overrides(monkeypatch, tmp_path) -> None:
    recorded = {"overrides": None}
    trial_cfgs = []

    def fake_compose(config_path, config_name, overrides):
        recorded["overrides"] = list(overrides)
        cfg = OmegaConf.create(
            {
                "training": {"max_epochs": 10},
                "hpo": {"study": {"name": "test_study", "direction": "maximize", "seed": 7}},
                "hydra": {"runtime": {"output_dir": str(tmp_path)}, "job": {"name": "job"}},
                "data_config": {},
            }
        )
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(overrides)))
        return cfg

    def fake_finalize(cfg, *_args, **_kwargs):
        return None

    def fake_cache_root(_cfg):
        return tmp_path

    def fake_manifests(cfg, aspect, cache_root):
        assert aspect == "BP"
        assert cache_root == tmp_path
        return []

    def fake_run_training(cfg, extra_callbacks=None):
        trial_cfgs.append(OmegaConf.to_container(cfg, resolve=True))
        return 0.1

    class DummyPruningCallback:
        def __init__(self, trial, monitor):
            self.trial = trial
            self.monitor = monitor

    monkeypatch.setattr(hpo, "run_training", fake_run_training)
    monkeypatch.setattr(hpo, "PyTorchLightningPruningCallback", DummyPruningCallback)
    monkeypatch.setattr(hpo, "apply_system_env", lambda _cfg: None)

    study = hpo.run_hpo_command(
        config_path="configs",
        config_name="local_smoke",
        aspect="bp",
        n_trials=2,
        max_epochs=5,
        hydra_overrides=["training.max_epochs=1", "aspect=MF"],
        compose_config_fn=fake_compose,
        finalize_config_fn=fake_finalize,
        resolve_cache_root_fn=fake_cache_root,
        ensure_manifests_fn=fake_manifests,
    )

    assert recorded["overrides"] == [
        "training.max_epochs=1",
        "aspect=MF",
        "+aspect=BP",
        "training.max_epochs=5",
    ]
    assert study.direction == StudyDirection.MAXIMIZE
    assert study.best_value == pytest.approx(0.1)
    assert trial_cfgs
    for cfg in trial_cfgs:
        assert cfg["training"]["max_epochs"] == 5
        assert cfg.get("aspect") == "BP"
        assert cfg["hpo"]["study"]["n_trials"] == 2
        assert cfg["data_config"]["batch_size"] in {32, 64, 96}
        assert cfg["scheduler"]["name"] == "step"
        assert cfg["optimizer"]["lr"] > 0
