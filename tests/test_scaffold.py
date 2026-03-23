from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    expected_directories = [
        ROOT / "config",
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "src" / "data",
        ROOT / "src" / "models",
        ROOT / "src" / "training",
        ROOT / "src" / "inference",
        ROOT / "src" / "dashboard" / "frontend",
        ROOT / "tests",
        ROOT / "docker",
        ROOT / "k8s",
    ]

    missing = [path for path in expected_directories if not path.is_dir()]
    assert not missing, f"Missing scaffold directories: {missing}"


def test_required_files_exist() -> None:
    expected_files = [
        ROOT / "CLAUDE.md",
        ROOT / "config" / "hparams.yaml",
        ROOT / "requirements.txt",
        ROOT / "src" / "data" / "paysim_loader.py",
        ROOT / "src" / "data" / "graph_builder.py",
        ROOT / "src" / "data" / "feature_engineer.py",
        ROOT / "src" / "data" / "smote.py",
        ROOT / "src" / "data" / "partitioner.py",
        ROOT / "src" / "models" / "gat.py",
        ROOT / "src" / "models" / "focal_loss.py",
        ROOT / "src" / "models" / "explainer.py",
        ROOT / "src" / "training" / "train.py",
        ROOT / "src" / "training" / "eval.py",
        ROOT / "src" / "training" / "ablation.py",
        ROOT / "src" / "inference" / "api.py",
        ROOT / "src" / "inference" / "graph_cache.py",
        ROOT / "src" / "inference" / "alerting.py",
        ROOT / "src" / "dashboard" / "backend.py",
        ROOT / "tests" / "test_data.py",
        ROOT / "tests" / "test_model.py",
        ROOT / "tests" / "test_inference.py",
        ROOT / "tests" / "test_api.py",
        ROOT / "docker" / "Dockerfile",
        ROOT / "docker" / "docker-compose.yml",
        ROOT / "k8s" / "deployment.yaml",
        ROOT / "k8s" / "hpa.yaml",
    ]

    missing = [path for path in expected_files if not path.is_file()]
    assert not missing, f"Missing scaffold files: {missing}"


def test_config_contains_phase_one_defaults() -> None:
    config = yaml.safe_load((ROOT / "config" / "hparams.yaml").read_text())

    assert config["data"]["npci"]["p2p_cap_inr"] == 100000.0
    assert config["data"]["fraud_rate"]["target"] == 0.0013
    assert config["data"]["simulation_window_days"] == 30


def test_requirements_cover_core_stack() -> None:
    contents = (ROOT / "requirements.txt").read_text()

    for package in ("pandas", "PyYAML", "torch", "fastapi", "pytest"):
        assert package in contents
