from pathlib import Path
import importlib

import yaml


def test_ppotune_alias():
    rlhf_ppotune = importlib.import_module("rlhf.ppotune")
    ppotune = importlib.import_module("ppotune")
    assert rlhf_ppotune is ppotune


def test_configs_and_scripts_exist():
    config_dir = Path("scripts/configs/ppo/tldr")
    script_dir = Path("scripts/bash/ppo")

    configs = sorted(config_dir.glob("*.yaml"))
    scripts = sorted(script_dir.glob("*.sh"))

    assert configs, "No PPO configs found"
    assert scripts, "No PPO bash scripts found"

    script_names = {s.stem for s in scripts}
    for cfg in configs:
        assert cfg.stem in script_names

        data = yaml.safe_load(cfg.read_text())
        assert "prefix" in data


def test_recipe_file_exists():
    recipe = Path("src/rlhf/recipes/ppo.py")
    assert recipe.exists()
