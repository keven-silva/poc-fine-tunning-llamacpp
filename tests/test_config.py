from pathlib import Path

from personas.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "qwen3-8b-personas.yaml"


def test_load_config_exposes_nested_attributes():
    cfg = load_config(CONFIG)
    assert cfg.model.base_id == "unsloth/Qwen3-8B"
    assert cfg.model.max_seq_length == 1536
    assert cfg.lora.r == 32
    assert cfg.train.per_device_train_batch_size == 1
    assert cfg.decode.top_k == 20


def test_effective_batch_size_is_sixteen():
    cfg = load_config(CONFIG)
    effective = cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps
    assert effective == 16


def test_splits_are_the_budget_from_adr_0012():
    cfg = load_config(CONFIG)
    assert (cfg.data.n_train, cfg.data.n_val, cfg.data.n_test) == (10000, 500, 200)

import os

import pytest

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))
LIGHT = CONFIGS_DIR / "qwen3-0.6b-personas.yaml"

REQUIRED_KEYS = {
    "model": ("base_id", "max_seq_length", "load_in_4bit"),
    "data": ("hf_dataset", "n_train", "n_val", "n_test", "seed", "max_name_drop_rate"),
    "lora": ("r", "lora_alpha", "target_modules"),
    "train": ("per_device_train_batch_size", "gradient_accumulation_steps",
              "num_train_epochs", "learning_rate", "group_by_length", "eval_strategy"),
    "decode": ("temperature", "top_p", "top_k", "repeat_penalty", "n_predict", "seed"),
    "llamacpp": ("commit",),
    "paths": ("data_dir", "outputs_dir", "llamacpp_dir"),
}


def _collisions(paths_by_config: dict[str, dict[str, str]]) -> list[str]:
    """Names every pair of configs that share a data or outputs directory.

    Paths are normalised, so 'data', './data' and 'data/' are the same directory.
    """
    seen: dict[tuple[str, str], str] = {}
    problems = []
    for name, paths in paths_by_config.items():
        for key in ("data_dir", "outputs_dir"):
            value = os.path.normpath(paths[key])
            owner = seen.setdefault((key, value), name)
            if owner != name:
                problems.append(f"{owner} and {name} share paths.{key} = {value}")
    return problems


def test_collisions_helper_normalises_spellings_of_the_same_directory():
    configs = {
        "a.yaml": {"data_dir": "data", "outputs_dir": "outputs"},
        "b.yaml": {"data_dir": "./data/", "outputs_dir": "outputs/b"},
    }
    assert _collisions(configs) == ["a.yaml and b.yaml share paths.data_dir = data"]


def test_collisions_helper_allows_a_dedicated_subfolder():
    configs = {
        "a.yaml": {"data_dir": "data", "outputs_dir": "outputs"},
        "b.yaml": {"data_dir": "data/b", "outputs_dir": "outputs/b"},
    }
    assert _collisions(configs) == []


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_every_config_has_the_keys_the_scripts_read(path):
    cfg = load_config(path)
    for section, keys in REQUIRED_KEYS.items():
        for key in keys:
            assert hasattr(getattr(cfg, section), key), f"{path.name}: {section}.{key} missing"


def test_no_two_configs_share_a_data_or_outputs_directory():
    paths = {p.name: vars(load_config(p).paths) for p in ALL_CONFIGS}
    assert _collisions(paths) == []


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_train_id_is_the_4bit_checkpoint_of_the_same_model_as_base_id(path):
    model = load_config(path).model
    if hasattr(model, "train_id"):
        assert model.train_id == model.base_id + "-bnb-4bit"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_in_training_eval_has_a_cadence(path):
    train = load_config(path).train
    if train.eval_strategy != "no":
        assert train.eval_steps > 0


def test_light_config_is_the_8b_task_with_only_the_documented_differences():
    light, heavy = load_config(LIGHT), load_config(CONFIG)
    assert light.model.base_id == "unsloth/Qwen3-0.6B"
    assert light.model.train_id == "unsloth/Qwen3-0.6B-bnb-4bit"
    assert light.model.max_seq_length == 2048
    assert light.data.hf_dataset == heavy.data.hf_dataset
    assert light.data.n_train == 2000
    assert (light.data.n_val, light.data.n_test) == (heavy.data.n_val, heavy.data.n_test)
    assert light.train.per_device_train_batch_size == 4
    assert light.train.gradient_accumulation_steps == 4
    assert light.train.eval_strategy == "steps"
    assert light.paths.data_dir == "data/qwen3-0.6b"
    assert light.paths.outputs_dir == "outputs/qwen3-0.6b"
    # Everything else is identical on purpose, so an A/B between sizes isolates the model.
    assert vars(light.lora) == vars(heavy.lora)
    assert vars(light.decode) == vars(heavy.decode)
    assert light.train.learning_rate == heavy.train.learning_rate
    assert light.train.lr_scheduler_type == heavy.train.lr_scheduler_type
    assert light.train.seed == heavy.train.seed
    assert light.llamacpp.commit == heavy.llamacpp.commit
    assert light.paths.llamacpp_dir == heavy.paths.llamacpp_dir
