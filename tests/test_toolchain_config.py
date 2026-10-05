"""Guards for the ROCm toolchain rules (ADR 0015/0016). Pure file checks: no GPU needed."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_torch_comes_from_the_rocm71_index():
    cfg = tomllib.loads(_read("pyproject.toml"))
    index = {i["name"]: i["url"] for i in cfg["tool"]["uv"]["index"]}
    assert index["pytorch-rocm"] == "https://download.pytorch.org/whl/rocm7.1"
    sources = cfg["tool"]["uv"]["sources"]
    assert sources["torch"] == {"index": "pytorch-rocm"}
    assert sources["torchvision"] == {"index": "pytorch-rocm"}


def test_torch_is_pinned_to_the_version_validated_on_the_host():
    train = tomllib.loads(_read("pyproject.toml"))["project"]["optional-dependencies"]["train"]
    assert "torch==2.13.0" in train


def test_unsloth_and_zoo_are_not_pinned_individually():
    train = tomllib.loads(_read("pyproject.toml"))["project"]["optional-dependencies"]["train"]
    assert "unsloth" in train and "unsloth-zoo" in train


def test_lock_resolves_the_rocm71_wheel_and_no_foreign_runtimes():
    lock_text = _read("uv.lock")
    assert "+rocm7.1" in lock_text
    assert "+rocm7.14" not in lock_text
    for banned in ('name = "rocm-sdk-core"', 'name = "pytorch-triton-rocm"'):
        assert banned not in lock_text, f"{banned} must not be locked"
    # triton-rocm is torch's own dependency; triton and triton-rocm must both come
    # from the rocm7.1 index.
    pkgs = {p["name"]: p for p in tomllib.loads(lock_text)["package"]}
    for name in ("triton", "triton-rocm"):
        assert pkgs[name]["source"] == {"registry": "https://download.pytorch.org/whl/rocm7.1"}
