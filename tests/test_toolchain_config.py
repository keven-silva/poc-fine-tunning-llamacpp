"""Guards for the ROCm toolchain rules (ADR 0015/0016). Pure file checks: no GPU needed."""
import re
import subprocess
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
    assert "torch==2.11.0" in train


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


import re


def test_makefile_has_no_hsa_overrides():
    makefile = _read("Makefile")
    for var in ("HSA_OVERRIDE_GFX_VERSION", "HIPBLASLT_ENABLE_CK",
                "PYTORCH_TUNABLEOP_ENABLED"):
        assert var not in makefile


def test_gpu_recipes_clear_ld_library_path():
    makefile = _read("Makefile")
    assert re.search(r"^GPU_ENV\s*=\s*env -u LD_LIBRARY_PATH$", makefile, re.M)
    for script in ("00_check_gpu.py", "02_train.py", "03_export_gguf.py", "05_evaluate.py"):
        lines = [l for l in makefile.splitlines() if script in l]
        assert lines, f"{script} not run by the Makefile"
        assert all("$(GPU_ENV)" in l for l in lines), script


def test_setup_runs_the_real_kernel_check_before_building():
    setup = _read("Makefile").split("setup:", 1)[1].split("\ndata:", 1)[0]
    order = [setup.index(s) for s in ("sync", "00_check_gpu.py", "00_setup_rocm.sh",
                                      "00_setup_llamacpp.sh")]
    assert order == sorted(order)
    assert "is_available" not in setup


def test_rocm_setup_script_is_valid_bash_and_idempotent_by_design():
    path = ROOT / "scripts/00_setup_rocm.sh"
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    text = path.read_text()
    for pkg in ("hip-dev", "hipcc", "rocm-llvm", "rocm-device-libs", "hipblas",
                "hipblas-dev", "rocsolver", "hsa-rocr-dev", "rocm-cmake", "rocm-core"):
        assert re.search(rf"\b{re.escape(pkg)}\b", text), pkg
    assert "Packages.gz" in text                 # versions are resolved, never hardcoded
    assert "$VENDOR/debs/" in text and '! -s "$deb"' in text  # skips already downloaded debs
    assert "/opt/rocm-" in text                  # follows the host ROCm version


def test_no_cuda_setup_remains():
    assert not (ROOT / "scripts/00_setup_cuda.sh").exists()


def test_rocm_setup_keys_caches_by_version_and_fails_loudly_on_download():
    text = (ROOT / "scripts/00_setup_rocm.sh").read_text()
    assert 'INDEX="$VENDOR/Packages-$ROCM_VERSION-$DIST"' in text   # no stale index reuse
    assert 'DEBS="$VENDOR/debs/$ROCM_VERSION"' in text
    assert 'EXTRACT="$VENDOR/extract-$ROCM_VERSION"' in text
    assert 'FATAL: cannot download' in text                          # set -e skips `&&` lists
    assert 'curl -fsSL -o "$deb.part" "$REPO/$file" && ' not in text


def test_llamacpp_build_uses_current_hip_flags():
    text = _read("scripts/00_setup_llamacpp.sh")
    assert "-DGGML_HIP=ON" in text
    assert "GGML_HIPBLAS" not in text
    assert "gfx1102" in text
    assert "--list-devices" in text          # runtime assertion after the build
    assert "vendor/rocm/prefix" in text or "ROCM_VENDOR_DIR" in text
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/00_setup_llamacpp.sh")]).returncode == 0


def test_serve_script_clears_ld_library_path_and_has_no_stale_vram_claim():
    text = _read("scripts/04_serve.sh")
    assert "unset LD_LIBRARY_PATH" in text
    assert "9.5GB" not in text


def test_unsloth_and_zoo_lock_to_the_same_recent_release_family():
    # Coupled set (see CLAUDE.md): the resolver once backtracked to an 8-month-old
    # unsloth beside a newer zoo. Both must share year.month and be >= 2026.9.
    pkgs = {p["name"]: p["version"] for p in tomllib.loads(_read("uv.lock"))["package"]}
    family = {n: tuple(int(x) for x in pkgs[n].split(".")[:2]) for n in ("unsloth", "unsloth-zoo")}
    assert family["unsloth"] == family["unsloth-zoo"], family
    assert family["unsloth"] >= (2026, 9), family
