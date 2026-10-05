# ROCm Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the whole pipeline (`make setup → data → train → export → serve → eval → report`) run on the Radeon RX 7600 8GB with ROCm, and document it the way the project documents decisions.

**Architecture:** The torch wheel's HIP runtime must match the host ROCm (7.1.1), so torch moves to the `rocm7.1` index. llama.cpp is built with HIP using a project-local toolchain in `vendor/rocm`, extracted without root from AMD's 7.1.1 `.deb` files and overlaid on the host's `/opt/rocm-7.1.1` as a symlink tree. Config guard tests pin these rules so they cannot regress silently.

**Tech Stack:** uv, torch 2.13.0+rocm7.1, Unsloth/bitsandbytes/trl/peft, llama.cpp (HIP, gfx1102), bash, pytest, GNU make.

**Spec:** [docs/superpowers/specs/2026-10-05-rocm-migration-design.md](../specs/2026-10-05-rocm-migration-design.md)

## Global Constraints

- Python is always the project `uv` venv on 3.12; never `pip install` into the host Python.
- `torch` and `torchvision` both come from `https://download.pytorch.org/whl/rocm7.1`; `torch==2.13.0`.
- No `rocm*` pip package may be installed (host runtime only); no `pytorch-triton-rocm` / `triton-rocm`.
- `unsloth` and `unsloth-zoo` are never pinned individually; the resolver picks the coupled set.
- llama.cpp stays at commit `bdeb855b30dfe7f6e695cba98445a7ba09e6416e`, built with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1102` (not `GGML_HIPBLAS`).
- `HSA_OVERRIDE_GFX_VERSION`, `HIPBLASLT_ENABLE_CK`, `PYTORCH_TUNABLEOP_ENABLED` are not set unless the smoke run proves a need.
- GPU recipes run with `LD_LIBRARY_PATH` cleared (`env -u LD_LIBRARY_PATH`).
- `TORCHINDUCTOR_COMPILE_THREADS=4` stays above the torch import in `scripts/02_train.py`.
- Accepted ADRs, `docs/RESULTS.md`, `docs/GPU-COMPARISON.md` and the 2026-09-12 spec bodies are not rewritten.
- Nothing in `src/personas/` changes (invariants 1, 2, 3, 6 stay untouched).
- No placeholder text in committed files; measured numbers come from the smoke run (Task 5).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Host ROCm is not 7.1.1 (e.g. upgraded): `00_setup_rocm.sh` must read the version from `/opt/rocm-*` and the wheel check must warn on a major.minor mismatch, not silently proceed.
2. Re-running `make setup` on an already-set-up tree: setup scripts must be idempotent (no re-download, no failure on existing symlink tree).
3. The user shell exports `LD_LIBRARY_PATH` with non-existent ROCm dirs: every GPU recipe and `04_serve.sh` must work with it set (guard test on the Makefile; `04_serve.sh` unsets it).
4. `torch.cuda.is_available()` is `True` while kernels crash: the GPU check must run a real kernel, and must print its diagnosis *before* it (a segfault cannot print afterwards).
5. `vendor/` missing or partially populated (it was deleted once): setup must rebuild from nothing, with a clear error if the AMD repo or `Packages.gz` is unreachable.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/test_toolchain_config.py` (new) | guard tests for pyproject, lock, Makefile, setup scripts |
| `pyproject.toml`, `uv.lock` | rocm7.1 index, torch pin |
| `scripts/00_check_gpu.py` (new) | real-kernel GPU check, host/wheel ROCm version comparison |
| `Makefile` | `GPU_ENV`, setup order, no HSA overrides |
| `scripts/00_setup_rocm.sh` | download/extract/overlay the HIP toolchain into `vendor/rocm` |
| `scripts/00_setup_llamacpp.sh` | HIP build against `vendor/rocm/prefix`, runtime assertion |
| `scripts/00_setup_cuda.sh` | deleted |
| `scripts/04_serve.sh`, `scripts/06_report.py` | remove CUDA-era text |
| `docs/adr/0015-*.md`, `0016-*.md`, `docs/adr/README.md` | decisions |
| `CLAUDE.md`, `README.md`, 2026-09-12 spec | pointers and hardware text |

---

### Task 1: Pin torch to the ROCm 7.1 wheel, with guard tests

**Files:**
- Create: `tests/test_toolchain_config.py`
- Modify: `pyproject.toml`
- Regenerate: `uv.lock`

**Interfaces:**
- Produces: `uv` venv with `torch 2.13.0+rocm7.1`, consumed by every later task; `tests/test_toolchain_config.py` module with helper `_read(path) -> str` reused by Tasks 2-4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_toolchain_config.py`:

```python
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
    lock = _read("uv.lock")
    assert "+rocm7.1" in lock
    assert "+rocm7.14" not in lock
    for banned in ('name = "rocm-sdk-core"', 'name = "pytorch-triton-rocm"',
                   'name = "triton-rocm"'):
        assert banned not in lock, f"{banned} must not be locked"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -15` (if `uv run` cannot start because the venv is gone, run `uv sync --extra dev` first)
Expected: FAIL (`rocm7.14` index and no `torch==2.13.0` pin).

- [ ] **Step 3: Edit `pyproject.toml`**

Change the first `train` entry and the index block. In `[project.optional-dependencies]`:

```toml
train = [
    "torch==2.13.0",
    "torchvision",
```

Replace the index block and the comment above it with:

```toml
# torch rocm7.1 wheels. The wheel's bundled HIP runtime MUST match the host ROCm
# (7.1.1 in /opt/rocm-7.1.1): the rocm7.14 wheel segfaults in libamdhip64 at the first
# kernel launch on this host while torch.cuda.is_available() still returns True. This is
# the opposite of CUDA's backward-compatibility rule in ADR 0011 (see ADR 0015).
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.1"
```

Also in `[tool.uv.sources]` change "different ROCm major version" comment wording to: `# torchvision MUST come from the same index as torch: transformers imports it, and a build against a different ROCm release aborts the import.` Leave the `torch` and `torchvision` source lines unchanged.

- [ ] **Step 4: Relock and rebuild the venv from scratch**

```bash
uv lock
uv sync --reinstall --extra train --extra eval --extra dev > /tmp/claude-1000/sync.log 2>&1; echo "exit $?"; tail -5 /tmp/claude-1000/sync.log
```
Expected: `exit 0`. Then:

```bash
uv pip list 2>/dev/null | grep -iE "^(torch|torchvision|unsloth|unsloth-zoo|bitsandbytes|triton|rocm|pytorch-triton)"
```
Expected: `torch 2.13.0+rocm7.1`, `torchvision 0.28.0+rocm7.1`, `unsloth` and `unsloth-zoo` present, `triton 3.8.0`, and **no** `rocm*`, `pytorch-triton-rocm`, `triton-rocm` lines.

If a `rocm-sdk-core` still appears, it is a transitive dependency of the wheel: stop and report it (the spec's premise would be wrong), do not paper over it.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -10`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_toolchain_config.py
git commit -m "fix: pin torch to the rocm7.1 wheel that matches the host runtime

The rocm7.14 wheel segfaults at the first kernel launch against the host's
ROCm 7.1.1 stack. Guard tests keep the index, the pin and the absence of
pip-provided ROCm runtimes from regressing.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Real-kernel GPU check and Makefile

**Files:**
- Create: `scripts/00_check_gpu.py`
- Modify: `Makefile` (whole file), `tests/test_toolchain_config.py` (append)

**Interfaces:**
- Consumes: Task 1 venv.
- Produces: `scripts/00_check_gpu.py` (exit 0 on success), Makefile variable `GPU_ENV = env -u LD_LIBRARY_PATH` used by Tasks 3-5.

- [ ] **Step 1: Append the failing Makefile tests**

Append to `tests/test_toolchain_config.py`:

```python
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
    order = [setup.index(s) for s in ("uv", "00_check_gpu.py", "00_setup_rocm.sh",
                                      "00_setup_llamacpp.sh")]
    assert order == sorted(order)
    assert "is_available" not in setup
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -12`
Expected: the three new tests FAIL.

- [ ] **Step 3: Create `scripts/00_check_gpu.py`**

```python
#!/usr/bin/env python
"""Stage 0 -- prove torch can run a kernel on the GPU, not merely see it.

`torch.cuda.is_available()` returned True with a wheel whose HIP runtime segfaulted at the
first kernel launch (ADR 0015), so the only meaningful check is to launch one. A segfault
kills this process before it can print anything, so the diagnosis is printed first.
"""
import glob
import re
import sys

import torch


def host_rocm_versions() -> list[str]:
    return sorted(
        re.sub(r".*/rocm-", "", p) for p in glob.glob("/opt/rocm-*") if re.search(r"rocm-\d", p)
    )


def main() -> int:
    if torch.version.hip is None:
        print("FATAL: torch is not a ROCm build (torch.version.hip is None).", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FATAL: torch cannot see the GPU. Check /dev/kfd permissions "
              "(groups render, video) and the host ROCm install.", file=sys.stderr)
        return 1

    wheel_hip = torch.version.hip.split("-")[0]
    hosts = host_rocm_versions()
    print(f">> torch {torch.__version__}, HIP {wheel_hip}, device {torch.cuda.get_device_name(0)}")
    print(f">> host ROCm trees: {', '.join(hosts) or 'none found'}")
    minor = ".".join(wheel_hip.split(".")[:2])
    if hosts and not any(h.startswith(minor) for h in hosts):
        print(f"WARNING: wheel HIP {wheel_hip} does not match any host ROCm ({', '.join(hosts)}). "
              "A mismatch can segfault at the first kernel; see ADR 0015.", file=sys.stderr)

    print(">> running a bf16 matmul (a segfault, exit 139, here means the wheel's HIP runtime "
          "does not match the host ROCm; see ADR 0015)")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    value = (a @ a).float().sum().item()
    if value != value:  # NaN
        print("FATAL: matmul returned NaN.", file=sys.stderr)
        return 1
    print(">> GPU kernel ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Replace `Makefile`**

```make
CONFIG ?= configs/qwen3-8b-personas.yaml
UV     ?= uv

# The user's shell may export LD_LIBRARY_PATH entries for ROCm trees that do not exist
# (e.g. /opt/rocm-6.4.2). Every recipe that runs GPU code clears it so the torch wheel and
# the llama.cpp binaries resolve their own libraries (ADR 0015, 0016).
GPU_ENV = env -u LD_LIBRARY_PATH

.PHONY: setup data train export serve eval report smoke test clean

setup:
	$(UV) sync --extra train --extra eval --extra dev
	$(GPU_ENV) $(UV) run python scripts/00_check_gpu.py
	bash scripts/00_setup_rocm.sh
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train:
	$(GPU_ENV) $(UV) run python scripts/02_train.py --config $(CONFIG)

export:
	$(GPU_ENV) $(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which tuned
	$(GPU_ENV) $(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which base

serve:
	bash scripts/04_serve.sh $(MODEL)

eval:
	$(GPU_ENV) $(UV) run python scripts/05_evaluate.py --config $(CONFIG)

report:
	$(UV) run python scripts/06_report.py --config $(CONFIG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG) --smoke
	@echo "Smoke data ready. Run: make train export eval report"

clean:
	rm -rf outputs/merged-16bit-* outputs/gguf/*-f16.gguf
```

(The Makefile must keep tab-indented recipes.)

- [ ] **Step 5: Run the tests and the kernel check**

```bash
uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -10
env -u LD_LIBRARY_PATH uv run python scripts/00_check_gpu.py > /tmp/claude-1000/gpu.log 2>&1; echo "exit $?"; cat /tmp/claude-1000/gpu.log
```
Expected: all tests pass; `exit 0`; log ends with `>> GPU kernel ok` and shows `HIP 7.1.` plus host tree `7.1.1`, no WARNING.

- [ ] **Step 6: Commit**

```bash
git add Makefile scripts/00_check_gpu.py tests/test_toolchain_config.py
git commit -m "fix: verify the GPU with a real kernel and drop the gfx override

is_available() passed with a runtime that segfaulted at the first kernel.
GPU recipes now clear LD_LIBRARY_PATH inherited from the user's shell.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: HIP toolchain without root (`00_setup_rocm.sh`)

**Files:**
- Modify: `scripts/00_setup_rocm.sh` (whole file), `tests/test_toolchain_config.py` (append)

**Interfaces:**
- Produces: `vendor/rocm/prefix` (merged `ROCM_PATH` tree, contains `bin/hipcc`, `llvm/bin/clang++`), `vendor/rocm/extract/opt/rocm-<ver>/lib`. Task 4 consumes both; env var `ROCM_VERSION` and dir var `ROCM_VENDOR_DIR` (default `vendor/rocm`).

- [ ] **Step 1: Append failing tests**

```python
import subprocess


def test_rocm_setup_script_is_valid_bash_and_idempotent_by_design():
    path = ROOT / "scripts/00_setup_rocm.sh"
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    text = path.read_text()
    for pkg in ("hip-dev", "hipcc", "rocm-llvm", "rocm-device-libs", "hipblas",
                "hipblas-dev", "rocsolver", "hsa-rocr-dev", "rocm-cmake", "rocm-core"):
        assert re.search(rf"\b{re.escape(pkg)}\b", text), pkg
    assert "Packages.gz" in text                 # versions are resolved, never hardcoded
    assert "$VENDOR/debs/" in text and "-s" in text  # skips already downloaded debs
    assert "/opt/rocm-" in text                  # follows the host ROCm version


def test_no_cuda_setup_remains():
    assert not (ROOT / "scripts/00_setup_cuda.sh").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -8`
Expected: both new tests FAIL (old script lacks the package list; `00_setup_cuda.sh` exists).

- [ ] **Step 3: Replace `scripts/00_setup_rocm.sh`**

```bash
#!/usr/bin/env bash
# Stage 0 -- project-local HIP toolchain for the llama.cpp build (ADR 0016).
#
# The host has the ROCm *runtime* (/opt/rocm-<ver>) but not hipcc, the HIP headers or
# rocm-llvm, and there is no passwordless sudo. The missing packages are downloaded from
# AMD's apt repository for the SAME ROCm version, unpacked with dpkg -x, and overlaid on a
# symlink copy of the host tree. Nothing outside vendor/rocm is modified.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-$(ls -d /opt/rocm-[0-9]* 2>/dev/null | sed 's|.*/rocm-||' | sort -V | tail -1)}"
[ -n "$ROCM_VERSION" ] || { echo "FATAL: no /opt/rocm-<version> on this host." >&2; exit 1; }
HOST_ROCM="/opt/rocm-$ROCM_VERSION"
[ -d "$HOST_ROCM" ] || { echo "FATAL: $HOST_ROCM not found." >&2; exit 1; }

VENDOR="$(mkdir -p "${ROCM_VENDOR_DIR:-vendor/rocm}" && cd "${ROCM_VENDOR_DIR:-vendor/rocm}" && pwd)"
DIST="${ROCM_APT_DIST:-jammy}"
REPO="https://repo.radeon.com/rocm/apt/$ROCM_VERSION"
PACKAGES="hip-dev hipcc rocm-llvm rocm-device-libs hipblas hipblas-dev rocsolver hsa-rocr-dev rocm-cmake rocm-core"

mkdir -p "$VENDOR/debs" "$VENDOR/extract"
echo ">> host ROCm $ROCM_VERSION; fetching missing HIP toolchain from $REPO ($DIST)"

if [ ! -s "$VENDOR/Packages" ]; then
  curl -fsSL "$REPO/dists/$DIST/main/binary-amd64/Packages.gz" | gunzip > "$VENDOR/Packages.tmp" \
    || { echo "FATAL: cannot fetch $REPO/dists/$DIST/main/binary-amd64/Packages.gz" >&2; exit 1; }
  mv "$VENDOR/Packages.tmp" "$VENDOR/Packages"
fi

for pkg in $PACKAGES; do
  file="$(awk -v p="$pkg" '$1=="Package:"&&$2==p{f=1} f&&$1=="Filename:"{print $2; exit}' "$VENDOR/Packages")"
  [ -n "$file" ] || { echo "FATAL: package $pkg not in $REPO ($DIST)" >&2; exit 1; }
  deb="$VENDOR/debs/$(basename "$file")"
  if [ ! -s "$deb" ]; then
    echo ">> downloading $pkg"
    curl -fsSL -o "$deb.part" "$REPO/$file" && mv "$deb.part" "$deb"
  fi
  dpkg -x "$deb" "$VENDOR/extract"
done

echo ">> overlaying the extracted toolchain on a symlink copy of $HOST_ROCM"
rm -rf "$VENDOR/prefix"
mkdir -p "$VENDOR/prefix"
cp -rs "$HOST_ROCM"/. "$VENDOR/prefix"/
cp -rsf "$VENDOR/extract/opt/rocm-$ROCM_VERSION"/. "$VENDOR/prefix"/

export ROCM_PATH="$VENDOR/prefix" HIP_PATH="$VENDOR/prefix"
"$ROCM_PATH/bin/hipcc" --version | head -2 \
  || { echo "FATAL: hipcc from $ROCM_PATH does not run." >&2; exit 1; }

if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
  echo ">> /dev/kfd is accessible"
else
  echo "WARNING: /dev/kfd is not read/write for this user; add it to 'render' and 'video'." >&2
fi
echo ">> ROCm toolchain ready at $ROCM_PATH"
```

- [ ] **Step 4: Remove the CUDA setup**

```bash
git rm -q scripts/00_setup_cuda.sh
chmod +x scripts/00_setup_rocm.sh
```

- [ ] **Step 5: Run the tests, then the script twice**

```bash
uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -8
bash scripts/00_setup_rocm.sh > /tmp/claude-1000/rocm1.log 2>&1; echo "run1 exit $?"; tail -4 /tmp/claude-1000/rocm1.log
bash scripts/00_setup_rocm.sh > /tmp/claude-1000/rocm2.log 2>&1; echo "run2 exit $?"; grep -c downloading /tmp/claude-1000/rocm2.log
```
Expected: tests pass; run1 exit 0 and the log ends with `>> ROCm toolchain ready` after an `AMD clang version 20` / `HIP version: 7.1.` line (a one-off `rocm_agent_enumerator: not found` message is harmless if hipcc still prints its version); run2 exit 0 with `0` downloads (idempotent). Disk: `du -sh vendor/rocm` is ~2.7GB.

- [ ] **Step 6: Commit**

```bash
git add scripts/00_setup_rocm.sh tests/test_toolchain_config.py
git commit -m "feat: build a project-local HIP toolchain from AMD's 7.1.1 packages

The host has the ROCm runtime but no hipcc or headers and no sudo. Missing
packages are unpacked with dpkg -x into vendor/rocm and overlaid on the host
tree. Replaces the CUDA toolkit installer, which has no remaining use.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: llama.cpp HIP build and serve script

**Files:**
- Modify: `scripts/00_setup_llamacpp.sh`, `scripts/04_serve.sh`, `tests/test_toolchain_config.py` (append)

**Interfaces:**
- Consumes: `vendor/rocm/prefix`, `vendor/rocm/extract` from Task 3; `.venv/bin/cmake`, `.venv/bin/ninja`.
- Produces: `vendor/llama.cpp/build/bin/{llama-server,llama-quantize,llama-perplexity}` that list a ROCm device with `LD_LIBRARY_PATH` unset.

- [ ] **Step 1: Append failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -8`
Expected: the two new tests FAIL.

- [ ] **Step 3: Edit `scripts/00_setup_llamacpp.sh`**

Replace the header comment block with:

```bash
# Stage 0 -- vendor and build llama.cpp with HIP/ROCm for the RX 7600 (gfx1102).
#
# Pinned to a specific commit (ADR 0011): a converter change must never silently alter
# results between runs. AMDGPU_TARGETS=gfx1102 compiles for this GPU alone, which cuts
# build time substantially. The compiler comes from vendor/rocm (ADR 0016).
```

After the `AMDGPU_TARGET=` line add:

```bash
ROCM_VENDOR_DIR="$(cd "${ROCM_VENDOR_DIR:-vendor/rocm}" 2>/dev/null && pwd)" \
  || { echo "FATAL: vendor/rocm missing. Run: bash scripts/00_setup_rocm.sh" >&2; exit 1; }
export ROCM_PATH="$ROCM_VENDOR_DIR/prefix" HIP_PATH="$ROCM_VENDOR_DIR/prefix"
[ -x "$ROCM_PATH/bin/hipcc" ] || { echo "FATAL: $ROCM_PATH/bin/hipcc missing. Run: bash scripts/00_setup_rocm.sh" >&2; exit 1; }
export PATH="$(pwd)/.venv/bin:$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
unset LD_LIBRARY_PATH
```

Replace the whole `# Configuração para compilação HIP/ROCm` block (the `echo ">> configuring"` and the `cmake -S ...` call) with:

```bash
echo ">> configuring (HIP/ROCm, target: $AMDGPU_TARGET)"
"$CMAKE" -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGET" \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
```

Keep the existing build and binary-existence loop. Before the final `echo ">> llama.cpp ready (ROCm/HIP)"` insert:

```bash
# The binaries must find the GPU with no environment help (rpath only).
devices="$("$LLAMA_DIR/build/bin/llama-server" --list-devices 2>&1 || true)"
echo "$devices" | grep -q "ROCm0" || {
  echo "FATAL: llama-server lists no ROCm device with LD_LIBRARY_PATH unset:" >&2
  echo "$devices" >&2; exit 1; }
echo ">> $(echo "$devices" | grep ROCm0)"
```

- [ ] **Step 4: Edit `scripts/04_serve.sh`**

Replace the header comment's second line with `# Only ONE model at a time: two Q4_K_M 8B models (~10GB) exceed the 8GB card (ADR 0010).` and add `unset LD_LIBRARY_PATH` on the line before `exec`.

- [ ] **Step 5: Run tests and the build**

```bash
uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -8
export LLAMA_COMMIT=bdeb855b30dfe7f6e695cba98445a7ba09e6416e
bash scripts/00_setup_llamacpp.sh > /tmp/claude-1000/llama.log 2>&1; echo "exit $?"; tail -5 /tmp/claude-1000/llama.log
```
Expected: tests pass; build exit 0 in roughly 5 minutes; the log ends with `>> ROCm0: AMD Radeon RX 7600 ...` then `>> llama.cpp ready (ROCm/HIP)`. Re-running the script must also exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/00_setup_llamacpp.sh scripts/04_serve.sh tests/test_toolchain_config.py
git commit -m "fix: build llama.cpp with GGML_HIP against the local toolchain

GGML_HIPBLAS is no longer accepted by the pinned commit. The build now asserts
that the binaries find the GPU with LD_LIBRARY_PATH cleared.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Smoke run and measurements

**Files:** none committed; measurements are copied into ADR 0015 in Task 6. If a run proves an env var or setting is needed, modify `Makefile` / `scripts/02_train.py` with the reason in a comment, and extend the guard test that forbids it.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: the numbers Task 6 records: peak VRAM, step time, base-GGUF serve VRAM, whether `expandable_segments` works, whether any HSA variable was needed.

- [ ] **Step 1: Full test suite**

Run: `uv run --extra dev pytest -q 2>&1 | tail -5`
Expected: all pass (prompt-parity tests need no GPU; tests marked `slow` that download the tokenizer may need network).

- [ ] **Step 2: Smoke data and training**

`make smoke` regenerates `data/` with a 200-row slice (the full split can be restored later with `make data`).

```bash
make smoke > /tmp/claude-1000/smoke-data.log 2>&1; echo "data exit $?"
(time make train > /tmp/claude-1000/smoke-train.log 2>&1; echo "train exit $?") 2>&1 | tail -4
grep -E "peak VRAM|finished in|s/it|expandable" /tmp/claude-1000/smoke-train.log | tail -5
```
Expected: `train exit 0`; line `>> finished in ...h, peak VRAM X.XX GB`; no segfault.
Record: **peak VRAM**, and seconds per step (from the progress bar or elapsed/steps).
Compare against ADR 0014 (7.06-7.10GB of 7.65GB). If it OOMs, stop and report; the fallback is a design change (new ADR), not an ad hoc config edit.

If `expandable_segments` prints a warning that it is unsupported, remove that `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)` line and its comment from `scripts/02_train.py`, rerun this step, and record it.

- [ ] **Step 3: Export, serve and evaluate**

```bash
make export > /tmp/claude-1000/smoke-export.log 2>&1; echo "export exit $?"
make eval > /tmp/claude-1000/smoke-eval.log 2>&1; echo "eval exit $?"
make report > /tmp/claude-1000/smoke-report.log 2>&1; echo "report exit $?"
ls -la outputs/gguf/*.gguf
```
Expected: three `exit 0`; two Q4_K_M GGUF files. While `make eval` serves a model, sample VRAM in another shell once: `cat /sys/class/drm/card*/device/mem_info_vram_used` (bytes) and record the peak serving figure for the 8B Q4_K_M model. Check there is no `<think>` in generations and format-validity is reported (invariant 2).

- [ ] **Step 4: Write the measurements down**

Create `/tmp/claude-1000/measurements.txt` with: peak training VRAM, step time, serving VRAM, `expandable_segments` (works / removed), HSA variables (none needed / which). These are inputs to Task 6; nothing is committed here except any fix made in Step 2.

- [ ] **Step 5: Commit only if a code or config fix was needed**

```bash
git add scripts/02_train.py Makefile tests/test_toolchain_config.py
git commit -m "fix: <what the smoke run proved was needed>

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: ADRs 0015 and 0016

**Files:**
- Create: `docs/adr/0015-target-hardware-rx-7600-rocm.md`, `docs/adr/0016-llamacpp-hip-local-toolchain.md`
- Modify: `docs/adr/README.md`

**Interfaces:**
- Consumes: the measurements from Task 5. Where the text below says `<MEASURED: ...>`, replace it with the recorded figure before committing; no `<MEASURED` string may survive.

- [ ] **Step 1: Create ADR 0015**

```markdown
# ADR 0015 — Target hardware: RX 7600 8GB on ROCm 7.1

- **Status:** Accepted
- **Date:** 2026-10-05
- **Supersedes:** [ADR 0013](0013-target-hardware-rtx-3070-ti.md) (hardware and CUDA
  toolchain only; its finding that 8GB constrains training is unchanged and was
  resolved by [ADR 0014](0014-training-qwen3-8b-on-8gb.md))
- **Amends:** [ADR 0011](0011-python-env-and-pinning.md) — the torch index and the
  "driver is backward compatible" argument.

## Context

The fork moves to an AMD machine:

| | |
|---|---|
| GPU | Radeon RX 7600, gfx1102 (RDNA3), 8176 MiB |
| CPU / RAM | Ryzen 5 5600X |
| Kernel / userspace | 6.14 inbox `amdgpu`; ROCm 7.1.1 packages in `/opt/rocm-7.1.1` |
| Missing on host | `hipcc`, HIP headers, `rocm-llvm`; no passwordless sudo |

The first migration attempt locked `torch 2.14.1+rocm7.14`. `make train` died with exit
139: a general protection fault in `libamdhip64.so.7` at the first kernel launch, with or
without `HSA_OVERRIDE_GFX_VERSION`. `torch.cuda.is_available()` stayed `True`, so the old
setup check passed. Preloading the host's 7.1.1 `libamdhip64` made the same script run.
`amd-smi` reporting "ROCm 7.14.1" came from the `rocm-sdk-core` pip package in the venv,
not from the host. The venv also held stale `pytorch-triton-rocm` and `triton-rocm`
beside the locked `triton`.

Validated on this host: `torch 2.13.0+rocm7.1` runs kernels and bf16 matmul with no gfx
override; `bitsandbytes 0.50.2` quantises to 4-bit on the GPU; Unsloth loads
`unsloth/Qwen3-8B-bnb-4bit` with 5.74GB reserved.

## Decision

1. The fork targets the RX 7600 8GB. Hardware figures in `CLAUDE.md` and the README describe
   this card; earlier ADRs, `docs/RESULTS.md` and `docs/GPU-COMPARISON.md` stay as the NVIDIA
   record.
2. `torch==2.13.0` and `torchvision` come from the `rocm7.1` index. **The wheel's bundled HIP
   runtime must match the host ROCm.** CUDA tolerates a newer toolkit than the driver;
   ROCm wheels on this host do not.
3. No ROCm runtime is installed through pip, and `HSA_OVERRIDE_GFX_VERSION` is not set: the
   wheel lists gfx1102 natively. <MEASURED: HSA variables — "none needed" or the variable
   and the reason>.
4. `make setup` verifies the GPU with a real bf16 matmul (`scripts/00_check_gpu.py`) and warns
   when the wheel's HIP major.minor matches no `/opt/rocm-*` tree.
5. GPU recipes clear `LD_LIBRARY_PATH`, which the user's shell sets to non-existent ROCm
   trees.
6. ADR 0014's settings (`train_id`, batch 1 × 16, `max_seq_length` 1536,
   `eval_strategy: "no"`) are unchanged. <MEASURED: `expandable_segments` — "honoured" or
   "removed because unsupported">.

## Consequences

- Smoke run on this card: training peak <MEASURED: X.XX GB of 7.98 GB>, <MEASURED: N s> per
  optimizer step at effective batch 16, serving the Q4_K_M 8B model peaks at <MEASURED: X.X GB>.
  ADR 0014 measured 7.06-7.10GB and ~25 s/step on the RTX 3070 Ti.
- A host ROCm upgrade means moving the torch index to the matching release; the check in
  `00_check_gpu.py` warns first.
- Results are not strictly comparable to the NVIDIA runs: different GPU, kernels and bf16
  numerics, on top of the differences listed in ADR 0014.
- Invariants 1-7 are unaffected; no code in `src/personas/` changed.

## Alternatives considered

- **Keep the `rocm7.14` wheel and upgrade the host ROCm.** Needs root and a full stack
  reinstall; the host has no passwordless sudo.
- **Preload the host `libamdhip64` over the 7.14 wheel.** Works in a probe, but mixes two
  runtimes in one process; a fragile workaround for a version mismatch.
- **`HSA_OVERRIDE_GFX_VERSION=11.0.0`.** Unneeded here (gfx1102 is in the wheel's arch list)
  and it hides real architecture issues.
```

- [ ] **Step 2: Create ADR 0016**

```markdown
# ADR 0016 — llama.cpp built with HIP from a project-local toolchain

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

Building llama.cpp for the GPU needs `hipcc`, the HIP headers, `rocm-llvm`, `hipblas` and
`rocsolver`. The host has the ROCm 7.1.1 runtime but none of these, and no passwordless
sudo. The first migration script only checked for `hipcc` on `PATH`, so it could not
succeed, and passed `-DGGML_HIPBLAS=ON`, which the pinned commit no longer uses.

The pip `rocm-sdk-core 7.14` ships a `hipcc`, but a binary built against it links the 7.14
runtime that segfaults on this host (ADR 0015).

Prototype on this host (2026-10-05): the 7.1.1 `.deb` packages from
`https://repo.radeon.com/rocm/apt/7.1.1` unpacked with `dpkg -x` and overlaid on a
symlink copy of `/opt/rocm-7.1.1` built llama.cpp `bdeb855` for gfx1102 in 4m32s;
`llama-server --list-devices` reported `ROCm0: AMD Radeon RX 7600 (8176 MiB)` with
`LD_LIBRARY_PATH` unset, and `/completion` generated with `-ngl 99 -fa on`.

## Decision

1. `scripts/00_setup_rocm.sh` downloads the missing packages for the host's ROCm version
   (names resolved from the repository's `Packages.gz`), extracts them under `vendor/rocm`,
   and builds `vendor/rocm/prefix`: a symlink tree of the host install with the extracted
   files overlaid. It is idempotent and requires no root.
2. `scripts/00_setup_llamacpp.sh` builds with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1102`,
   `CMAKE_HIP_COMPILER` and `CMAKE_PREFIX_PATH` pointing at the prefix, using Ninja. The
   pinned commit is unchanged. It then fails the build unless `llama-server --list-devices`
   shows a ROCm device with `LD_LIBRARY_PATH` unset.
3. `scripts/00_setup_cuda.sh` is removed.

## Consequences

- ~2.7GB under `vendor/rocm` (not tracked by git), including ~540MB of downloads.
- The toolchain version follows the host ROCm install; a host upgrade changes what is
  downloaded on the next `make setup`.
- Needs network access to `repo.radeon.com` the first time; the script stops with the URL
  when it is unreachable.
- Invariant 5 holds: base and tuned GGUFs go through the same binaries.

## Alternatives considered

- **Vulkan backend (`-DGGML_VULKAN=ON`).** Needs no ROCm, but `glslc` is also absent and the
  backend is slower on RDNA3. Not needed once HIP built.
- **The pip `hipcc` 7.14.** Links the runtime that crashes here.
- **Installing `hip-dev` with apt.** Needs sudo.
```

- [ ] **Step 3: Substitute the measured values**

Replace each `<MEASURED: ...>` in ADR 0015 with the Task 5 figure.

Run: `grep -rn "MEASURED\|TBD\|TODO" docs/adr/0015-*.md docs/adr/0016-*.md`
Expected: no output.

- [ ] **Step 4: Update `docs/adr/README.md`**

Change the 0013 row status to `Superseded by 0015` and add after the 0014 row:

```markdown
| [0015](0015-target-hardware-rx-7600-rocm.md) | Target hardware: RX 7600 8GB on ROCm 7.1; torch wheel must match host ROCm | Accepted |
| [0016](0016-llamacpp-hip-local-toolchain.md) | llama.cpp built with HIP from a project-local toolchain | Accepted |
```

- [ ] **Step 5: Commit**

```bash
git add docs/adr
git commit -m "docs: record the ROCm migration as ADR 0015 and 0016

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: CLAUDE.md, README, report footer, spec pointer

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `scripts/06_report.py:114`, `docs/superpowers/specs/2026-09-12-persona-finetune-design.md`, `docs/superpowers/specs/2026-10-05-rocm-migration-design.md`, `tests/test_toolchain_config.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def test_no_nvidia_text_left_in_live_files():
    for path in ("scripts/06_report.py", "scripts/04_serve.sh", "Makefile",
                 "scripts/00_setup_llamacpp.sh", "scripts/00_setup_rocm.sh"):
        text = _read(path)
        assert "RTX" not in text and "CUDA" not in text.replace("PYTORCH_CUDA_ALLOC_CONF", ""), path


def test_claude_md_describes_the_current_hardware():
    text = _read("CLAUDE.md")
    assert "RX 7600" in text and "ROCm" in text
    assert "vendor/rocm" in text
    assert "vendor/cuda" not in text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_toolchain_config.py -v 2>&1 | tail -8`
Expected: both FAIL (the report footer says RTX 3070 Ti; CLAUDE.md mentions CUDA).

- [ ] **Step 3: `scripts/06_report.py` footer**

Change line 114 from `... &middot; Q4_K_M on RTX 3070 Ti &middot; identical` to `... &middot; Q4_K_M &middot; identical` (the report names no card; the hardware lives in the ADRs, and a hardcoded card was already wrong once).

- [ ] **Step 4: `CLAUDE.md`**

Make these edits (keep the rest verbatim):
- Intro paragraph: replace "**This fork targets an RTX 3070 Ti (8GB)** — see ADR 0013 and 0014 for what that changes." with "**This fork targets a Radeon RX 7600 (8GB, ROCm)** — see ADR 0014 for the 8GB constraints and ADR 0015/0016 for the ROCm migration."
- Hardware table: GPU row `Radeon RX 7600 **8GB** (gfx1102), ~8.0GB visible`; RAM/disk row stays; the Driver row becomes `| ROCm | host 7.1.1 (\`/opt/rocm-7.1.1\`); torch wheel \`rocm7.1\`; HIP toolchain in \`vendor/rocm\` |`.
- Sentence "Earlier ADRs, the design spec and `docs/RESULTS.md` were written for and measured on the RTX 3060 12GB." gains "(and the RTX 3070 Ti 8GB)".
- VRAM section first line becomes: "Training peaks at **<peak from ADR 0015>GB of ~8.0GB**" using the figure in ADR 0015; its three settings list is unchanged. Serving line uses the ADR 0015 serving figure.
- Replace the bullet starting "**The llama.cpp build needs `nvcc`" with: "**The llama.cpp build needs `hipcc`, which the host does not ship** (runtime only, no sudo). `scripts/00_setup_rocm.sh` extracts AMD's 7.1.1 packages into `vendor/rocm` and overlays them on the host tree; `00_setup_llamacpp.sh` then asserts the binaries find the GPU with `LD_LIBRARY_PATH` unset."
- Add two bullets to the gotchas:
  - "**The torch wheel's HIP runtime must match the host ROCm.** The `rocm7.14` wheel segfaulted (exit 139) at the first kernel on a 7.1.1 host while `torch.cuda.is_available()` was still `True`. `make setup` runs a real matmul (`scripts/00_check_gpu.py`). Never add a `rocm*` pip package (ADR 0015)."
  - "**Clear `LD_LIBRARY_PATH` for GPU runs.** The shell exports non-existent ROCm dirs; the Makefile uses `GPU_ENV = env -u LD_LIBRARY_PATH`."

- [ ] **Step 5: `README.md`**

- Line 5 sentence and the Requirements paragraph (lines ~195-203): say AMD RX 7600 8GB with ROCm 7.1, `~8GB visible`, remove "NVIDIA driver 535+" and the micromamba/CUDA paragraph, replacing it with: "If the host has the ROCm 7.1.1 runtime but no compiler, `make setup` downloads AMD's HIP packages into `vendor/rocm` (~2.7GB, no root), so the only system package needed is a C++ compiler."
- Setup comment `# uv venv (Python 3.12) + CUDA build of llama.cpp` → `# uv venv (Python 3.12) + HIP build of llama.cpp`; stage-table row text `llama.cpp CUDA build` → `llama.cpp HIP build`.
- Leave every RTX 3060 / 3070 Ti result and timing table as is (historical record). Add one sentence under the title block: "This tree now targets an RX 7600 8GB with ROCm ([ADR 0015](docs/adr/0015-target-hardware-rx-7600-rocm.md)); the tables below are the NVIDIA runs."

- [ ] **Step 6: Spec pointers**

Append to the 2026-09-12 design spec, after the section 3 table, one line: `> GPU toolchain superseded for the AMD fork: see [2026-10-05-rocm-migration-design.md](2026-10-05-rocm-migration-design.md).`

In `2026-10-05-rocm-migration-design.md`: change Status to `Approved; implemented by docs/superpowers/plans/2026-10-05-rocm-migration.md` and replace the `scripts/06_report.py` table row's change cell with "report footer no longer names a card".

- [ ] **Step 7: Run all tests and a final grep**

```bash
uv run --extra dev pytest -q 2>&1 | tail -5
grep -rniE "nvcc|vendor/cuda|HSA_OVERRIDE" CLAUDE.md README.md Makefile scripts pyproject.toml
```
Expected: all tests pass; the grep prints nothing (ADR text and historical tables are excluded because they are not in these paths).

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md README.md scripts/06_report.py docs/superpowers tests/test_toolchain_config.py
git commit -m "docs: point CLAUDE.md, README and specs at the ROCm target

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** 3.1 → Task 1; 3.2 → Task 2; 3.3 → Task 3; 3.4 → Task 4; 3.5 → Tasks 2 and 5 (HSA / `expandable_segments` measured); 3.6 file table → Tasks 3, 4, 6, 7 (`00_setup_cuda.sh` removed in Task 3); 3.7 invariants → Global Constraints; section 5 verification → Task 5; risks → Review Focus.
- **Deviation from the spec:** the spec says `06_report.py` reads the GPU name from results. The plan drops the card name from the footer instead (no GPU info flows through `results`, and adding it would touch stage 5 for a cosmetic line). Task 7 Step 6 updates the spec to match.
- **Type/name consistency:** `GPU_ENV`, `ROCM_VENDOR_DIR`, `ROCM_VERSION`, `vendor/rocm/prefix`, `vendor/rocm/extract` are used identically in Tasks 3, 4 and 7.
