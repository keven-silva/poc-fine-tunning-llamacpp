# Design — Migration from NVIDIA CUDA to AMD ROCm (RX 7600 8GB)

- **Date:** 2026-10-05
- **Status:** Draft, awaiting review
- **Extends:** [2026-09-12-persona-finetune-design.md](2026-09-12-persona-finetune-design.md)
  (sections 3 and 4 only; the task, data, prompt and evaluation design are unchanged)

## 1. Goal

Run the whole pipeline (`make setup → data → train → export → serve → eval → report`) on a
**Radeon RX 7600 (gfx1102, RDNA3, 8GB)** with ROCm, keeping every invariant in `CLAUDE.md`.
Same model (Qwen3-8B QLoRA), same prompt renderer, same evaluation. Only the GPU toolchain
changes.

**Success criteria**

1. `make setup` ends with a *real* GPU kernel check, not only `torch.cuda.is_available()`.
2. `make smoke` completes: train 10 steps, export, serve, eval, on the RX 7600.
3. The measured peak VRAM of training is recorded in an ADR (not estimated).
4. `make train` no longer segfaults; the llama.cpp binaries run without `LD_LIBRARY_PATH`.
5. Docs follow the existing conventions: new ADRs instead of rewriting accepted ones;
   `docs/RESULTS.md` and earlier ADR numbers stay as the historical record.

**Non-goals:** changing the base model, retraining for quality comparison with the 3060 /
3070 Ti runs, Vulkan or CPU-only support, multi-GPU.

## 2. Findings (measured 2026-10-05)

| Host | Value |
|---|---|
| GPU | Radeon RX 7600, gfx1102, 8176 MiB |
| CPU / RAM | Ryzen 5 5600X |
| Kernel / userspace | 6.14 (inbox `amdgpu`), ROCm **7.1.1** packages in `/opt/rocm-7.1.1` |
| Missing on host | `hipcc`, HIP headers, `rocm-llvm`, `hipblas`, `rocsolver`; no `/opt/rocm` link; no passwordless sudo |
| Shell | exports `LD_LIBRARY_PATH` and `PATH` entries for `/opt/rocm-6.4.2` and `/opt/rocm`, which do not exist |

**Why `make train` fails (exit 139).** `uv.lock` resolved `torch 2.14.1+rocm7.14`. Its HIP
7.14 runtime raises a general protection fault in `libamdhip64.so.7` at the first kernel
launch against this host's driver stack, with or without `HSA_OVERRIDE_GFX_VERSION`.
`torch.cuda.is_available()` still returns `True`, so the old `make setup` check passed.
Preloading the host's 7.1.1 `libamdhip64` makes the same script work. `amd-smi` reporting
"ROCm 7.14.1" is only the pip `rocm-sdk-core` package inside the venv.

**Other conflicts found**

- The venv carried stale `pytorch-triton-rocm 3.2.0` and `triton-rocm 3.8.0` beside the locked
  `triton 3.8.0`, all writing the same `triton/` package directory.
- `scripts/00_setup_rocm.sh` aborts (no `hipcc`), and `00_setup_llamacpp.sh` passes
  `-DGGML_HIPBLAS=ON`, a flag the pinned llama.cpp commit no longer uses (`-DGGML_HIP=ON`).
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set twice in the Makefile. The torch wheel lists
  gfx1102 natively, so it is not needed.

**Validated in a throwaway venv (outside the project):**

| Check | Result |
|---|---|
| `torch 2.13.0+rocm7.1`, matmul bf16, no override | works |
| `bitsandbytes 0.50.2` 4-bit quantise on GPU | works |
| `unsloth 2026.3.11` + `unsloth-zoo 2026.9.9`, `triton 3.8.0`, transformers 5.3.0, trl 0.24.0, peft 0.21.2 | resolve and import |
| `unsloth/Qwen3-8B-bnb-4bit`, `use_exact_model_name=True`, `max_seq_length=1536` | loads, **5.74GB reserved** (3070 Ti: 5.73GB) |
| llama.cpp `bdeb855`, HIP 7.1.1 toolchain extracted from AMD `.deb`s, gfx1102 | builds in 4m32s |
| `llama-server --list-devices` without `LD_LIBRARY_PATH` | `ROCm0: AMD Radeon RX 7600 (8176 MiB)` |
| `/completion` with `-ngl 99 -fa on` (Qwen3-0.6B Q4_K_M) | generates |

Not yet validated: a full QLoRA training step and Qwen3-8B serving on this card. Those are
the `make smoke` acceptance steps.

## 3. Design

### 3.1 Python environment (supersedes the CUDA parts of ADR 0011)

- `pyproject.toml`: index `pytorch-rocm` → `https://download.pytorch.org/whl/rocm7.1`;
  `torch==2.13.0` and `torchvision` both sourced from it. The comment states the rule:
  **the wheel's HIP runtime must match the host ROCm**, which is the opposite of the CUDA
  backward-compatibility argument in ADR 0011.
- Do not add `rocm-sdk-core` or any `rocm*` pip package. The host runtime is the one in use.
- `unsloth` and `unsloth-zoo` stay unpinned and are resolved as a coupled set, per `CLAUDE.md`.
- Recreate the venv from scratch (`uv sync --reinstall`), then relock. The stale triton
  packages must not survive.
- `torchvision` stays a direct dependency so `[tool.uv.sources]` applies.

### 3.2 `make setup`

Order: `uv sync` → GPU sanity → `00_setup_rocm.sh` → `00_setup_llamacpp.sh`.

GPU sanity runs an actual kernel (small bf16 matmul plus `.item()`), fails with a message
naming the likely cause (wheel/host ROCm mismatch), and is also what `make smoke` relies on.
It replaces the `is_available()` assertion.

### 3.3 HIP toolchain without root (`scripts/00_setup_rocm.sh`)

Replaces the `hipcc`-on-PATH check, which cannot succeed on this host. Same role as the old
`00_setup_cuda.sh`: a project-local, removable toolchain in `vendor/rocm`.

1. Read the host ROCm version from `/opt/rocm-*` (default 7.1.1) and the Ubuntu series
   (jammy `.deb`s from `https://repo.radeon.com/rocm/apt/<version>`).
2. Download only the packages the host lacks: `hip-dev`, `hipcc`, `rocm-llvm`,
   `rocm-device-libs`, `hipblas`, `hipblas-dev`, `rocsolver`, `hsa-rocr-dev`, `rocm-cmake`,
   `rocm-core`. Resolve file names from the repo's `Packages.gz` rather than hardcoding
   versions.
3. `dpkg -x` them into `vendor/rocm/extract`.
4. Build `vendor/rocm/prefix`, a symlink tree: first `cp -rs /opt/rocm-<ver>/.`, then overlay
   the extracted files. This is the `ROCM_PATH` for the build.
5. Verify `hipcc --version`, and `/dev/kfd` read/write access as today.

Cost: ~2.1GB extracted plus ~540MB of downloads, untracked by git (`vendor/` is ignored).
The script is idempotent and skips packages already extracted.

### 3.4 llama.cpp build (`scripts/00_setup_llamacpp.sh`)

- `-DGGML_HIP=ON` (not `GGML_HIPBLAS`), `-DAMDGPU_TARGETS=gfx1102`,
  `-DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++`, `-DCMAKE_PREFIX_PATH=$ROCM_PATH`,
  `-G Ninja` (ninja already comes from the `dev` extra).
- Build with `ROCM_PATH`/`HIP_PATH` exported to `vendor/rocm/prefix` and `LD_LIBRARY_PATH`
  unset.
- Bake the rpath so binaries run with no environment: the build already records the extracted
  lib dir and `/opt/rocm-<ver>/lib`; the script then asserts it by running
  `llama-server --list-devices` with `LD_LIBRARY_PATH` cleared and failing if no ROCm device
  is listed.
- The pinned commit (`llamacpp.commit` in the config) is unchanged. Invariant 5 is preserved
  because base and tuned both go through the same binaries.

### 3.5 Makefile and runtime environment

- Remove `HSA_OVERRIDE_GFX_VERSION`, `HIPBLASLT_ENABLE_CK` and `PYTORCH_TUNABLEOP_ENABLED`
  from `train`. Reintroduce one only if `make smoke` shows a concrete need, and document
  that need next to it.
- Recipes that run GPU code clear `LD_LIBRARY_PATH` (`env -u LD_LIBRARY_PATH`), because the
  user shell exports non-existent ROCm dirs and the wheel must resolve its own libraries.
  Applies to `train`, `export` and `serve`/`eval`.
- `scripts/02_train.py` keeps `TORCHINDUCTOR_COMPILE_THREADS=4` above the torch import and
  `PYTORCH_CUDA_ALLOC_CONF`. Whether `expandable_segments` is honoured on ROCm is checked
  during the smoke run; if not, the setting is dropped (and noted in ADR 0015) rather than
  left as a silent no-op.

### 3.6 Scripts and docs touched

| File | Change |
|---|---|
| `pyproject.toml`, `uv.lock` | rocm7.1 index, torch 2.13.0, relock |
| `Makefile` | per 3.2 and 3.5 |
| `scripts/00_setup_rocm.sh`, `00_setup_llamacpp.sh` | per 3.3 and 3.4 |
| `scripts/00_setup_cuda.sh` | removed (no CUDA path remains; recoverable from git) |
| `scripts/04_serve.sh` | VRAM comment ("~9.5GB free" → measured figure for this card) |
| `scripts/06_report.py` | report footer "RTX 3070 Ti" → GPU name read from the results, no hardcoded card |
| `docs/adr/0015-target-hardware-rx-7600-rocm.md` | new: hardware, ROCm 7.1 wheel rule, measured VRAM/time; amends 0013 |
| `docs/adr/0016-llamacpp-hip-local-toolchain.md` | new: toolchain in `vendor/rocm`; Vulkan and ROCm 7.14 rejected, with measurements |
| `docs/adr/README.md` | index rows for 0015, 0016; 0013 status "Superseded by 0015" |
| `CLAUDE.md` | hardware table, VRAM section, ROCm gotchas, `vendor/rocm` instead of `vendor/cuda` |
| `README.md` | hardware and setup text; the 3060 / 3070 Ti result tables stay untouched |
| design spec of 2026-09-12 | not rewritten; section 3 gets a one-line pointer to this spec |

ADR status rule: accepted ADRs are immutable, so ADR 0011 and 0013 are only referenced and
marked by the status column, not edited in their bodies.

### 3.7 Invariants check

| Invariant | Effect |
|---|---|
| 1 One prompt renderer, 2 thinking off, 3 raw `/completion`, 6 `PROMPT_VERSION` | untouched; no Python in `src/` changes |
| 4 Never concurrent | holds; same 8GB class |
| 5 Same quantisation lineage | holds; same llama.cpp commit and steps |
| 7 LangChain only in stages 5-6 | untouched |
| VRAM settings (`train_id`, batch 1 × 16, 1536, `eval_strategy: "no"`) | kept as is; ADR 0015 records whether the RX 7600 peak, which can differ from the 3070 Ti's 7.1GB, leaves the same margin |

## 4. Risks

| Risk | Mitigation |
|---|---|
| bitsandbytes/unsloth kernels misbehave in a real training step (only import, load and quantise were validated) | `make smoke` is the acceptance gate before any multi-hour run |
| RX 7600 training peak exceeds the card's usable VRAM | the config knobs of ADR 0014 are already at their floor; the fallback is a new ADR with a smaller base model, as today |
| AMD repo layout or package names change | resolve names from `Packages.gz`, fail loudly with the URL; versions follow the host's ROCm |
| Host ROCm is upgraded later | `00_setup_rocm.sh` keys off the host version and `pyproject.toml` comment names the matching rule |
| ~2.1GB toolchain on a disk at 95% | documented; `vendor/rocm/extract` is removable after the build, the build keeps working through the prefix only if the libs it links stay, so the script keeps them |
| `vendor/`, `.venv/`, `outputs/` were deleted from the working tree on 2026-10-05 | all are regenerated by `make setup`, `make data`, `make train`, `make export` |

## 5. Verification plan

1. `uv sync` from the new lock; assert installed `torch` is `2.13.0+rocm7.1`, and that
   `pytorch-triton-rocm` and `triton-rocm` are absent.
2. `make setup` end to end on a clean tree.
3. `uv run pytest` (prompt-parity tests do not need the GPU).
4. `make smoke`: 200-row data, training steps, export of both models, eval, report. Record
   peak VRAM and step time.
5. Only then is `make train` (full run) recommended, and it is out of scope for this work.

## 6. Open questions

None blocking. To be resolved by measurement during implementation and written into
ADR 0015: whether any HSA/hipBLASLt environment variable is needed, whether
`expandable_segments` applies on ROCm, and the real step time on this card.
