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
| GPU | Radeon RX 7600, gfx1102 (RDNA3), 8176 MiB (7.98 GiB) |
| CPU / RAM | Ryzen 5 5600X |
| Kernel / userspace | 6.14 inbox `amdgpu`; ROCm 7.1.1 packages in `/opt/rocm-7.1.1` |
| Missing on host | `hipcc`, HIP headers, `rocm-llvm`; no passwordless sudo |

How this was found, in order:

- The first migration attempt locked `torch 2.14.1+rocm7.14`. `make train` died with exit
  139: a general protection fault in `libamdhip64.so.7` at the first kernel launch, with or
  without `HSA_OVERRIDE_GFX_VERSION`. `torch.cuda.is_available()` stayed `True`, so the old
  setup check passed. Preloading the host's 7.1.1 `libamdhip64` made the same script run.
- `amd-smi` reporting "ROCm 7.14.1" came from the `rocm-sdk-core` pip package in the venv,
  not from the host.
- The old venv also held a stale `pytorch-triton-rocm 3.2.0` (an earlier install, not in the
  lock). `triton-rocm` itself is legitimate: the torch ROCm wheel depends on it (3.6.0 with
  torch 2.11.0, alongside `triton` 3.8.0, both from the `rocm7.1` index).
- Moving to the `rocm7.1` index and pinning `torch==2.13.0` made the resolver backtrack:
  it picked unsloth 2026.3.11 (mismatched with unsloth-zoo 2026.9.9) and transformers
  5.3.0, whose `SFTConfig` rejects `group_by_length`. `make train` then failed with a
  `TypeError`. With `torch==2.11.0` the lock resolves unsloth 2026.9.14, unsloth-zoo
  2026.9.9, transformers 5.5.0, trl 0.24.0 and peft 0.21.2, which matches the NVIDIA-era set
  (torch 2.11.0, transformers 5.5.0, unsloth 2026.9.4, zoo 2026.9.3). Plain `uv lock` keeps
  the previous resolution; `uv lock --upgrade` is required after changing the torch pin.

Validated on this host: the `rocm7.1` torch wheel runs kernels and bf16 matmul with no gfx
override; `bitsandbytes 0.50.2` quantises to 4-bit on the GPU; Unsloth loads
`unsloth/Qwen3-8B-bnb-4bit` with 5.74GB reserved.

## Decision

1. The fork targets the RX 7600 8GB. Hardware figures in `CLAUDE.md` and the README describe
   this card; earlier ADRs, `docs/RESULTS.md` and `docs/GPU-COMPARISON.md` stay as the NVIDIA
   record.
2. `torch==2.11.0` and `torchvision` come from the `rocm7.1` index. **The wheel's bundled HIP
   runtime must match the host ROCm.** CUDA tolerates a newer toolkit than the driver;
   ROCm wheels on this host do not. `unsloth` and `unsloth-zoo` are never pinned
   individually: the resolver picks them as a coupled set, and a test requires both to share
   a year.month of 2026.9 or later.
3. No ROCm runtime is installed through pip, and `HSA_OVERRIDE_GFX_VERSION` is not set: the
   wheel lists gfx1102 natively. No HSA or hipBLASLt override was needed to run kernels,
   quantise or load the model.
4. `make setup` verifies the GPU with a real bf16 matmul (`scripts/00_check_gpu.py`) and warns
   when the wheel's HIP major.minor matches no `/opt/rocm-*` tree.
5. GPU recipes clear `LD_LIBRARY_PATH`, which the user's shell sets to non-existent ROCm
   trees.
6. ADR 0014's settings (`train_id`, batch 1 × 16, `max_seq_length` 1536,
   `eval_strategy: "no"`) are unchanged. `expandable_segments` stays in
   `scripts/02_train.py`: no unsupported-option warning appeared, but its effect is
   unconfirmed because the run died first (see Consequences).

## Consequences

- **Training fit on this card is unverified.** The smoke run starts and reaches the step-1
  backward pass, then fails with `torch.OutOfMemoryError: 7.18 GiB allocated, 94 MiB free,
  tried 210 MiB`. The desktop and browser held ~1.28GB of VRAM before the run, against
  ~0.3-0.4GB on the RTX 3070 Ti where ADR 0014 measured a 7.06-7.10GB peak and ~25 s/step.
  Whether the ADR 0014 configuration fits needs a retry with the GPU free of the desktop
  (TTY, or browser closed). Until then the training peak and s/step on the RX 7600 are not
  measured.
- Serving VRAM for the Q4_K_M 8B model, and the export and eval path, are also not measured:
  disk was at 99% and export needs ~50GB at peak.
- A host ROCm upgrade means moving the torch index to the matching release; the check in
  `00_check_gpu.py` warns first.
- Results are not strictly comparable to the NVIDIA runs: different GPU, kernels and bf16
  numerics, on top of the differences listed in ADR 0014.
- Pre-existing, left unchanged on purpose: on transformers 5.5.0, `group_by_length` is not a
  `TrainingArguments` field. Unsloth's patched `SFTConfig` accepts it but warns that it is
  ignored (the replacement is `train_sampling_strategy="group_by_length"`). The NVIDIA runs
  used the same transformers 5.5.0, so their timings were measured with length grouping
  effectively off.
- Warnings seen on ROCm, harmless so far: "Attempting to use CK GEMM on an unsupported
  architecture", Unsloth `import_fixes` deprecation warnings, and FlashAttention-2 being
  broken so xformers is used instead.
- Invariants 1-7 are unaffected; no code in `src/personas/` changed.

## Alternatives considered

- **Keep the `rocm7.14` wheel and upgrade the host ROCm.** Needs root and a full stack
  reinstall; the host has no passwordless sudo.
- **Preload the host `libamdhip64` over the 7.14 wheel.** Works in a probe, but mixes two
  runtimes in one process; a fragile workaround for a version mismatch.
- **`HSA_OVERRIDE_GFX_VERSION=11.0.0`.** Unneeded here (gfx1102 is in the wheel's arch list)
  and it hides real architecture issues.
- **`torch==2.13.0` from the `rocm7.1` index.** Resolves, but drags unsloth and transformers
  back to a mismatched, older set (see Context).
