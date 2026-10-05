# poc-fine-tunning-llamacpp

Fine-tune **Qwen3-8B** with **Unsloth QLoRA** on NVIDIA's synthetic Brazilian persona
corpus, export to **GGUF**, serve and evaluate with **llama.cpp** — originally on a single
RTX 3060 (12GB). **This fork targets a Radeon RX 7600 (8GB, ROCm)** — see ADR 0014 for the 8GB
constraints and ADR 0015/0016 for the ROCm migration.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona narrative out.

## Read first

- [`docs/superpowers/specs/2026-09-12-persona-finetune-design.md`](docs/superpowers/specs/2026-09-12-persona-finetune-design.md) — the design
- [`docs/adr/README.md`](docs/adr/README.md) — why each choice was made

## Hardware and environment

| | |
|---|---|
| GPU | Radeon RX 7600 **8GB** (gfx1102), ~8.0GB visible |
| RAM / disk | 31GB / 259GB free |
| ROCm | host 7.1.1 (`/opt/rocm-7.1.1`); torch 2.11.0 wheel from the `rocm7.1` index; HIP toolchain in `vendor/rocm` (~4GB) |
| Python | always use the project `uv` venv on 3.12, never the host Python |

Earlier ADRs, the design spec and `docs/RESULTS.md` were written for and measured on the
RTX 3060 12GB (and the RTX 3070 Ti 8GB). They are the historical record; do not rewrite their numbers.

Never `pip install` into the host Python. Use `uv run` / `uv sync`; dependencies are
pinned in the committed `uv.lock` (ADR 0011).

## Pipeline

```
make setup → make data → make train → make export → make serve → make eval → make report
```

`make smoke` runs the whole chain on a 200-row slice in minutes. Run it before any
multi-hour training run.

## Invariants — breaking these silently invalidates results

1. **One prompt renderer.** Every prompt comes from `src/personas/prompt.py`. Never build a
   prompt string anywhere else. `tests/test_prompt_parity.py` enforces this (ADR 0006).
2. **Thinking mode off everywhere.** Always `apply_chat_template(..., enable_thinking=False)`.
   Any `<think>` in a generation is a bug and is scored as a format failure.
3. **Raw `/completion` only.** Never evaluate through `/v1/chat/completions` — llama-server
   re-applies its own Jinja template and breaks train/inference parity (ADR 0006, 0007).
4. **Base and tuned never run concurrently.** Two Q4_K_M 8B models (~10GB) do not fit in 8GB.
   Eval is sequential; the Makefile owns server lifecycle (ADR 0010).
5. **Base and tuned share a quantisation lineage.** Both go through the same merge →
   convert → quantise steps so the A/B measures the LoRA and nothing else (ADR 0008).
6. **Changing the system prompt invalidates the adapter.** Bump `PROMPT_VERSION` in
   `prompt.py`; eval checks it against the dataset manifest.
7. **LangChain lives only in stages 5-6.** It must never become a training dependency —
   its release cadence cannot be allowed to break a 4-hour GPU run (ADR 0007).

## VRAM

Training peak and s/step on the RX 7600 are **not measured**. ADR 0014 measured a
7.06-7.10GB peak on the 3070 Ti (7.65GB visible, desktop ~0.3-0.4GB); the three settings
below are what got it there, and they are unchanged. Do not undo any of them on this card.
The smoke run on the RX 7600 (7.98 GiB total) started and then hit `OutOfMemoryError` at
the step-1 backward pass (7.18 GiB allocated, 94 MiB free) with the desktop and browser
holding ~1.28GB of VRAM (ADR 0015). Train from a TTY or with the browser closed, and
record the measured peak in a follow-up ADR.

- `model.train_id: unsloth/Qwen3-8B-bnb-4bit` — the standard 4-bit checkpoint (5.66GB).
  `unsloth/Qwen3-8B` resolves to the dynamic one (6.97GB), which fails at load.
- batch 1 × grad-accum 16, `max_seq_length` 1536.
- `train.eval_strategy: "no"` — an in-training eval pass materialises full fp32 logits and
  OOMs.

There is no further fallback short of a smaller base model.

Serving VRAM for Q4_K_M 8B on this card is also not measured (~5.5GB on the 3070 Ti, ADR
0013/0014); only a small-model generation check was run (ADR 0016).

## Operational gotchas found the hard way

- **System RAM, not VRAM, killed the first training run.** torch inductor spawns one
  compile worker per CPU core (16 here) when kernels are JIT-compiled at the first step.
  `scripts/02_train.py` caps `TORCHINDUCTOR_COMPILE_THREADS=4` before importing torch.
  Keep that cap, and keep it above the torch import.
- **Never pin `unsloth` without pinning `unsloth-zoo`.** They move together. Pinning one
  resolved an 8-month-old Unsloth against transformers 5.17, whose zoo needed a torch
  symbol that did not exist yet. Let the resolver pick the coupled set as a unit.
- **`torchvision` must come from the same index as `torch`.** transformers imports it, and
  a build against a different CUDA major aborts the import. It is declared as a direct
  dependency purely so `[tool.uv.sources]` applies — source mappings do not reach
  transitive dependencies.
- **`datasets` streaming aborts the process at interpreter shutdown**, turning a
  successful run into exit 134. `scripts/01_prepare_data.py` exits via `os._exit` after
  flushing.
- **The llama.cpp build needs `hipcc`, which the host does not ship** (runtime only, no
  sudo). `scripts/00_setup_rocm.sh` extracts AMD's 7.1.1 packages into `vendor/rocm` (~4GB)
  and overlays them on the host tree; `00_setup_llamacpp.sh` then asserts the binaries find
  the GPU with `LD_LIBRARY_PATH` unset.
- **The torch wheel's HIP runtime must match the host ROCm.** The `rocm7.14` wheel
  segfaulted (exit 139) at the first kernel on a 7.1.1 host while
  `torch.cuda.is_available()` was still `True`. `make setup` runs a real matmul
  (`scripts/00_check_gpu.py`). Never add a `rocm*` pip package (ADR 0015).
- **Pinning torch backtracked unsloth and transformers.** `torch==2.13.0` resolved an old
  unsloth beside a newer zoo and transformers 5.3.0, which broke `make train`. The pin is
  `torch==2.11.0` (`rocm7.1` index); after changing it run `uv lock --upgrade`, since plain
  `uv lock` keeps the previous resolution. A test requires unsloth and unsloth-zoo to share
  a 2026.9+ release.
- **Clear `LD_LIBRARY_PATH` for GPU runs.** The shell exports non-existent ROCm dirs; the
  Makefile uses `GPU_ENV = env -u LD_LIBRARY_PATH`.
- **triton needs `Python.h` at the first training step.** A distro Python without its
  `-dev` package fails there with a gcc error, minutes in. `pyproject.toml` sets
  `python-preference = "only-managed"` so the venv uses uv's CPython, which ships headers.
- **Never pipe a long-running script through `tail`** to inspect it — the pipeline's exit
  code is `tail`'s, so failures report as success. Redirect to a file instead.

## Data note

The dataset is CC-BY-4.0 and **fully synthetic** — no real PII, despite being "person
data". Generated samples can be shared freely.
