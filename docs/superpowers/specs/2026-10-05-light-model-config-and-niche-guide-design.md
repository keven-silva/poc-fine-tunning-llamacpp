# Design — Light-model config for learning, and a guide to new niches

- **Date:** 2026-10-05
- **Status:** Draft, awaiting review
- **Extends:** [2026-09-12-persona-finetune-design.md](2026-09-12-persona-finetune-design.md) and
  [2026-10-05-rocm-migration-design.md](2026-10-05-rocm-migration-design.md). The task, prompt,
  data source and evaluation are unchanged.

## 1. Goal

The owner wants to **understand how fine-tuning works end to end** on a model small enough to
train in minutes, and to know **how the same pipeline would be adapted to other niches**
(cyber security, politics, and so on).

Deliverables:

1. `configs/qwen3-0.6b-personas.yaml`: a second config, same persona task, Qwen3-0.6B.
2. `docs/guides/adapting-to-a-new-niche.md`: a Portuguese guide, written for someone who
   wants to learn, explaining each stage and what to replace for a new niche.
3. `docs/adr/0017-light-model-config-for-learning.md`: the decision record.
   The guide also explains, as an optional section, how to use Weights & Biases to track and
   compare runs and evaluations (explained, not integrated: no dependency or pipeline change).
4. A guard test so a second config can never overwrite the first one's data or outputs.
5. One small code change: the report title stops hardcoding "Qwen3-8B".

**Non-goals.** Refactoring the code into a multi-task framework, implementing a new niche,
touching the 8B config or its results, changing any invariant in `CLAUDE.md`.

**Success criteria**

1. `make data train export eval report CONFIG=configs/qwen3-0.6b-personas.yaml` runs without
   touching `data/` or `outputs/` of the 8B config (they use their own subfolders).
2. The measured numbers of that run (peak VRAM, time per step, total time, metrics) are
   recorded in the ADR. Nothing is estimated.
3. The guide lets a reader locate, for each stage, the file that must change for a new niche,
   and states which parts are niche-independent.
4. `pytest` passes, including the new guard test.

## 2. Findings

- Model, dataset, hyperparameters and paths all come from the YAML. Every script takes
  `--config`; the Makefile has `CONFIG ?=`. A second config needs no code change to run.
- The **task** is not in the config. It is in code: the persona schema and the six-section
  prompt (`src/personas/prompt.py`, `schema.py`), name extraction (`names.py`), data
  preparation from `nvidia/Nemotron-Personas-Brazil` (`scripts/01_prepare_data.py`) and the
  grounding metrics (`grounding.py`, `05_evaluate.py`). This is what a new niche replaces,
  and it is why niche support is a separate, larger project.
- Qwen3 sizes share one tokenizer and chat template (`<|im_start|>` markers), so
  `train_on_responses_only`, `enable_thinking=False` and the raw `/completion` path stay
  valid with no change (invariants 1-3, 5, 6 hold).
- Output names are fixed per outputs directory (`personas-{base,tuned}-q4_k_m.gguf`,
  `merged-16bit-*`, `adapter/`), so separation must come from `paths.*`, not from file names.
- The 8B config's tight settings (batch 1, `max_seq_length: 1536`, `eval_strategy: "no"`)
  exist only because of 8GB (ADR 0014). They do not apply to a 0.6B model.
- **Disk is the binding constraint.** The host has ~5.5GB free (2026-10-05). Qwen3-0.6B needs
  about 5GB at peak (fp16 base 1.2GB, merged 1.2GB, f16 GGUF 1.2GB, Q4 ~0.4GB, caches); 1.7B
  needs ~12GB and 4B ~25GB.

## 3. Design

### 3.1 The config

`configs/qwen3-0.6b-personas.yaml`, same structure and keys as the 8B file, differences only:

| Key | 8B config | 0.6B config | Why |
|---|---|---|---|
| `model.base_id` | `unsloth/Qwen3-8B` | `unsloth/Qwen3-0.6B` | the point of the exercise |
| `model.train_id` | `unsloth/Qwen3-8B-bnb-4bit` | `unsloth/Qwen3-0.6B-bnb-4bit` | standard 4-bit checkpoint, loaded with `use_exact_model_name` as today |
| `model.max_seq_length` | 1536 | 2048 | the 1536 cap was a VRAM workaround |
| `data.n_train` | 10000 | 2000 | minutes, not hours; still a visible effect |
| `train.per_device_train_batch_size` / `gradient_accumulation_steps` | 1 / 16 | 4 / 4 | same effective batch 16, better throughput |
| `train.eval_strategy` | `"no"` | `"steps"` | the logits OOM of ADR 0014 does not apply; the learner sees validation loss |
| Cadence: `train.logging_steps` / `eval_steps` / `save_steps` / `save_total_limit` | 10 / 100 / 250 / 3 | 5 / 25 / 100 / 1 | 125 optimizer steps; disk is tight; no effect on the model |
| `paths.data_dir` | `data` | `data/qwen3-0.6b` | no clobbering |
| `paths.outputs_dir` | `outputs` | `outputs/qwen3-0.6b` | no clobbering |

Everything else is identical on purpose (LoRA `r=32`, learning rate, scheduler, decode
parameters, seeds, llama.cpp commit), so the A/B between sizes isolates the model. A comment
at the top lists the three edits needed to move to 1.7B or 4B (the two model ids and the
paths) and the disk each needs.

The 8B config gets no change except what it already has. `Makefile`'s default `CONFIG`
stays the 8B file; the new config is selected with `CONFIG=...`.

### 3.2 Guard test (`tests/test_config.py`, appended)

For every `configs/*.yaml`: loads with `load_config`; has the keys the scripts read; and the
pair (`paths.data_dir`, `paths.outputs_dir`) is **unique across configs**: two configs may not
share either directory. A subfolder of another config's directory is allowed, because the 8B
config writes files directly into `data/` and `outputs/` and a dedicated subfolder
(`data/qwen3-0.6b`) never collides with them. The failure message names the two configs.

### 3.3 The report title

`scripts/06_report.py` builds the `<h1>` from `cfg.model.base_id` (the part after `/`) instead
of the literal "Qwen3-8B". No other change to the report.

### 3.4 The guide (`docs/guides/adapting-to-a-new-niche.md`, Portuguese, nine sections)

Audience: the repo owner, learning. Plain language, each stage tied to a file. Sections:

1. **O que é fine-tuning aqui:** base model, LoRA adapter, what is trained, and why 4-bit
   (QLoRA). One diagram of the pipeline.
2. **Passo a passo com a config leve:** the commands, what each prints, how to read the
   training log (loss, validation loss, step time), and how to read the eval table.
3. **O que muda entre o 8B e o 0.6B:** the table of 3.1 with the reason for each line, so the
   learner sees which knobs are about the model and which about the hardware.
4. **Mapa do código:** for each stage, whether it is niche-independent (training loop, export,
   serving, `/completion` path, thinking-off rule) or niche-specific (schema, prompt, data
   preparation, metrics), with file names.
5. **Adaptando para um novo nicho**, a checklist: (1) define the task as input → output;
   (2) find or build a dataset and check its licence and provenance; (3) write the prompt
   renderer and bump the prompt version (ADR 0006); (4) write the data preparation and the
   length filter; (5) design metrics that fit the task, since persona grounding does not
   transfer; (6) train, export, evaluate against the base model; (7) decide whether the
   result is good enough or the task needed retrieval instead.
6. **Dois exemplos, sem implementação:**
   - *Cyber security:* e.g. turning an alert or log excerpt into a triage summary with a
     fixed schema. Data: public advisories (CVE/NVD text, vendor bulletins) and synthetic
     logs. Metrics: schema validity, field correctness against ground truth, no invented
     CVE ids. Risks: dual-use (a model tuned to write exploits), leaking real indicators or
     customer data, outdated facts.
   - *Política:* e.g. summarising a bill or speech with neutral framing. Data: public
     legislative records. Metrics: faithfulness to the source, and a balanced-framing check
     built by humans, not by the model. Risks: bias in the training data and in the judge,
     persuasion or targeted political content, personal data about real people, and the
     difficulty of measuring neutrality at all.
7. **Quando fine-tuning não é a ferramenta:** facts that change (laws, CVEs) belong in
   retrieval; tuning teaches format, tone and behaviour, not up-to-date knowledge.
8. **Armadilhas que o projeto já pagou:** prompt/template mismatch between training and
   serving, thinking mode, a changed prompt invalidating the adapter, tuned and base needing
   the same quantisation lineage (links to the ADRs).

9. **Weights & Biases (opcional):** setup (`wandb login`, install outside the lock), privacy and
   the offline mode, an example script that logs `results.json` and `generations.json` after
   `make eval` (no pipeline change), how to compare the 0.6B and 8B runs, how training curves
   would be enabled through a `train.report_to` key (needs its own spec and ADR), and when the
   sibling product Weave is worth it.

The guide describes what exists today and labels the niche sections as design guidance, not
implemented features.

### 3.5 ADR 0017

Records: the 0.6B config as a learning configuration; separate `paths.*` rather than separate
file names; which ADR 0014 constraints do not apply and why; the measured numbers from the
run; disk as the constraint on larger models; the rejected alternative of refactoring to a
multi-task framework now. Marks no earlier ADR superseded; it adds a configuration.

## 4. Risks

| Risk | Mitigation |
|---|---|
| A careless `rm -rf data` or `rm -rf outputs` removes both configs' files, because the 0.6B folders live inside the 8B ones | the guide and ADR say so next to the cleanup command; the verification step lists the 8B files before and after the run |
| 0.6B quality may be too low for six-section Portuguese output, so the "tuned beats base" result looks weak | that is a legitimate thing to learn from; the ADR records the metrics as measured, and the guide says small models are for understanding the mechanism |
| `make clean` only removes the 8B outputs' merged files | the guide documents the per-config cleanup command; no Makefile change |
| Disk (~5.5GB free) is too tight for the 0.6B export | the run records `df` before and after; if it does not fit, the implementer stops and reports instead of deleting caches |
| The ADR numbers are lost if the run is skipped | the ADR is written only after the run; the plan makes that ordering explicit |

## 5. Verification plan

1. `uv run --extra dev pytest -q`, including the guard test (red first).
2. `make data CONFIG=configs/qwen3-0.6b-personas.yaml` (the dataset stream is in the HF
   cache), then `make train`, `make export`, `make eval`, `make report` with the same
   `CONFIG`. Record `df -h /` before and after.
3. Confirm `data/` and `outputs/` of the 8B config are byte-for-byte untouched (the 8B files
   that exist today are listed before the run and compared after).
4. Record peak VRAM, seconds per step, total wall time, validation loss, and the eval metrics
   for base and tuned.
5. Read the guide once against the real file names (every path it mentions must exist).

## 6. Open questions

None blocking. Measured during the run and written into ADR 0017: whether `eval_strategy:
"steps"` is affordable at batch 4 on this card with the desktop resident, and the real
training time for 2000 rows.
