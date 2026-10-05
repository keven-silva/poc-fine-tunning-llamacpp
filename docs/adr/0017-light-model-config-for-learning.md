# ADR 0017 — A light-model config for learning

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

The main config trains Qwen3-8B. On an 8GB card that took ADR 0014's three workarounds
(standard 4-bit checkpoint, batch 1, no in-training evaluation) and hours of training, which
makes it a poor tool for learning how each stage behaves. The owner wants to run the pipeline
end to end quickly, and to understand how it would be adapted to other niches.

The pipeline is already config-driven: model, rows, hyperparameters and paths come from the
YAML, and every script takes `--config`. The task itself (schema, prompt, data preparation,
metrics) is in code and is specific to personas, so other niches are a separate, larger
project (see the guide).

## Decision

1. Add `configs/qwen3-0.6b-personas.yaml`: the same persona task on `unsloth/Qwen3-0.6B`,
   trained from `unsloth/Qwen3-0.6B-bnb-4bit`. It differs from the 8B config only in
   `max_seq_length` (2048), `n_train` (2000), in-training evaluation (`steps`, every 25), the
   logging, validation and checkpoint cadence (`logging_steps` 5, `eval_steps` 25, `save_steps`
   100, `save_total_limit` 1: 125 optimizer steps, no effect on the model) and the two paths.
   The batch shape is the 8B's (1 × 16): a first run with 4 × 4 at 2048 tokens measured ~60 s
   per optimizer step (57-73 s) with VRAM at 7.89 of 8.0 GB, and a second attempt ran out of
   memory at step 8 (a 1.50 GiB allocation with 1.46 GiB free). The fp32 logits over the
   151,936-token vocabulary cost the same memory at any Qwen3 size (the same cause as
   ADR 0014), so that shape was abandoned. LoRA, learning rate,
   scheduler, decoding, seeds and the llama.cpp commit are identical, so a comparison between
   the two isolates model size.
2. Separate the runs by `paths.data_dir` (`data/qwen3-0.6b`) and `paths.outputs_dir`
   (`outputs/qwen3-0.6b`), not by file names: output names are fixed per directory. A test
   rejects two configs that share either directory.
3. The 8B config, its tests and its results are untouched. The Makefile default stays the 8B
   file; the light config is selected with `CONFIG=`.
4. The report title comes from `model.base_id`.
5. `docs/guides/adapting-to-a-new-niche.md` explains the pipeline, the 8B-to-0.6B differences,
   which code is niche-specific, and how a new niche (cyber security, politics) would be
   approached, as guidance and not as implemented features.

## Consequences

- Measured on the RX 7600 8GB (2026-10-05), 2000 training rows, 125 optimizer steps at the
  1 × 16 shape: training took 2818 s (47 minutes), about 22.5 s per optimizer step by wall
  time (the script measured 21.09 s; early steps ran ~19.5 s and rose to ~33 s around the
  validation passes). The script reported a peak of 7.20 GB of VRAM (reserved memory). Total
  VRAM sampled from `/sys`, desktop included, grew from 4.6 GB at 5 minutes to 6.4 GB at 20,
  7.3 GB at 38 and 8.07 GB at the end, against ~3.5 GB in the first 10 steps: the allocator
  reserves more as the run goes on, so the card was effectively full, though nothing ran out
  of memory. In-training validation loss (every 25 steps) read 1.78, 1.581, 1.502, 1.47 and
  1.464, against a training loss going from 2.511 at step 5 to 1.473 at step 125 (mean 1.643).
- Export took 71 s for the tuned model and 22 s for the base. Each Q4_K_M GGUF is 397 MB; the
  q8_0 files (639 MB each) are kept by the export script.
- Evaluation on 200 held-out rows (base → tuned): perplexity 10.71 → 5.97, format validity
  0.0 → 0.825, grounding recall 0.95 → 0.867. Think-leak rate was 0 → 0 and the Portuguese
  rate 0.625 → 1.0; mean output length went from 688 to 2792 characters. Generation took
  205 s for the base and 768 s for the tuned model. No generation contained `<think>`. These
  numbers are not comparable with `docs/RESULTS.md` (different model, rows and sequence
  length).
- Format validity of 0.825 means about 17% of tuned outputs still fail the format check. The
  0.6B result is modest, which is the point of a learning config.
- Grounding recall went down. By field (tuned): name 0.95, municipality 0.97, state 0.79,
  occupation 0.78, age 0.845; base: 0.995, 0.89, 0.89, 0.985, 0.99. A hypothesis, not
  verified: the base's short, mostly unformatted outputs echo the prompt, so they score high
  recall, while the tuned outputs are longer and paraphrase the fields.
- The first `make eval` run failed on the tuned model: every request hit `ReadTimeout`
  (300 s × 3 attempts, 1872 s in total). A standalone tuned server (same arguments, started
  with `scripts/04_serve.sh`, the eval's exact request body, concurrency 2) answered 24 of 24
  requests in 5-10 s with EOS, and an unchanged re-run of `make eval` passed. The cause is
  unexplained and was not reproduced.
- Free disk went from 134 GB to 127 GB over the run (about 7 GB net, including model
  downloads, the 397 MB Q4_K_M and 639 MB q8_0 GGUFs and the leftover merged directories).
  The export script removes the f16 file itself, but the `merged-16bit-*` directories are left
  behind and `make clean` does not reach `outputs/qwen3-0.6b/` (it only covers the 8B layout),
  so cleaning those is manual: `rm -rf outputs/qwen3-0.6b/merged-16bit-*`.
- The 8B files under `data/` and `outputs/` were compared before and after the run: untouched.
- The 0.6B folders live inside `data/` and `outputs/`, so deleting those two folders removes
  both configs' files.
- Larger models (1.7B, 4B) need only the model ids and the two paths changed, plus the disk
  noted at the top of the config (~12GB and ~25GB at the export peak). Their batch shape must
  be re-measured, since the logits cost does not shrink with the model.
- ADR 0014's constraints, one by one. Batch size carries over: the fp32 logits over the
  vocabulary do not depend on model size, so the 0.6B keeps batch 1 x 16, and the evaluation
  batch is 1 as in the 8B config. Sequence length does not carry over: 2048 is used here
  because the 1536 limit was set by the 8B's memory. The standard 4-bit checkpoint is used for
  consistency, and in-training evaluation is back on (`eval_strategy: "steps"`).
- No earlier ADR is superseded; ADR 0014's constraints still hold for the 8B config.

## Alternatives considered

- **Refactor the code into a multi-task framework now.** It would let a niche be a config,
  but it changes the prompt, data and evaluation code that invariants 1-3 and 6 protect, and
  the owner's goal is to understand the mechanism first. Deferred as its own project.
- **Reuse the same output paths with different file names.** Fragile: the scripts derive
  names from the stage, and two runs would still overwrite `adapter/` and the eval results.
- **Batch 4 × 4 for speed.** Tried first and measured, see Decision 1; ~60 s/step, memory at
  the limit, and an out-of-memory failure in a second attempt.
- **Start from 1.7B.** More capable, but ~12GB at the export peak (a rough estimate), and the
  disk was nearly full when the config was designed. It was cleared before the run (134 GB
  free), so disk is no longer the obstacle it looked like, but the 1.7B batch shape would
  still need to be measured.
