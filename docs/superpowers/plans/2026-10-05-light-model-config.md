# Light-Model Config and Niche Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Qwen3-0.6B config that trains in minutes on the existing persona task, a Portuguese guide to adapting the pipeline to new niches (including an optional section on tracking runs and evaluations with Weights & Biases), and an ADR with measured numbers, without touching the 8B config or its outputs.

**Architecture:** The pipeline is already config-driven (model, data, hyperparameters, paths), so the new model is a second YAML pointing at its own `data/qwen3-0.6b` and `outputs/qwen3-0.6b`. A guard test makes path collisions between configs impossible. The report title stops hardcoding the model name. The guide documents the pipeline stage by stage and marks which parts are niche-specific.

**Tech Stack:** YAML config, pytest, uv, Unsloth QLoRA, llama.cpp (HIP), Markdown.

**Spec:** [docs/superpowers/specs/2026-10-05-light-model-config-and-niche-guide-design.md](../specs/2026-10-05-light-model-config-and-niche-guide-design.md)

## Global Constraints

- The 8B config `configs/qwen3-8b-personas.yaml`, its tests, `docs/RESULTS.md` and every accepted ADR body stay untouched. The Makefile default `CONFIG` stays the 8B file.
- Same task, prompt, dataset, LoRA (`r=32`), learning rate, scheduler, decode parameters, seeds and llama.cpp commit as the 8B config; only the keys in the spec's table (section 3.1) differ.
- Qwen3 family only: same tokenizer and chat template, so invariants 1-3, 5, 6 of `CLAUDE.md` hold unchanged. Thinking mode off everywhere; raw `/completion` only.
- Nothing in `src/personas/` changes.
- Weights & Biases is **explained, not integrated**: no new dependency, no `uv.lock` change, `report_to="none"` in `scripts/02_train.py` stays, and the example script lives only inside the guide. Evaluation keeps going through the raw `/completion` path and `scripts/05_evaluate.py` unchanged (invariant 3).
- `paths.data_dir` is `data/qwen3-0.6b` and `paths.outputs_dir` is `outputs/qwen3-0.6b`; two configs never share a directory.
- Measured numbers go into ADR 0017 only after the run; no estimated figures, no placeholder text in committed files.
- Python is the project `uv` venv only. Never pipe long-running commands through `tail` to judge success; redirect to a file in `/tmp/claude-1000`.
- Disk is ~5.5GB free (99% used). Never delete caches or other people's files to make room: stop and report.
- The user's desktop session may hold ~1.3GB of VRAM; never close their applications.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. The same directory spelled two ways (`data` vs `./data/`) must still count as a collision between two configs; the guard test pins this with a synthetic case.
2. `model.train_id` must be the `-bnb-4bit` checkpoint of the same family as `model.base_id`; a mismatched pair would train one model and export another.
3. `eval_strategy: "steps"` without `eval_steps` must be rejected, since `scripts/02_train.py` would run with a silently default cadence.
4. Moving to 1.7B or 4B must be an edit of the model ids and the two paths only; the config comment names exactly those edits and the disk each needs.
5. The run must not alter the 8B config's existing files under `data/` and `outputs/` (compared before and after).
6. The guide must not mention a file that does not exist; a test checks every backticked path.
7. The W&B example script in the guide must parse (`ast.parse` in a test) and may read only what `make eval` writes (`results.json`, `generations.json`); it must not re-run or alter the evaluation. The guide must also say that logging generations sends text to a cloud service, and give the offline alternative, because the niches discussed (cyber security, politics) can involve sensitive text.

---

## File Structure

| File | Responsibility |
|---|---|
| `configs/qwen3-0.6b-personas.yaml` (new) | the light config |
| `tests/test_config.py` (append) | config guard tests |
| `scripts/06_report.py` | `report_title()` helper and its use in the `<h1>` |
| `tests/test_report_title.py` (new) | tests for `report_title` |
| `docs/guides/adapting-to-a-new-niche.md` (new) | the learning guide |
| `tests/test_guide.py` (new) | guide structure and path-existence tests |
| `docs/adr/0017-light-model-config-for-learning.md`, `docs/adr/README.md` | decision record |
| `README.md`, `CLAUDE.md` | one pointer each |

---

### Task 1: The light config and its guard tests

**Files:**
- Create: `configs/qwen3-0.6b-personas.yaml`
- Modify: `tests/test_config.py` (append)

**Interfaces:**
- Produces: config file path `configs/qwen3-0.6b-personas.yaml` with keys `model.{base_id,train_id,max_seq_length,load_in_4bit}`, `data.*`, `lora.*`, `train.*` (including `eval_strategy`, `eval_steps`, `eval_subset_rows`, `group_by_length`), `decode.*`, `llamacpp.commit`, `paths.{data_dir,outputs_dir,llamacpp_dir}`; test helper `_collisions(paths_by_config: dict[str, dict[str, str]]) -> list[str]` in `tests/test_config.py`.

- [ ] **Step 1: Create the working branch**

The ROCm work is already merged into `master` (fast-forward, commit `07f46f0`), so the new branch starts from `master`. The spec and this plan are untracked files and travel with the checkout. Run `git branch --show-current` first and `git status --short`: expected `master` and only those two untracked files.

```bash
git checkout -b feat/light-model-config
git add docs/superpowers/specs/2026-10-05-light-model-config-and-niche-guide-design.md docs/superpowers/plans/2026-10-05-light-model-config.md
git commit -m "docs: spec and plan for the light-model config and niche guide

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_config.py`:

```python
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
```

- [ ] **Step 2b: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_config.py -q > /tmp/claude-1000/cfg-red.log 2>&1; echo "exit $?"; grep -E "passed|failed|FileNotFoundError" /tmp/claude-1000/cfg-red.log | head -5`
Expected: the helper tests pass; `test_light_config_is_the_8b_task_with_only_the_documented_differences` fails with `FileNotFoundError` for `qwen3-0.6b-personas.yaml`.

- [ ] **Step 3: Create `configs/qwen3-0.6b-personas.yaml`**

```yaml
# Light configuration for learning (ADR 0017): the same persona task as the 8B config, on
# Qwen3-0.6B. It trains in minutes, so every stage can be run and read end to end.
#
# Only these things differ from configs/qwen3-8b-personas.yaml: the model ids, the sequence
# length, the number of training rows, the batch shape (same effective batch of 16), in-training
# evaluation (on again: the 8B config turned it off only to fit 8GB, ADR 0014) and the two
# paths. Everything else is identical on purpose, so comparing the two isolates model size.
#
# To move to a bigger model, edit ONLY model.base_id, model.train_id, paths.data_dir and
# paths.outputs_dir. Free disk needed at peak for the export stage: 0.6B ~5GB, 1.7B ~12GB,
# 4B ~25GB. Examples: unsloth/Qwen3-1.7B (+ -bnb-4bit), unsloth/Qwen3-4B (+ -bnb-4bit).
# Use `make <stage> CONFIG=configs/qwen3-0.6b-personas.yaml`.

model:
  base_id: unsloth/Qwen3-0.6B
  # Standard bnb 4-bit checkpoint, loaded with use_exact_model_name (as in ADR 0014).
  # Tokenizer and the fp16 base export still come from base_id.
  train_id: unsloth/Qwen3-0.6B-bnb-4bit
  max_seq_length: 2048
  load_in_4bit: true

data:
  hf_dataset: nvidia/Nemotron-Personas-Brazil
  n_train: 2000
  n_val: 500
  n_test: 200
  seed: 3407
  max_name_drop_rate: 0.05

lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.0
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_gradient_checkpointing: unsloth
  random_state: 3407

train:
  # Effective batch 16, as in the 8B config, but 4 sequences at a time: a 0.6B model has room.
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  group_by_length: true
  num_train_epochs: 1
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  optim: adamw_8bit
  weight_decay: 0.01
  # 2000 rows / batch 16 = 125 optimizer steps: log often so the loss curve is readable.
  logging_steps: 5
  # On: the fp32-logits OOM of ADR 0014 belongs to the 8B model. 125 steps / 25 = 5 readings.
  eval_strategy: "steps"
  eval_steps: 25
  eval_subset_rows: 150
  save_steps: 100
  save_total_limit: 1
  seed: 3407

decode:
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  repeat_penalty: 1.05
  n_predict: 1024
  seed: 3407

llamacpp:
  # Read by scripts/00_setup_llamacpp.sh (an exported LLAMA_COMMIT overrides it).
  commit: "bdeb855b30dfe7f6e695cba98445a7ba09e6416e"

paths:
  data_dir: data/qwen3-0.6b
  outputs_dir: outputs/qwen3-0.6b
  llamacpp_dir: vendor/llama.cpp
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_config.py -q > /tmp/claude-1000/cfg-green.log 2>&1; echo "exit $?"; grep -E "passed|failed" /tmp/claude-1000/cfg-green.log`
Expected: `exit 0`, all passed (the original 8B tests plus the new ones). If `test_every_config_has_the_keys_the_scripts_read` fails for the 8B config on a key in `REQUIRED_KEYS`, the key list is wrong for the real scripts: check which key the scripts actually read (`grep -n "cfg\." scripts/*.py`) and correct the list, never the 8B config.

- [ ] **Step 5: Commit**

```bash
git add configs/qwen3-0.6b-personas.yaml tests/test_config.py
git commit -m "feat: add a Qwen3-0.6B config for learning, with collision guards

Same persona task and hyperparameters as the 8B config; only the model, sequence length, rows, batch shape, in-training eval and paths differ. Tests keep two configs from sharing data or outputs directories.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The report title follows the config

**Files:**
- Modify: `scripts/06_report.py` (add `report_title`, use it at the `<h1>` line, ~line 112)
- Create: `tests/test_report_title.py`

**Interfaces:**
- Produces: `report_title(base_id: str) -> str` in `scripts/06_report.py`, returning `"<model name without org> persona fine-tune"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_title.py`:

```python
import importlib.util
from pathlib import Path

REPORT = Path(__file__).parent.parent / "scripts" / "06_report.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("report_script", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_title_drops_the_organisation_prefix():
    assert _load_report().report_title("unsloth/Qwen3-0.6B") == "Qwen3-0.6B persona fine-tune"


def test_title_keeps_a_bare_model_name():
    assert _load_report().report_title("Qwen3-8B") == "Qwen3-8B persona fine-tune"


def test_title_uses_the_last_path_segment_of_a_nested_id():
    assert _load_report().report_title("org/sub/Model-X") == "Model-X persona fine-tune"


def test_report_source_no_longer_hardcodes_a_model_name():
    assert "Qwen3-8B persona" not in REPORT.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_report_title.py -q > /tmp/claude-1000/title-red.log 2>&1; echo "exit $?"; grep -E "passed|failed|AttributeError|Error" /tmp/claude-1000/title-red.log | head -5`
Expected: FAIL (`AttributeError` for `report_title`, and the last test fails on the hardcoded string). If importing the script fails for another reason (for example it executes `main()` at import), stop and report: the script must stay guarded by `if __name__ == "__main__":`; adding that guard is allowed.

- [ ] **Step 3: Implement**

In `scripts/06_report.py`, add this function above `main()` (next to `metric_row`):

```python
def report_title(base_id: str) -> str:
    """'unsloth/Qwen3-0.6B' -> 'Qwen3-0.6B persona fine-tune'."""
    return f"{base_id.rsplit('/', 1)[-1]} persona fine-tune"
```

and change the heading line inside the page f-string from

```html
<h1>Qwen3-8B persona fine-tune &mdash; base vs tuned</h1>
```

to

```html
<h1>{html.escape(report_title(cfg.model.base_id))} &mdash; base vs tuned</h1>
```

Change nothing else in the script.

- [ ] **Step 4: Run the tests and the whole suite**

Run: `uv run --extra dev pytest -q > /tmp/claude-1000/title-green.log 2>&1; echo "exit $?"; grep -E "passed|failed" /tmp/claude-1000/title-green.log`
Expected: `exit 0`, all passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/06_report.py tests/test_report_title.py
git commit -m "fix: derive the report title from the config instead of hardcoding Qwen3-8B

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The guide, with tests that keep it honest

**Files:**
- Create: `docs/guides/adapting-to-a-new-niche.md`, `tests/test_guide.py`

**Interfaces:**
- Consumes: file names that exist after Tasks 1-2 (`configs/qwen3-0.6b-personas.yaml`, `scripts/06_report.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guide.py`:

```python
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
GUIDE = ROOT / "docs" / "guides" / "adapting-to-a-new-niche.md"

# Created while the pipeline runs, or tool output folders: not checked into git.
RUNTIME_PREFIXES = ("data/", "outputs/", "vendor/")


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_has_the_nine_sections_in_order():
    headings = re.findall(r"^## (\d)\. ", _text(), re.M)
    assert headings == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


def test_every_file_the_guide_names_exists():
    prose = re.sub(r"```.*?```", "", _text(), flags=re.S)  # fenced blocks may show runtime output
    named = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|yaml|md|sh|toml))`", prose))
    assert named, "the guide should reference real files"
    missing = sorted(
        p for p in named
        if not p.startswith(RUNTIME_PREFIXES) and not (ROOT / p).exists()
    )
    assert missing == [], f"the guide names files that do not exist: {missing}"


def test_guide_covers_both_example_niches_and_the_retrieval_caveat():
    text = _text().lower()
    for needle in ("cyber security", "política", "retrieval"):
        assert needle in text, needle


def test_wandb_section_covers_setup_privacy_evaluation_and_training():
    section = _text().split("## 9. ", 1)[1]
    for needle in ("wandb login", "WANDB_MODE=offline", "wandb sync", "wandb.Table",
                   "report_to", "WANDB_PROJECT", "uv.lock"):
        assert needle in section, needle


def test_wandb_example_scripts_parse_and_read_only_what_eval_writes():
    import ast

    blocks = re.findall(r"```python\n(.*?)```", _text(), flags=re.S)
    assert blocks, "the guide should carry the W&B example script"
    for block in blocks:
        ast.parse(block)
    wandb_blocks = [b for b in blocks if "import wandb" in b]
    assert len(wandb_blocks) == 1
    script = wandb_blocks[0]
    assert "results.json" in script and "generations.json" in script
    assert "/completion" not in script and "subprocess" not in script  # never re-runs the eval


def test_guide_states_the_core_invariants_a_new_niche_must_keep():
    text = _text()
    for needle in ("/completion", "enable_thinking=False", "PROMPT_VERSION"):
        assert needle in text, needle
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_guide.py -q > /tmp/claude-1000/guide-red.log 2>&1; echo "exit $?"; grep -E "passed|failed|FileNotFoundError" /tmp/claude-1000/guide-red.log | head -3`
Expected: FAIL with `FileNotFoundError` for the guide.

- [ ] **Step 3: Write the guide**

Create `docs/guides/adapting-to-a-new-niche.md` with exactly this content:

````markdown
# Guia: entender o fine-tuning e adaptar o pipeline para outro nicho

Este guia é para quem quer **aprender como o fine-tuning funciona** neste projeto e saber o
que mudaria para treinar um modelo em outro assunto (cyber security, política, etc.).
Ele descreve o que existe hoje; as partes sobre novos nichos são orientação de projeto, não
funcionalidades implementadas.

## 1. O que é fine-tuning aqui

- Há um **modelo base** (Qwen3). Ele já sabe português e escrever texto, mas não sabe o nosso
  formato: receber atributos demográficos e devolver uma persona em seis seções.
- Não retreinamos o modelo inteiro. Congelamos os pesos e treinamos um **adaptador LoRA**:
  matrizes pequenas (`r=32`) somadas às camadas de atenção e MLP. Por isso cabe numa GPU de 8GB.
- O modelo base é carregado em **4 bits** (QLoRA) só durante o treino, para ocupar menos
  memória. Depois, o adaptador é fundido no modelo em 16 bits e convertido para **GGUF**
  quantizado (Q4_K_M) para rodar no llama.cpp.
- A perda (loss) é calculada **só na resposta do assistente**, não no prompt. Isso é o
  `train_on_responses_only` em `scripts/02_train.py`.

Fluxo completo:

```
dados (Hugging Face) -> JSONL de treino/validação/teste -> treino LoRA -> adaptador
   -> fusão + GGUF (tuned) e GGUF do base -> llama-server -> avaliação base vs tuned -> relatório
```

Decisões por trás de cada passo estão em `docs/adr/README.md`.

## 2. Passo a passo com a config leve

A config `configs/qwen3-0.6b-personas.yaml` usa o Qwen3-0.6B: treina em minutos. Todos os
comandos recebem `CONFIG=`:

```
make data   CONFIG=configs/qwen3-0.6b-personas.yaml
make train  CONFIG=configs/qwen3-0.6b-personas.yaml
make export CONFIG=configs/qwen3-0.6b-personas.yaml
make eval   CONFIG=configs/qwen3-0.6b-personas.yaml
make report CONFIG=configs/qwen3-0.6b-personas.yaml
```

O que cada etapa faz e como ler a saída:

1. **`make data`** (`scripts/01_prepare_data.py`): baixa o dataset, monta o texto de cada
   exemplo, descarta os longos demais para `max_seq_length` e grava `train.jsonl`,
   `val.jsonl`, `test.jsonl` e `manifest.json` em `data/qwen3-0.6b/`.
2. **`make train`** (`scripts/02_train.py`): imprime a *loss* a cada 5 passos. Ela deve cair
   rápido no começo e depois estabilizar. Com a avaliação ligada, aparece também a *eval_loss*
   (validação) a cada 25 passos: se a loss de treino cai e a de validação sobe, é overfitting.
   Ao final há uma linha com o tempo e o **pico de VRAM**.
3. **`make export`** (`scripts/03_export_gguf.py`): funde o adaptador, converte e quantiza.
   Gera os GGUF do modelo *tuned* e do *base* pelo mesmo caminho, para que a comparação meça
   só o LoRA.
4. **`make eval`** (`scripts/05_evaluate.py`): sobe o `llama-server` com um modelo de cada
   vez (`scripts/04_serve.sh`) e mede perplexidade, validade de formato, e se o texto cita os
   atributos de entrada (*grounding*, `src/personas/grounding.py`).
5. **`make report`** (`scripts/06_report.py`): gera uma página com as métricas e exemplos
   lado a lado.

Como ler a tabela do eval: o modelo base costuma ter **validade de formato 0** (não sabe as seis
seções) e o tuned perto de 1; a perplexidade do tuned deve ser menor. Num modelo pequeno, espere
ganhos de formato claros e ganhos de qualidade de conteúdo mais modestos. Isso também é
aprendizado.

## 3. O que muda entre o 8B e o 0.6B

| Chave | 8B | 0.6B | Por quê |
|---|---|---|---|
| `model.base_id` / `train_id` | Qwen3-8B | Qwen3-0.6B | o objetivo do exercício |
| `model.max_seq_length` | 1536 | 2048 | o limite do 8B era para caber em 8GB |
| `data.n_train` | 10000 | 2000 | minutos em vez de horas |
| batch × acumulação | 1 × 16 | 4 × 4 | mesmo batch efetivo (16); o modelo pequeno tem folga |
| `train.eval_strategy` | `"no"` | `"steps"` | o OOM da avaliação (ADR 0014) era do 8B |
| `paths.*` | `data`, `outputs` | `data/qwen3-0.6b`, `outputs/qwen3-0.6b` | não sobrescrever o 8B |

Todo o resto é idêntico (LoRA, learning rate, decodificação, sementes). Assim, comparar os dois
isola o efeito do tamanho. Para subir para 1,7B ou 4B, edite apenas os ids do modelo e os dois
caminhos, e confira o disco livre necessário no topo do YAML.

## 4. Mapa do código: o que é do nicho e o que não é

**Independente do nicho** (reaproveite como está):

- o laço de treino e o LoRA: `scripts/02_train.py`
- fusão, conversão e quantização: `scripts/03_export_gguf.py`
- o servidor e o caminho `/completion`: `scripts/04_serve.sh`, `src/personas/llm.py`
- a leitura da config: `src/personas/config.py`
- a regra de paridade de prompt e o modo thinking desligado (ver seção 8)

**Específico de persona** (é isto que você troca para um nicho novo):

- o formato de entrada e saída: `src/personas/schema.py`
- o prompt e o parser das seções: `src/personas/prompt.py`
- a extração de nome: `src/personas/names.py`
- a preparação dos dados a partir do dataset da NVIDIA: `scripts/01_prepare_data.py`
- as métricas de grounding e de formato: `src/personas/grounding.py` e `scripts/05_evaluate.py`

Hoje esses itens específicos estão no código, não na config. Por isso trocar de nicho exige
escrever código novo; transformar isso em algo configurável seria um projeto à parte.

## 5. Adaptando para um novo nicho: o checklist

1. **Defina a tarefa como entrada → saída.** Se você não consegue escrever 5 exemplos à mão,
   ainda não está pronto para treinar.
2. **Ache ou construa o dataset.** Verifique a licença e a origem. Dados sintéticos geram menos
   risco de privacidade, mas herdam os vieses de quem os gerou.
3. **Escreva o renderizador de prompt e suba a versão** (`PROMPT_VERSION`): mudar o prompt
   invalida o adaptador treinado.
4. **Escreva a preparação dos dados**, incluindo o filtro por tamanho em tokens.
5. **Projete métricas que combinem com a tarefa.** O *grounding* de persona não se transfere:
   saída estruturada pede validade de esquema; resumo pede fidelidade à fonte.
6. **Treine, exporte e compare com o modelo base** pelo mesmo caminho de quantização.
7. **Decida se o resultado basta**, ou se o problema pedia outra ferramenta (seção 7).

## 6. Dois exemplos, sem implementação

**Cyber security: triagem de alertas.** Entrada: um trecho de log ou alerta. Saída: um resumo
com campos fixos (tipo de ameaça, severidade, ação sugerida).

- Dados: avisos públicos (texto de CVE/NVD, boletins de fornecedores) e logs sintéticos.
- Métricas: validade do esquema, campos corretos contra um gabarito, **nenhum CVE inventado**.
- Riscos: *dual-use* (um modelo ajustado para escrever exploits), vazamento de indicadores ou
  dados reais de clientes, fatos desatualizados.

**Política: resumo neutro de projetos de lei ou discursos.** Entrada: o texto da fonte. Saída:
um resumo com enquadramento neutro.

- Dados: registros legislativos públicos.
- Métricas: fidelidade à fonte e uma checagem de equilíbrio feita por pessoas, não pelo próprio
  modelo.
- Riscos: viés nos dados e no avaliador, conteúdo persuasivo ou direcionado, dados pessoais de
  pessoas reais, e a dificuldade de medir neutralidade.

## 7. Quando fine-tuning não é a ferramenta

Fine-tuning ensina **formato, tom e comportamento**, não fatos atualizados. Se o que muda é o
conhecimento (leis novas, CVEs novos), use *retrieval* (buscar o documento e colocá-lo no
prompt). Na prática costuma-se combinar: fine-tuning para o formato, retrieval para os fatos.

## 8. Armadilhas que este projeto já pagou

- **Prompt diferente entre treino e serviço** quebra os resultados sem erro visível. Todo prompt
  vem de um único renderizador, e `tests/test_prompt_parity.py` garante isso
  (`docs/adr/0006-prompt-parity-and-thinking-mode.md`).
- **Thinking mode desligado em todo lugar** (`enable_thinking=False`); qualquer `<think>` na
  geração é falha de formato.
- **Avaliar só por `/completion`**: o endpoint de chat reaplica um template próprio e quebra a
  paridade.
- **Mudar o prompt invalida o adaptador**: suba o `PROMPT_VERSION` e retreine.
- **Base e tuned precisam da mesma linhagem de quantização**
  (`docs/adr/0008-gguf-export-path.md`), senão a comparação mede a quantização, não o LoRA.
- **A VRAM manda**: veja `docs/adr/0014-training-qwen3-8b-on-8gb.md` para o que cada ajuste de
  memória custou.

## 9. Acompanhando experimentos e avaliação com Weights & Biases (opcional)

O [Weights & Biases](https://wandb.ai/site/) (W&B) é um serviço que guarda, para cada execução,
os hiperparâmetros, as curvas e as tabelas, e compara execuções lado a lado numa página web.
Ele **não faz parte do pipeline hoje**: o treino usa `report_to="none"` em `scripts/02_train.py`
e a avaliação grava `results.json` e `generations.json`. Nada desta seção é necessário para
rodar o projeto; ela mostra como você acrescentaria o W&B para aprender e comparar execuções.

### 9.1 Preparação, custo e privacidade

1. Crie uma conta em wandb.ai, copie a chave de API (Settings) e rode `wandb login`.
2. Instale no venv do projeto: `uv pip install wandb`. Isso **não entra no `uv.lock`**, então
   serve para experimentar. Para adotar de verdade, o `wandb` entra em um extra do
   `pyproject.toml`, o lock é refeito e isso vira uma ADR (a regra de dependências fixadas está
   em `docs/adr/0011-python-env-and-pinning.md`).
3. **Privacidade:** o W&B envia o que você loga para a nuvem. O dataset deste projeto é
   sintético, então logar métricas e gerações é seguro. Em um nicho com texto sensível (logs de
   clientes em cyber security, conteúdo político de pessoas reais), logue **só números** ou use
   o modo offline: `WANDB_MODE=offline`. Os dados ficam na pasta local `wandb/` e você os envia
   depois, se quiser, com `wandb sync wandb/run-<data>-<id>`. Ao adotar, adicione `wandb/` ao
   `.gitignore`.

### 9.2 Avaliação: logar o resultado do `make eval`

Não é preciso mexer no pipeline. Depois do `make eval`, um script lê os dois arquivos que ele
já grava e os envia ao W&B. Salve-o fora do repositório, ou em um arquivo seu:

```python
"""Loga o resultado de `make eval` no Weights & Biases.

Uso: python log_eval_to_wandb.py configs/qwen3-0.6b-personas.yaml
"""
import json
import sys
from pathlib import Path

import wandb
import yaml


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    eval_dir = Path(cfg["paths"]["outputs_dir"]) / "eval"
    results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    generations = json.loads((eval_dir / "generations.json").read_text(encoding="utf-8"))

    run = wandb.init(
        project="persona-finetune",
        name=Path(config_path).stem,  # ex.: qwen3-0.6b-personas
        config={
            "base_id": cfg["model"]["base_id"],
            "n_train": cfg["data"]["n_train"],
            "lora_r": cfg["lora"]["r"],
            "learning_rate": cfg["train"]["learning_rate"],
            "prompt_version": results["prompt_version"],
            "n_test_rows": results["n_rows"],
        },
    )
    for model_name, metrics in results["models"].items():  # "base" e "tuned"
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                run.summary[f"{model_name}/{metric}"] = value

    table = wandb.Table(columns=["uuid", "reference", "base", "tuned"])
    for row in generations[:50]:  # uma amostra basta para ler lado a lado
        table.add_data(row["uuid"], row["reference"], row["base"], row["tuned"])
    run.log({"generations": table})
    run.finish()


if __name__ == "__main__":
    main(sys.argv[1])
```

Como usar para comparar: rode o script uma vez para a config do 0,6B e outra para a do 8B, no
mesmo projeto. Na página do projeto, selecione as duas execuções em *Runs*: as métricas
`base/format_validity`, `tuned/format_validity`, `tuned/perplexity` e `tuned/grounding_recall`
aparecem lado a lado, e a tabela `generations` mostra o texto de referência, do base e do tuned
para os mesmos exemplos. Nomeie as execuções pela config e **não compare execuções com
`prompt_version` diferente**: o prompt mudou, então a comparação deixa de ser válida.

### 9.3 Treino: curvas de loss no W&B

O treino registra `loss`, `eval_loss`, taxa de aprendizado e norma do gradiente a cada
`logging_steps`. Para enviá-los ao W&B, o caminho correto neste projeto (os valores vivem na
config, não no script) é:

1. adicionar `report_to: "none"` em `train:` nos YAMLs, e em `scripts/02_train.py` trocar
   `report_to="none"` por `report_to=cfg.train.report_to`;
2. para ligar em uma execução, mudar o valor no YAML para `"wandb"`;
3. definir o projeto e rodar: `WANDB_PROJECT=persona-finetune make train CONFIG=...`
   (com `WANDB_MODE=offline` para guardar só localmente).

Isso muda código e dependências, então siga o fluxo normal do projeto (spec e ADR) antes de
adotar. Para aprender, as mesmas curvas já saem no log do terminal.

### 9.4 E o Weave?

O W&B tem um produto irmão, o Weave, feito para avaliar aplicações de LLM com *scorers* (funções
que dão nota a cada saída, inclusive um LLM como juiz). Aqui as métricas são determinísticas e
já calculadas em `scripts/05_evaluate.py`, então o Weave não traz nada de novo. Ele só passaria a
valer em um nicho em que qualidade precisa de julgamento, como a neutralidade na política. Lembre
que um juiz LLM tem viés e deve ser conferido por pessoas.
````

- [ ] **Step 4: Run the tests**

Run: `uv run --extra dev pytest tests/test_guide.py -q > /tmp/claude-1000/guide-green.log 2>&1; echo "exit $?"; grep -E "passed|failed|missing" /tmp/claude-1000/guide-green.log`
Expected: `exit 0`, 6 passed. If `test_every_file_the_guide_names_exists` lists a missing file, the file name in the guide is wrong: fix the guide (verify with `ls`), never create the file.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/adapting-to-a-new-niche.md tests/test_guide.py
git commit -m "docs: add a learning guide to the pipeline and to adapting it to new niches

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Run the light pipeline and record real numbers

**Files:** none committed. Outputs: `/tmp/claude-1000/light-run.txt` (numbers for Task 5). Nothing under `data/qwen3-0.6b` or `outputs/qwen3-0.6b` is committed (git-ignored).

**Interfaces:**
- Consumes: Tasks 1-2 (config and report title), the venv, `vendor/llama.cpp`.
- Produces: the measured values Task 5 writes into ADR 0017: total training wall time, seconds per optimizer step, peak VRAM, validation loss readings, eval metrics for base and tuned, free disk before and after, whether the run completed.

- [ ] **Step 1: Preconditions and a baseline of the 8B files**

```bash
df -h / | tail -1
find data outputs -type f -not -path 'data/qwen3-0.6b/*' -not -path 'outputs/qwen3-0.6b/*' -printf '%p %s %T@\n' 2>/dev/null | sort > /tmp/claude-1000/8b-files-before.txt
wc -l /tmp/claude-1000/8b-files-before.txt
env -u LD_LIBRARY_PATH uv run python scripts/00_check_gpu.py > /tmp/claude-1000/gpu.log 2>&1; echo "gpu check exit $?"
cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -2
```
Expected: GPU check exit 0. If free disk is under 5GB, stop and report BLOCKED (do not delete caches or other files).

- [ ] **Step 2: Data**

```bash
make data CONFIG=configs/qwen3-0.6b-personas.yaml > /tmp/claude-1000/light-data.log 2>&1; echo "data exit $?"
ls -la data/qwen3-0.6b
```
Expected: exit 0; `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json` in `data/qwen3-0.6b/` with 2000 / 500 / 200 rows. A needed dataset download beyond what is cached is acceptable if small; if it starts downloading more than ~1GB, stop and report.

- [ ] **Step 3: Train**

```bash
(time make train CONFIG=configs/qwen3-0.6b-personas.yaml > /tmp/claude-1000/light-train.log 2>&1; echo "train exit $?") 2>&1 | tail -4
grep -E "peak VRAM|finished in|eval_loss|'loss'" /tmp/claude-1000/light-train.log | head -30
```
Expected: exit 0 and a `>> finished in ...h, peak VRAM X.XX GB` line. Record: wall time, seconds per optimizer step (total seconds / 125, say so), peak VRAM, the five `eval_loss` readings and the first and last training `loss`. If it OOMs, record the error and the VRAM held by the desktop before the run, then stop and report (do not change the config).

- [ ] **Step 4: Export, keeping the disk under control**

Free disk is tight, so remove each model's intermediate files as soon as its Q4 exists:

```bash
df -h / | tail -1
env -u LD_LIBRARY_PATH uv run python scripts/03_export_gguf.py --config configs/qwen3-0.6b-personas.yaml --which tuned > /tmp/claude-1000/light-export-tuned.log 2>&1; echo "tuned exit $?"
rm -rf outputs/qwen3-0.6b/merged-16bit-tuned outputs/qwen3-0.6b/gguf/personas-tuned-f16.gguf
df -h / | tail -1
env -u LD_LIBRARY_PATH uv run python scripts/03_export_gguf.py --config configs/qwen3-0.6b-personas.yaml --which base > /tmp/claude-1000/light-export-base.log 2>&1; echo "base exit $?"
rm -rf outputs/qwen3-0.6b/merged-16bit-base outputs/qwen3-0.6b/gguf/personas-base-f16.gguf
ls -la outputs/qwen3-0.6b/gguf; df -h / | tail -1
```
Expected: both exit 0; `personas-tuned-q4_k_m.gguf` and `personas-base-q4_k_m.gguf` exist (each a few hundred MB). The only deletions allowed are the two intermediates per model named above, inside `outputs/qwen3-0.6b/`. If an export fails for lack of disk, record `df` and stop.

- [ ] **Step 5: Evaluate and report**

```bash
make eval CONFIG=configs/qwen3-0.6b-personas.yaml > /tmp/claude-1000/light-eval.log 2>&1; echo "eval exit $?"
make report CONFIG=configs/qwen3-0.6b-personas.yaml > /tmp/claude-1000/light-report.log 2>&1; echo "report exit $?"
cat outputs/qwen3-0.6b/eval/results.json | head -c 1500; echo
grep -c "<think>" outputs/qwen3-0.6b/eval/generations.json
grep -o "<h1>[^<]*" outputs/qwen3-0.6b/*.html outputs/qwen3-0.6b/eval/*.html 2>/dev/null | head -2
```
Expected: both exit 0; `<think>` count 0; the report heading reads `Qwen3-0.6B persona fine-tune`. Record perplexity, format validity and grounding recall for base and tuned.

- [ ] **Step 6: Prove the 8B files were not touched, and record everything**

```bash
find data outputs -type f -not -path 'data/qwen3-0.6b/*' -not -path 'outputs/qwen3-0.6b/*' -printf '%p %s %T@\n' 2>/dev/null | sort > /tmp/claude-1000/8b-files-after.txt
diff /tmp/claude-1000/8b-files-before.txt /tmp/claude-1000/8b-files-after.txt && echo "8B files untouched"
df -h / | tail -1
```
Write `/tmp/claude-1000/light-run.txt` with: `df` before/after, wall time per stage, seconds per step, peak VRAM, loss readings, eval metrics for base and tuned, the "8B files untouched" result, and every anomaly (warnings, retries, any stage skipped and why). Nothing is committed in this task.

---

### Task 5: ADR 0017 and the pointers

**Files:**
- Create: `docs/adr/0017-light-model-config-for-learning.md`
- Modify: `docs/adr/README.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `/tmp/claude-1000/light-run.txt` from Task 4. Where the ADR text below has a value in angle brackets prefixed with `RUN:`, replace it with the number from that file. If a stage did not run or failed, say so in plain words instead of a number. No `RUN:` marker may remain.

- [ ] **Step 1: Create the ADR**

```markdown
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
   `max_seq_length` (2048), `n_train` (2000), the batch shape (4 × 4, still effective 16),
   in-training evaluation (`steps`, every 25) and the two paths. LoRA, learning rate,
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

- Measured on the RX 7600 8GB (2026-10-05), 2000 training rows, 125 optimizer steps:
  training took <RUN: wall time>, about <RUN: seconds> per step, with a peak of
  <RUN: peak VRAM> GB of VRAM. In-training validation loss read <RUN: the five eval_loss
  readings> against a training loss going from <RUN: first loss> to <RUN: last loss>.
- Evaluation on 200 held-out rows (base → tuned): perplexity <RUN: base> → <RUN: tuned>,
  format validity <RUN: base> → <RUN: tuned>, grounding recall <RUN: base> → <RUN: tuned>.
  These numbers are not comparable with `docs/RESULTS.md` (different model, rows and
  sequence length).
- Free disk went from <RUN: before> to <RUN: after>; the export needed the per-model cleanup
  of the f16 and merged intermediates (documented in the run, not automated).
- The 8B files under `data/` and `outputs/` were compared before and after the run:
  <RUN: untouched / what changed>.
- The 0.6B folders live inside `data/` and `outputs/`, so deleting those two folders removes
  both configs' files.
- Larger models (1.7B, 4B) need only the model ids and the two paths changed, plus the disk
  noted at the top of the config (~12GB and ~25GB at the export peak).
- No earlier ADR is superseded; ADR 0014's constraints still hold for the 8B config.

## Alternatives considered

- **Refactor the code into a multi-task framework now.** It would let a niche be a config,
  but it changes the prompt, data and evaluation code that invariants 1-3 and 6 protect, and
  the owner's goal is to understand the mechanism first. Deferred as its own project.
- **Reuse the same output paths with different file names.** Fragile: the scripts derive
  names from the stage, and two runs would still overwrite `adapter/` and the eval results.
- **Start from 1.7B.** More capable, but ~12GB at the export peak against ~5GB free disk when
  this was decided.
```

Then replace every `<RUN: ...>` with the recorded value (rewording the sentence if a stage did not run), and run `grep -n "RUN:" docs/adr/0017-light-model-config-for-learning.md`: expected no output.

- [ ] **Step 2: Update the ADR index**

Append to the table in `docs/adr/README.md`, after the 0016 row:

```markdown
| [0017](0017-light-model-config-for-learning.md) | A Qwen3-0.6B config for learning, with its own data and output folders | Accepted |
```

- [ ] **Step 3: Add one pointer each to README.md and CLAUDE.md**

In `README.md`, after the paragraph that introduces the stage table or the quickstart (find the "Quickstart" or equivalent section heading with `grep -n "^## " README.md`), add:

```markdown
### Learning with a light model

`configs/qwen3-0.6b-personas.yaml` runs the same pipeline on Qwen3-0.6B in minutes
(`make train CONFIG=configs/qwen3-0.6b-personas.yaml`, same for the other stages), keeping its
data and outputs under `data/qwen3-0.6b` and `outputs/qwen3-0.6b`. The guide
[docs/guides/adapting-to-a-new-niche.md](docs/guides/adapting-to-a-new-niche.md) explains each
stage and how to adapt the pipeline to another niche ([ADR 0017](docs/adr/0017-light-model-config-for-learning.md)).
```

In `CLAUDE.md`, under "Read first", add one bullet:

```markdown
- [`docs/guides/adapting-to-a-new-niche.md`](docs/guides/adapting-to-a-new-niche.md) — the pipeline explained stage by stage, the light 0.6B config (ADR 0017), and what is niche-specific
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run --extra dev pytest -q > /tmp/claude-1000/final.log 2>&1; echo "exit $?"; grep -E "passed|failed" /tmp/claude-1000/final.log`
Expected: `exit 0`, all passed. Also run `grep -rn "RUN:" docs/adr/0017-light-model-config-for-learning.md README.md CLAUDE.md`: expected no output.

- [ ] **Step 5: Commit**

```bash
git add docs/adr/0017-light-model-config-for-learning.md docs/adr/README.md README.md CLAUDE.md
git commit -m "docs: record the light-model config as ADR 0017 with measured numbers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** 3.1 config → Task 1; 3.2 guard test → Task 1 (collisions helper, keys, uniqueness); 3.3 report title → Task 2; 3.4 guide (9 sections, including the W&B explanation) → Task 3 (heading, path, W&B content and script-parse tests); 3.5 ADR → Task 5; success criterion 1 and 3 and 2 → Task 4; criterion 4 → every task's pytest run; risks (rm -rf, disk, quality, ADR ordering) → Task 4 steps, ADR consequences, Task 5 ordering after the run.
- **Spec correction:** the spec's test section describes "unique across configs" and permits subfolders; Task 1's `_collisions` implements exactly that (no parent/child ban).
- **Type consistency:** `_collisions(paths_by_config: dict[str, dict[str, str]]) -> list[str]` is used the same way in all three tests; `report_title(base_id: str) -> str` matches its call in the `<h1>`; config keys match `REQUIRED_KEYS`.
- **Known limit:** Task 4's numbers are unknown until it runs; Task 5 refuses to commit if any `RUN:` marker remains.
