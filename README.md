# Brazilian persona generator — Qwen3-8B QLoRA on a single consumer GPU

Fine-tunes Qwen3-8B with Unsloth QLoRA on `nvidia/Nemotron-Personas-Brazil`, exports to
GGUF, and evaluates against the base model through llama.cpp. Everything runs on one GPU:
first an RTX 3060 12GB, then an RTX 3070 Ti 8GB, and now, in this fork, a **Radeon RX 7600 8GB** (ROCm 7.1).

On 8GB, training fits with the standard 4-bit checkpoint, batch 1 and in-training
evaluation off, at a 7.1GB peak. See [ADR 0013](docs/adr/0013-target-hardware-rtx-3070-ti.md)
and [ADR 0014](docs/adr/0014-training-qwen3-8b-on-8gb.md).

This tree now targets an RX 7600 8GB with ROCm ([ADR 0015](docs/adr/0015-target-hardware-rx-7600-rocm.md)); the tables below are the NVIDIA runs.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona out.

| | Base Qwen3-8B | Fine-tuned on RTX 3070 Ti 8GB | Fine-tuned on RTX 3060 12GB |
|---|---|---|---|
| **Format validity** | 0% | **99%** | **100%** |
| **Perplexity** (held-out) | 4.74 | **2.80** | **2.77** |
| Training time | — | **4h 26m** | 7h 25m |
| Output | a demographic *report about* the person | a persona *of* the person | a persona *of* the person |

Measured on 200 held-out rows, identical decoding parameters, same Q4_K_M quantisation.
The two runs use slightly different training configurations and test splits, so they are
close rather than identical. Full numbers in [`docs/RESULTS.md`](docs/RESULTS.md); the
two cards side by side in [`docs/GPU-COMPARISON.md`](docs/GPU-COMPARISON.md).

---

## Before and after

Examples from the RTX 3060 run. Both models get the **same prompt** — only the
attributes, nothing else:

```
Nome: Gilcimara Souza | Sexo: Feminino | Idade: 19 | Estado civil: Solteiro
Escolaridade: Fundamental completo e médio incompleto
Ocupação: Ocupação elementar
Município: Rio de Janeiro | Estado: Rio de Janeiro | País: Brasil
```

### ❌ Before — base Qwen3-8B

> **1. Identificação:**
> Gilcimara Souza, uma jovem brasileira de 19 anos, residente no Rio de Janeiro, estado do
> Rio de Janeiro. Solteira e com 19 anos, sua vida está em fase de formação pessoal e profissional.
>
> **2. Características Demográficas:**
> Gilcimara tem 19 anos, está solteira e reside na cidade do Rio de Janeiro. Sua escolaridade
> é fundamental completa e médio incompleto, o que **indica que** ela ainda está em processo de
> educação formal. Sua ocupação é considerada elementar, **sugerindo que** ela está envolvida em
> atividades de baixa complexidade…
>
> **3. Contexto Social e Econômico:**
> Como residente do Rio de Janeiro, Gilcimara vive em uma das cidades mais populosas…

It restates the input as a numbered form and hedges everything it cannot know
("indica que", "sugerindo que"). It reads like a case file, not a person. **None of the
six required sections appear.**

### ✅ After — fine-tuned

> ## Síntese
> Gilcimara Souza é uma jovem carioca prática e sociável que combina trabalho no varejo, fé
> ativa e paixão por música ao vivo, buscando estabilidade financeira enquanto cultiva sua
> criatividade.
>
> ## Perfil profissional
> Gilcimara Souza, 19, trabalha como atendente em um comércio de bairro no Rio, onde utiliza
> seu domínio de caixa e negociação para lidar com o fluxo intenso de clientes, enquanto sua
> praticidade e curiosidade a impulsionam a buscar uma posição de supervisão…
>
> ## Origem cultural
> Gilcimara nasceu e foi criada no **bairro da Tijuca**, em um lar de classe trabalhadora onde
> a família valorizava o esforço e a solidariedade. Cresceu ouvindo **samba e pagode** nas ruas,
> participando das festas de **Carnaval e dos blocos de rua**… nos fins de semana costuma
> reunir-se com amigos na **praia de Copacabana** ou no **bar da Lapa**, onde a música ao vivo anima…

Correct structure, confident prose, and invented-but-plausible regional detail — Tijuca,
samba, blocos de rua, Lapa. It committed to a specific person.

### 📋 What the dataset actually contains (ground truth)

> ## Síntese
> Gilcimara Souza, 19 anos, combina energia sociável, organização prática e fé católica ao
> trabalhar no varejo, curtir a praia, criar artesanato e planejar seu futuro empreendedor.

The fine-tune landed in the right register — retail work, faith, beach, creative
side-interests — on a row it had never seen.

<details>
<summary><b>Second example — Pedro Joaquim Alves, 52, construction worker, Eunápolis/BA</b></summary>

```
Nome: Pedro Joaquim Alves | Sexo: Masculino | Idade: 52 | Estado civil: Solteiro
Escolaridade: Médio completo e superior incompleto
Ocupação: Trabalhador qualificado, operário ou artesão da construção…
Município: Eunápolis | Estado: Bahia | País: Brasil
```

**❌ Before:**

> **1. Identificação:**
> Pedro Joaquim Alves, um brasileiro masculino de 52 anos, solteiro, residente na cidade de
> Eunápolis, estado da Bahia…
>
> **2. Características Demográficas:**
> Pedro pertence à geração que viveu a transição para uma sociedade mais moderna… A idade de
> 52 anos o **posiciona como** um adulto maduro… Sua localização no interior do estado da Bahia
> **indica uma possível ligação** com a cultura e tradições rurais ou semiurbanas.

**✅ After:**

> ## Síntese
> Pedro Joaquim Alves, 52 anos, é um mestre da construção que combina disciplina e
> criatividade para liderar obras sustentáveis em Eunápolis, enquanto cultiva a fé católica,
> o futebol de várzea e o cuidado com a família.
>
> ## Perfil profissional
> Pedro Joaquim Alves, 52 anos, é mestre da construção civil que combina sua experiência em
> alvenaria, carpintaria e soldagem com um planejamento meticuloso e liderança assertiva,
> buscando coordenar obras sustentáveis e abrir sua própria empresa de reformas em Eunápolis.

</details>

Thirty more side-by-side comparisons are generated into `outputs/eval/report.html` by
`make report`.

---

## Process metrics

Measured end to end on each card.

| Stage | Command | RTX 3070 Ti 8GB | RTX 3060 12GB | Produces |
|---|---|---|---|---|
| Environment + llama.cpp HIP build | `make setup` | ~25 min* | ~25 min* | venv + `llama-server`/`quantize`/`perplexity` |
| Data preparation | `make data` | ~2 min* | ~3 min* | 10k/500/200 JSONL splits |
| **Fine-tuning (QLoRA)** | `make train` | **4h 26m** | **7h 25m** | 344 MB LoRA adapter |
| GGUF export (per model) | `make export` | **6 min 13 s** | **6 min 26 s** | 4.7 GB Q4_K_M + 8.1 GB Q8_0 |
| Evaluation (200 rows × 2 models) | `make eval` | **29 min 25 s** | **46 min 19 s** | `results.json`, `generations.json` |
| Report | `make report` | < 1 s | < 1 s | `report.html` |
| **Total** | | **≈ 5h 30m** | **≈ 9 h** | |

<sub>* approximate — dominated by downloads (torch/CUDA wheels, 15.3 GB base model, dataset shards) and therefore bandwidth-dependent. Starred rows are wall clock on a first run; everything else is measured precisely from logs.</sub>

### Training detail

| | RTX 3070 Ti 8GB | RTX 3060 12GB |
|---|---|---|
| Rows / epochs | 10,000 / 1 | 10,000 / 1 |
| Optimizer steps | 625 (1 × 16 grad-accum) | 625 (2 × 8 grad-accum) |
| 4-bit weights | standard bnb (5.66 GB) | Unsloth dynamic (6.97 GB) |
| `max_seq_length` | 1536 | 2048 |
| Throughput | **25.6 s/step** | **42.7 s/step**, ≈ 570 tokens/s |
| Peak VRAM | **7.10 GB** of 7.65 GB | **9.07 GB** of 11.63 GB |
| Trainable params | 87,293,952 (**1.05 %**) | 87,293,952 (**1.05 %**) |
| In-training validation | off — it OOMs on 8GB | 5 passes × 150 rows, ~2 min each |
| Train loss | 1.72 → 0.84 | 1.72 → 0.91 (run mean) |

On the 3060, the card is the bottleneck, not the code: at 4-bit with gradient checkpointing the step
time is dominated by memory bandwidth. An initial estimate of 3-5 h proved **2.4× optimistic**;
the schedule was re-derived from a measured smoke run — see
[ADR 0012](docs/adr/0012-revised-training-budget.md), which supersedes the guess.

On the 3070 Ti the same step takes 40 % less time, in line with its higher memory bandwidth
(608 vs 360 GB/s) and compute.

### Inference / evaluation detail

| | 3070 Ti base | 3070 Ti fine-tuned | 3060 base | 3060 fine-tuned |
|---|---|---|---|---|
| 200 generations | 666 s | 1001 s | 1050 s | 1610 s |
| Per row | 3.3 s | 5.0 s | 5.3 s | 8.1 s |
| Mean output | 2072 chars | 2998 chars | 2093 chars | 3072 chars |

The tuned model takes ~50 % longer per row because it writes ~50 % more text — it fills all
six sections instead of stopping early. Single-stream generation on the 3060 runs at
**~58 tok/s** with full GPU offload (`-ngl 99`).

Base and tuned are evaluated **sequentially, never concurrently**: two Q4_K_M 8B models need
~10 GB plus KV cache and fit on neither card together
([ADR 0010](docs/adr/0010-evaluation-strategy.md)).

---

## Quickstart

```bash
make setup    # uv venv (Python 3.12) + HIP build of llama.cpp
make smoke    # 200-row end-to-end check — run this first
make data     # 10k/500/200 splits
make train    # 4h 26m QLoRA on the 3070 Ti (measured)
make export   # GGUF for tuned and base
make eval     # sequential base-vs-tuned scoring
make report   # outputs/eval/report.html
```

## Requirements

An AMD RX 7600 8GB with ROCm 7.1 (~8GB visible). Training peak and serving VRAM on this card are not yet measured: the first smoke run hit an OOM at step 1 with the desktop resident (~1.28GB), so train from a TTY or with the browser closed ([ADR 0015](docs/adr/0015-target-hardware-rx-7600-rocm.md)). 30GB RAM · 60GB free disk · a C++ compiler.

Host Python is not used — `uv` manages a project-local 3.12 environment, and `cmake`
is installed into it. If the host has the ROCm 7.1.1 runtime but no compiler, `make setup`
downloads AMD's HIP packages into `vendor/rocm` (~4GB, no root), so the only system package
needed is a C++ compiler.

## Pinning llama.cpp

The commit built and verified for this fork is recorded in
`configs/qwen3-8b-personas.yaml` (`llamacpp.commit`). Export it so builds stay
reproducible:

```bash
export LLAMA_COMMIT=bdeb855b30dfe7f6e695cba98445a7ba09e6416e
```

## If training runs out of memory

The config already uses batch 1, `max_seq_length` 1536 and no in-training evaluation
(ADR 0014). What is left is freeing the GPU: close GPU-accelerated apps, or stop the
desktop session (`sudo systemctl isolate multi-user.target`).

## Documentation

- `docs/superpowers/specs/2026-09-12-persona-finetune-design.md` — the design
- `docs/superpowers/plans/2026-09-12-persona-finetune.md` — the implementation plan
- `docs/RESULTS.md` — measured results of both runs
- `docs/GPU-COMPARISON.md` — RTX 3060 12GB vs RTX 3070 Ti 8GB, side by side
- `docs/adr/README.md` — why each decision was made
- `CLAUDE.md` — invariants that must not be broken

## Data

`nvidia/Nemotron-Personas-Brazil` is CC-BY-4.0 and fully synthetic. No real PII.
