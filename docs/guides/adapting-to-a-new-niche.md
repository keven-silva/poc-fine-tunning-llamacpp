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

A config `configs/qwen3-0.6b-personas.yaml` usa o Qwen3-0.6B: cerca de 47 min de treino (~20 s por passo) nesta
placa. Todos os comandos recebem `CONFIG=`:

```
make data   CONFIG=configs/qwen3-0.6b-personas.yaml
make train  CONFIG=configs/qwen3-0.6b-personas.yaml
make export CONFIG=configs/qwen3-0.6b-personas.yaml
make eval   CONFIG=configs/qwen3-0.6b-personas.yaml
make report CONFIG=configs/qwen3-0.6b-personas.yaml
```

**Atenção à memória:** o 0,6B não é leve em VRAM. O pico reservado foi de 7,20 GB, e o total da
placa, com a área de trabalho, chegou a 8,07 GB (nada deu OOM, mas a placa ficou cheia). Treine
com o desktop leve ou a partir de um TTY, como no 8B (`CLAUDE.md`).

O que cada etapa faz e como ler a saída:

1. **`make data`** (`scripts/01_prepare_data.py`): baixa o dataset, monta o texto de cada
   exemplo, descarta os longos demais para `max_seq_length` e grava `train.jsonl`,
   `val.jsonl`, `test.jsonl` e `manifest.json` em `data/qwen3-0.6b/`.
2. **`make train`** (`scripts/02_train.py`): imprime a *loss* a cada 5 passos. Ela deve cair
   rápido no começo e depois estabilizar. Com a avaliação ligada, aparece também a *eval_loss*
   (validação) a cada 25 passos: se a loss de treino cai e a de validação sobe, é overfitting.
   Ao final há uma linha com o tempo e o **pico de VRAM**. O tempo por passo aparece no tqdm
   (`s/it`): é estável e fica mais lento durante as passagens de validação. Cerca de 20 s por
   passo é normal para esta pilha nesta placa.
3. **`make export`** (`scripts/03_export_gguf.py`): funde o adaptador, converte e quantiza.
   Gera os GGUF do modelo *tuned* e do *base* pelo mesmo caminho, para que a comparação meça
   só o LoRA.
4. **`make eval`** (`scripts/05_evaluate.py`): sobe o `llama-server` com um modelo de cada
   vez (`scripts/04_serve.sh`) e mede perplexidade, validade de formato, e se o texto cita os
   atributos de entrada (*grounding*, `src/personas/grounding.py`).
5. **`make report`** (`scripts/06_report.py`): gera uma página com as métricas e exemplos
   lado a lado.

Como ler a tabela do eval: o modelo base tem **validade de formato 0** (não sabe as seis
seções) e a perplexidade do tuned é menor. Medido no 0,6B (200 linhas), base -> tuned:
perplexidade 10,7 -> 6,0; validade de formato 0 -> 0,825; grounding 0,95 -> 0,87. Ou seja, o
0,6B aprende o formato, mas perde precisão nos campos de entrada (estado e ocupação caíram).
Isso é aprendizado: é o que um modelo pequeno faz. Os números completos estão na
`docs/adr/0017-light-model-config-for-learning.md`.

**Limpeza e pastas aninhadas:** `make clean` não alcança `outputs/qwen3-0.6b/`. Os diretórios
`merged-16bit-*` que sobram do export podem ser removidos com
`rm -rf outputs/qwen3-0.6b/merged-16bit-*` (o script de export já apaga o arquivo f16). Cuidado:
`rm -rf data` ou `rm -rf outputs` apaga os arquivos das DUAS configs, porque as pastas do 0,6B
ficam dentro delas.

## 3. O que muda entre o 8B e o 0.6B

| Chave | 8B | 0.6B | Por quê |
|---|---|---|---|
| `model.base_id` / `train_id` | Qwen3-8B | Qwen3-0.6B | o objetivo do exercício |
| `model.max_seq_length` | 1536 | 2048 | o limite do 8B era para caber em 8GB |
| `data.n_train` | 10000 | 2000 | poucas dezenas de minutos em vez de horas |
| batch × acumulação | 1 × 16 | 1 × 16 | idêntico: os logits sobre o vocabulário (151.936 tokens) custam a mesma memória em qualquer tamanho de modelo (medido: batch 4 × 2048 rodou a ~60 s/passo com a VRAM no limite) |
| `train.eval_strategy` | `"no"` | `"steps"` | avaliação ligada de novo: `prediction_loss_only` e batch 1 mantêm o custo baixo |
| `train.logging_steps` / `eval_steps` / `save_steps` / `save_total_limit` | 10 / 100 / 250 / 3 | 5 / 25 / 100 / 1 | só 125 passos e pouco disco; não afeta o modelo |
| `paths.*` | `data`, `outputs` | `data/qwen3-0.6b`, `outputs/qwen3-0.6b` | não sobrescrever o 8B |

Todo o resto é idêntico (LoRA, learning rate, decodificação, sementes). Assim, comparar os dois
isola o efeito do tamanho. Para subir para 1,7B ou 4B, edite esses quatro (`model.base_id`,
`model.train_id` e os dois caminhos) e meça de novo os segundos por passo e a VRAM antes de uma
rodada completa: o formato de batch do 0,6B não foi validado para modelos maiores. As estimativas
de disco estão no topo do YAML.

O 0,6B também não é leve em memória: pico de 7,20 GB reservados e 8,07 GB no total da placa com
o desktop. Treine com o desktop leve ou de um TTY, como no 8B.

## 4. Mapa do código: o que é do nicho e o que não é

**Independente do nicho** (reaproveite como está):

- o laço de treino e o LoRA: `scripts/02_train.py` (o laço não depende do nicho, mas o script
  importa `PROMPT_VERSION` e lê os JSONL de persona)
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
   depois, se quiser, com `wandb sync wandb/offline-run-*`. Ao adotar, adicione `wandb/` ao
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
