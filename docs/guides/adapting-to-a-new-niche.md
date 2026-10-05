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
