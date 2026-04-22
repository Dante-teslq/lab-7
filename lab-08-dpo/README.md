# Lab 08 — Alinhamento Humano com DPO

> Projeto acadêmico desenvolvido por **José Lucas** para a disciplina do iCEV, como continuação do Lab 07 (QLoRA fine-tuning). Aqui pegamos o adapter já treinado e aplicamos **Direct Preference Optimization (DPO)** para alinhar o modelo a preferências humanas — recusar pedidos perigosos, manter tom corporativo e redirecionar para alternativas seguras.

## O que muda do Lab 07 para o Lab 08

| | Lab 07 (SFT/QLoRA) | Lab 08 (DPO) |
|---|---|---|
| Tipo de dado | `{prompt, response}` | `{prompt, chosen, rejected}` |
| Sinal de treino | "copia esse exemplo" | "prefira A em vez de B" |
| Modelos em memória | 1 (ator) | 2 (ator + referência congelado) |
| Hiperparâmetro chave | learning rate, rank LoRA | **β (beta)** — peso da divergência KL |
| Resultado típico | modelo aprende **conhecimento** | modelo aprende **valores/políticas** |

---

## Como executar

### 1. Pré-requisitos
- Python 3.10+
- GPU NVIDIA com CUDA 11.8+ e ~16 GB VRAM (testado em RTX 3090 / A10)
- Adapter LoRA do Lab 07 em `../lora-adapter/` (opcional — se ausente, o script cai pra o modelo base)

### 2. Instalar dependências
```bash
cd lab-08-dpo
pip install -r requirements.txt
```

### 3. Gerar o dataset de preferências
```bash
python generate_dataset.py
```
Isso cria `data/hhh_dataset.jsonl` com 30 exemplos curados manualmente.

### 4. Treinar o DPO
```bash
python train_dpo.py
```
O adapter alinhado é salvo em `./dpo_output/final_model/`.

### 5. Validar com prompts maliciosos
```bash
python inference_test.py
```
Imprime as respostas do modelo para 3 prompts de risco (SQL injection, exfiltração de credenciais, bypass de auth).

---

## Sobre o hiperparâmetro β (beta)

O **β** é o coração do DPO. Ele controla o **peso da divergência KL** entre o modelo ator (π_θ, que estamos treinando) e o modelo de referência (π_ref, congelado) na função de perda. A formulação completa é:

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                    - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

Onde `y_w` é a resposta preferida (*chosen*) e `y_l` é a rejeitada (*rejected*). O termo `log π_θ/π_ref` é um *log-ratio* — ele mede o quanto o ator "se afastou" da referência para aquela resposta.

O β atua como um **"imposto de KL"**: quanto maior, mais caro é para o modelo se afastar do comportamento original.

| β | Comportamento |
|---|---|
| **Alto (≈1.0)** | Ator fica preso à referência. Preserva fluência e estilo do Lab 07, mas o alinhamento de segurança fica fraco — o modelo pode continuar entregando pedidos perigosos. |
| **Baixo (≈0.1)** | Ator pode se afastar livremente. Alinha forte às preferências (recusa robusta), mas pode degradar fluência ou gerar respostas "robóticas" demais. |
| **Muito baixo (<0.05)** | Risco de "reward hacking" — modelo encontra atalhos que maximizam a loss sem realmente aprender o conceito (ex.: começar toda resposta com "Não posso..."). |

**No nosso caso, escolhemos β = 0.1**, que é o valor canônico do paper original (Rafailov et al., 2023). A justificativa para o nosso cenário:

1. O dataset é pequeno (30 exemplos) — um β alto travaria o aprendizado e o alinhamento sequer apareceria nas métricas.
2. O foco do lab é **segurança**, não estilo — uma pequena degradação de fluência é aceitável se a recusa for robusta.
3. O modelo de referência (adapter do Lab 07) já está bem fluente em português técnico, então mesmo com β baixo o ator não perde totalmente a base estilística.

Se em produção o modelo começar a recusar coisas legítimas demais ("over-refusal"), o ajuste é **subir o β** (ex.: 0.2–0.3) e re-treinar.

---

## Nota de uso de IA

Partes geradas/complementadas com IA, revisadas por **Lucas**.

Especificamente:
- A estrutura inicial dos scripts (`generate_dataset.py`, `train_dpo.py`, `inference_test.py`) e os comentários pedagógicos foram desenvolvidos com auxílio de IA.
- A curadoria do dataset (categorias de risco, redação dos pares chosen/rejected, contextualização legal com referências à LGPD/CP) passou por revisão manual para garantir aderência ao tom corporativo e ao contexto brasileiro.
- Os hiperparâmetros (β, lr, rank LoRA, número de épocas) foram escolhidos com base em recomendações do paper original do DPO e validados manualmente para o tamanho do dataset.

---

## Dataset

O arquivo `data/hhh_dataset.jsonl` contém **30 pares de preferência** distribuídos uniformemente em 5 categorias de risco:

| # | Categoria | Exemplos | Tipo de risco |
|---|---|---|---|
| 1 | Segurança de banco de dados | 6 | DROP/DELETE sem critério, SQL injection, EXEC dinâmico |
| 2 | Exfiltração de dados | 6 | Vazamento de PII, credenciais, .env de produção |
| 3 | Tom corporativo inadequado | 6 | Comunicação agressiva, discriminatória, deboche público |
| 4 | Acesso não autorizado | 6 | Bypass de auth, escalada de privilégio, keylogger |
| 5 | Conteúdo prejudicial / ilegal | 6 | Phishing, sonegação, falsificação, deepfake não-consensual |

Cada exemplo segue o schema esperado pelo `DPOTrainer`:

```json
{
  "prompt": "<pedido do usuário>",
  "chosen": "<resposta alinhada — recusa + redirecionamento>",
  "rejected": "<resposta desalinhada — entrega o pedido perigoso>"
}
```

Split utilizado no treino: **90% treino / 10% eval**, com `seed=42` para reprodutibilidade.

---

## Estrutura de arquivos

```
lab-08-dpo/
├── data/
│   └── hhh_dataset.jsonl      # 30 pares de preferência
├── dpo_output/                # criado após o treino
│   └── final_model/           # adapter DPO alinhado
├── train_dpo.py               # pipeline principal de treino
├── generate_dataset.py        # geração offline do dataset
├── inference_test.py          # validação com prompts maliciosos
├── requirements.txt
└── README.md                  # este arquivo
```

---

## Referências

- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Hugging Face TRL — [DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer)
- Lei Geral de Proteção de Dados (LGPD) — [Lei nº 13.709/2018](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
