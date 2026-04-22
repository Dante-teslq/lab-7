"""
train_dpo.py
------------
Lab 08 — Alinhamento Humano com DPO (José Lucas - estudante).

Pipeline principal de Direct Preference Optimization, usando o adaptador
LoRA do Lab 07 como ponto de partida.

Diferença conceitual entre os labs:
  - Lab 07 (SFT/QLoRA): "olha esses exemplos, copia o estilo".
  - Lab 08 (DPO):       "entre essas duas respostas, prefira a primeira".

Por que DPO em vez de RLHF clássico?
  RLHF tradicional precisa de 3 modelos (policy, reward model, reference)
  e um loop de PPO bem complicado de estabilizar. O paper do DPO (Rafailov
  et al., 2023) mostra que dá pra eliminar o reward model resolvendo a
  otimização em forma fechada — sobra só uma loss tipo classificação
  binária. É bem mais simples e estável.

A função de perda do DPO:

  L = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x)
                  - log π_θ(y_l|x)/π_ref(y_l|x)))]

Onde:
  π_θ   = modelo "ator" (que estamos treinando)
  π_ref = modelo de referência (CONGELADO — geralmente o mesmo modelo
          após SFT, no nosso caso o adapter do Lab 07)
  y_w   = resposta "winner" (chosen)
  y_l   = resposta "loser"  (rejected)
  β     = hiperparâmetro de controle (ver README)

Pipeline:
  1. Carrega modelo base em 4-bit (mesma config do Lab 07)
  2. Aplica adapter do Lab 07 -> esse vira o modelo de REFERÊNCIA (frozen)
  3. Cria um segundo PEFT adapter por cima -> esse é o ATOR (vai treinar)
  4. Carrega dataset de preferências (gerado pelo generate_dataset.py)
  5. Roda DPOTrainer com beta=0.1
  6. Salva o adapter alinhado em ./dpo_output/final_model
"""

import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tratamento de erro p/ bitsandbytes — ele só funciona com CUDA, em CPU
# (ex.: rodar pra checagem sintática) o import explode e mata o script.
# Aqui a gente importa "lazy" pra dar uma mensagem mais amigável.
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("AVISO: bitsandbytes/BitsAndBytesConfig indisponível. "
          "O treino DPO completo precisa de GPU NVIDIA com CUDA.")

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


# -----------------------------------------------------------------------------
# Configurações principais (ficam no topo pra facilitar ajuste)
# -----------------------------------------------------------------------------

# Mesmo modelo base do Lab 07 — assim o adapter LoRA encaixa perfeitamente.
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"

# Caminhos candidatos pro adapter do Lab 07. O script tenta o primeiro que
# existir; se nenhum existir, usa o modelo base puro como referência (com aviso).
LAB07_ADAPTER_CANDIDATES = [
    "../lora-adapter",          # caminho relativo, considerando rodar de lab-08-dpo/
    "../lab-7/lora-adapter",    # caso lab-08 esteja em outro lugar
    "./lab07_adapter",          # caso o usuário tenha copiado pra cá
]

# Onde fica o dataset de preferências
DATASET_FILE = os.path.join(os.path.dirname(__file__), "data", "hhh_dataset.jsonl")

# Onde salvar o adapter alinhado
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dpo_output")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")


def find_lab07_adapter():
    """
    Procura o adapter do Lab 07 nos caminhos conhecidos.
    Retorna o caminho absoluto ou None se não encontrar.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in LAB07_ADAPTER_CANDIDATES:
        candidate = os.path.normpath(os.path.join(base_dir, rel))
        # adapter_config.json é o arquivo que o PEFT sempre cria
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return candidate
    return None


def build_quantization_config():
    """
    Mesma config 4-bit do Lab 07. Encapsulada pra reaproveitar em ator e ref.

    bnb_4bit_quant_type="nf4"          -> NormalFloat4 (paper QLoRA)
    bnb_4bit_compute_dtype=float16     -> contas em fp16 (mais rápido)
    bnb_4bit_use_double_quant=True     -> economiza ainda mais VRAM
    """
    if not BNB_AVAILABLE:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(quant_config, has_gpu):
    """Carrega o modelo base do HuggingFace, aplicando quantização se houver GPU."""
    print(f"[1/6] Carregando modelo base: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config if has_gpu else None,
        device_map="auto" if has_gpu else None,
        trust_remote_code=True,
    )
    # Mesmas flags que usamos no Lab 07.
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def load_tokenizer():
    """Tokenizer do Llama-2 — mesma config do Lab 07 pra manter compatibilidade."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Llama-2 não tem pad_token nativo; reusar EOS é o padrão da comunidade.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main():
    # -------------------------------------------------------------------------
    # 0. Checagens iniciais
    # -------------------------------------------------------------------------
    has_gpu = torch.cuda.is_available()
    print(f"CUDA disponível: {has_gpu}")
    if not has_gpu:
        print("AVISO: sem GPU. O treino DPO real precisa de GPU NVIDIA — "
              "este script ainda valida sintaxe e estrutura, mas não vai "
              "conseguir rodar o trainer.train() sem CUDA + bitsandbytes.")

    if not os.path.isfile(DATASET_FILE):
        print(f"ERRO: dataset não encontrado em {DATASET_FILE}")
        print("Rode 'python generate_dataset.py' antes deste script.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1-2. Tokenizer + Modelo base
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer()
    quant_config = build_quantization_config()
    base_model = load_base_model(quant_config, has_gpu)

    if has_gpu:
        # prepare_model_for_kbit_training: habilita gradiente checkpointing
        # e converte LayerNorms pra fp32 (estabilidade numérica em 4-bit).
        base_model = prepare_model_for_kbit_training(base_model)

    # -------------------------------------------------------------------------
    # 3. Modelo de REFERÊNCIA (congelado): base + adapter Lab 07
    # -------------------------------------------------------------------------
    # No DPO, o ref_model é a "âncora" — ele NÃO treina, só é usado pra
    # calcular log π_ref(y|x) na loss. É ele que define o "ponto de partida"
    # do qual o ator pode se afastar (controlado pelo beta).
    print("[2/6] Carregando modelo de referência (Lab 07 adapter, congelado)")

    lab07_path = find_lab07_adapter()
    if lab07_path:
        print(f"  -> Adapter Lab 07 encontrado em: {lab07_path}")
        # is_trainable=False -> congela explicitamente os pesos do adapter
        ref_model = PeftModel.from_pretrained(
            base_model, lab07_path, is_trainable=False
        )
    else:
        print("  -> Nenhum adapter Lab 07 encontrado. Usando o modelo base "
              "puro como referência (fallback). Para um alinhamento mais "
              "forte, treine primeiro o Lab 07 e coloque o adapter em "
              "../lora-adapter/")
        ref_model = base_model

    # eval() desabilita dropout e fixa BatchNorm — boa prática pra modelo
    # que não vai treinar.
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------------------
    # 4. Modelo ATOR: base + NOVO adapter LoRA (esse vai treinar)
    # -------------------------------------------------------------------------
    # OBS importante: a trl >= 0.8 cuida internamente da criação do adapter
    # quando passamos `peft_config` pro DPOTrainer. Então aqui apenas
    # configuramos o LoraConfig — o trainer aplica no actor_model.
    print("[3/6] Configurando LoRA do modelo ATOR (esse vai treinar)")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,                # rank menor que no Lab 07 — DPO é refinamento,
                             # não precisa de tanta capacidade nova
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    actor_model = base_model  # DPOTrainer vai injetar o adapter aqui

    # -------------------------------------------------------------------------
    # 5. Dataset de preferências
    # -------------------------------------------------------------------------
    print(f"[4/6] Carregando dataset: {DATASET_FILE}")
    full_ds = load_dataset("json", data_files=DATASET_FILE, split="train")

    # Split 90/10 train/eval. seed fixa pra resultado reproduzível.
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    print(f"  -> Treino: {len(train_data)} exemplos | Eval: {len(eval_data)} exemplos")

    # -------------------------------------------------------------------------
    # 6. DPOConfig + DPOTrainer
    # -------------------------------------------------------------------------
    # beta=0.1 é o valor "canônico" do paper original de DPO. Ver README pra
    # discussão completa do trade-off (alinhamento x fluência).
    print("[5/6] Configurando DPOTrainer (beta=0.1)")

    # bf16 só funciona em GPUs Ampere+ (A100, RTX 30xx pra cima). Em GPUs
    # mais antigas (T4, V100) cai pro fp16. Detecção:
    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = has_gpu and not use_bf16

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,                                 # KL "tax" — ver README
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,            # batch efetivo = 4
        learning_rate=5e-5,                       # menor que SFT (refino)
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_32bit",                # bitsandbytes paged optim
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,              # DPO usa colunas chosen/rejected
        report_to="none",                         # sem W&B
        max_length=1024,                          # truncamento total
        max_prompt_length=512,                    # truncamento só do prompt
    )

    trainer = DPOTrainer(
        model=actor_model,
        ref_model=ref_model if ref_model is not actor_model else None,
        # ref_model=None faz o trainer usar o ator com adapter desabilitado
        # como referência — útil quando ator e ref compartilham os mesmos
        # pesos base (economia de VRAM).
        args=dpo_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # -------------------------------------------------------------------------
    # 7. Treino + salvamento
    # -------------------------------------------------------------------------
    print("[6/6] Iniciando treino DPO...")
    trainer.train()

    print(f"Treino finalizado. Salvando modelo em: {FINAL_MODEL_DIR}")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("OK — modelo alinhado salvo. Use inference_test.py para validar.")


if __name__ == "__main__":
    main()
